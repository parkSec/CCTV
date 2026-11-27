"""학습 없는 전통 저조도 개선 기법 6개. 입출력은 BGR uint8 (base.py 계약).

- HE     : 밝기 채널 히스토그램 평활화
- AGCWD  : Huang, Cheng, Chiu, IEEE TIP 22(3), 2013
- MSRCR  : Jobson, Rahman, Woodell, IEEE TIP 6(7), 1997
- LIME   : Guo, Li, Ling, IEEE TIP 26(2), 2017
- Dong   : Dong et al., ICME 2011 (반전 → 안개 제거 → 반전)
- BIMEF  : Ying, Li, Gao, arXiv 1711.00591, 2017

논문 그대로 옮기지 않은 부분은 각 클래스 설명에 적었다.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import cg

from .base import PassThroughEnhancer

_EPS = 1e-3


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- HE
class HistEqEnhancer(PassThroughEnhancer):
    """전역 히스토그램 평활화. CLAHE와 같게 LAB의 L 채널에만 적용해 색은 유지."""

    def __init__(self):
        self.name = "he"
        self.params = {}

    def infer(self, img: np.ndarray) -> np.ndarray:
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        return cv2.cvtColor(cv2.merge([cv2.equalizeHist(l), a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- AGCWD
def agcwd_lut(v: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """이미지 밝기 분포로 감마 곡선을 만든다 (논문 식 (6)~(8)).

    pdf_w(l) = pdf_max * ((pdf(l) - pdf_min) / (pdf_max - pdf_min))^alpha
    cdf_w(l) = sum(pdf_w[0..l]) / sum(pdf_w)
    T(l)     = 255 * (l/255)^(1 - cdf_w(l))
    """
    hist = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    pdf = hist / hist.sum()
    pmin, pmax = pdf.min(), pdf.max()
    if pmax - pmin < 1e-12:
        pdf_w = np.ones(256)
    else:
        pdf_w = pmax * ((pdf - pmin) / (pmax - pmin)) ** alpha
    cdf_w = np.cumsum(pdf_w) / pdf_w.sum()
    levels = np.arange(256) / 255.0
    lut = np.round(255.0 * levels ** (1.0 - cdf_w))
    lut = np.maximum.accumulate(np.clip(lut, 0, 255))   # 반올림으로 생길 수 있는 역전 방지
    return lut.astype(np.uint8)


class AgcwdEnhancer(PassThroughEnhancer):
    """HSV의 V 채널에 이미지별 감마 곡선을 적용. 영상용 프레임 간 재사용(논문 3.3절)은 쓰지 않음(이미지 단위 평가)."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.name = f"agcwd_a{self.alpha}"
        self.params = {"alpha": self.alpha}

    def infer(self, img: np.ndarray) -> np.ndarray:
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        v2 = cv2.LUT(v, agcwd_lut(v, self.alpha))
        return cv2.cvtColor(cv2.merge([h, s, v2]), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------- MSRCR
def _blur(x: np.ndarray, sigma: float, small_sigma: float = 16.0) -> np.ndarray:
    """가우시안 블러. sigma가 크면 sigma/small_sigma 배 줄여서 블러하고 다시 키운다.

    sigma 250을 원래 크기에서 하면 커널 폭이 약 1500이라 1280px 한 장에 수 초가 걸린다.
    큰 sigma의 블러 결과는 아주 부드러워서 줄였다 키워도 차이가 작다 (테스트: 최대 오차 3% 미만).
    """
    if sigma <= small_sigma:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    f = sigma / small_sigma
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), small_sigma), (w, h), interpolation=cv2.INTER_LINEAR)


class MsrcrEnhancer(PassThroughEnhancer):
    """Multi-Scale Retinex with Color Restoration.

    MSR_i = mean_s [log I_i - log(G_s * I_i)],  s = 15, 80, 250
    C_i   = beta * [log(alpha * I_i) - log(sum_c I_c)],  alpha=125, beta=46
    출력  = 채널마다 MSR_i * C_i 를 하위 1%~상위 1% 기준으로 0~255에 늘림.
    논문의 이득·오프셋(G, b)은 마지막 늘림에서 상쇄되므로 따로 두지 않았다.
    """

    def __init__(self, sigmas=(15, 80, 250), alpha: float = 125.0, beta: float = 46.0, clip: float = 1.0):
        self.sigmas = tuple(float(s) for s in sigmas)
        self.alpha, self.beta, self.clip = float(alpha), float(beta), float(clip)
        self.name = "msrcr"
        self.params = {"sigmas": list(self.sigmas), "alpha": self.alpha, "beta": self.beta, "clip_pct": self.clip}

    def infer(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) + 1.0
        logx = np.log(x)
        msr = np.zeros_like(x)
        for s in self.sigmas:
            msr += logx - np.log(_blur(x, s))
        msr /= len(self.sigmas)
        crf = self.beta * (np.log(self.alpha * x) - np.log(x.sum(axis=2, keepdims=True)))
        y = msr * crf
        out = np.empty_like(y)
        for c in range(3):
            lo, hi = np.percentile(y[..., c], [self.clip, 100.0 - self.clip])
            out[..., c] = (y[..., c] - lo) / max(hi - lo, 1e-9)
        return _to_uint8(out)


# ---------------------------------------------------------------- 조명 지도 다듬기 (LIME, BIMEF 공통)
def _diff_ops(h: int, w: int):
    def d(m):
        if m == 1:
            return sp.csr_matrix((1, 1))
        D = sp.diags([-np.ones(m), np.ones(m - 1)], [0, 1], shape=(m, m), format="lil")
        D[m - 1, m - 1] = 0     # 마지막 칸은 차분 없음
        return D.tocsr()
    return sp.kron(sp.eye(h), d(w), format="csr"), sp.kron(d(h), sp.eye(w), format="csr")


def _weights(t_hat: np.ndarray, kind: str, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    gh = np.diff(t_hat, axis=1, append=t_hat[:, -1:])
    gv = np.diff(t_hat, axis=0, append=t_hat[-1:, :])
    sgh = cv2.GaussianBlur(gh, (0, 0), sigma)
    sgv = cv2.GaussianBlur(gv, (0, 0), sigma)
    if kind == "lime":   # LIME 논문 식 (23) 전략 II
        return 1.0 / (np.abs(sgh) + _EPS), 1.0 / (np.abs(sgv) + _EPS)
    if kind == "bimef":  # BIMEF 공개 코드 computeTextureWeights
        return 1.0 / (np.abs(sgh) * np.abs(gh) + _EPS), 1.0 / (np.abs(sgv) * np.abs(gv) + _EPS)
    raise ValueError(kind)


def refine_illumination(t_hat: np.ndarray, alpha: float, weight: str = "lime", sigma: float = 2.0,
                        tol: float = 1e-4, maxiter: int = 1000) -> np.ndarray:
    """(I + alpha * sum_d D_d^T diag(W_d) D_d) t = t_hat 을 푼다 (LIME 식 (19)).

    직접 풀기(spsolve)는 1280×720에서 20초 이상 걸려서, 행렬이 대칭 양의 정부호인 점을 이용해
    대각 전처리 켤레기울기법(CG)으로 상대 오차 tol까지 푼다.
    """
    if alpha == 0:
        return t_hat.astype(np.float64).copy()
    h, w = t_hat.shape
    Dh, Dv = _diff_ops(h, w)
    wh, wv = _weights(t_hat.astype(np.float64), weight, sigma)
    A = sp.eye(h * w, format="csr") + alpha * (Dh.T @ sp.diags(wh.ravel()) @ Dh + Dv.T @ sp.diags(wv.ravel()) @ Dv)
    b = t_hat.ravel().astype(np.float64)
    M = sp.diags(1.0 / A.diagonal())
    x, _ = cg(A, b, x0=b.copy(), rtol=tol, maxiter=maxiter, M=M)
    return x.reshape(h, w)


# ---------------------------------------------------------------- LIME
class LimeEnhancer(PassThroughEnhancer):
    """조명 지도 T: 픽셀별 R·G·B 최댓값 → 구조 보존 다듬기 → T^gamma, 결과 = I / T.

    논문 마지막 단계의 BM3D 잡음 제거는 넣지 않았다 (느리고, 탐지 전처리 비교에서는 밝기 보정 효과만 본다).
    """

    def __init__(self, alpha: float = 0.15, gamma: float = 0.8, sigma: float = 2.0):
        self.alpha, self.gamma, self.sigma = float(alpha), float(gamma), float(sigma)
        self.name = "lime"
        self.params = {"alpha": self.alpha, "gamma": self.gamma, "sigma": self.sigma}

    def infer(self, img: np.ndarray) -> np.ndarray:
        x = _to_float(img)
        t = refine_illumination(x.max(axis=2), self.alpha, "lime", self.sigma)
        t = np.clip(t, _EPS, 1.0) ** self.gamma
        return _to_uint8(x / np.maximum(t, 0.05)[..., None])


# ---------------------------------------------------------------- Dong
def _dark_channel(x: np.ndarray, patch: int) -> np.ndarray:
    return cv2.erode(x.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He, Sun, Tang, Guided Image Filtering (ECCV 2010 / TPAMI 2013)의 회색조 버전."""
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda a: cv2.boxFilter(a, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT)
    mi, ms = box(guide), box(src)
    cov = box(guide * src) - mi * ms
    var = box(guide * guide) - mi * mi
    a = cov / (var + eps)
    b = ms - a * mi
    return box(a) * guide + box(b)


class DongEnhancer(PassThroughEnhancer):
    """반전 영상 R = 1 - I 에 He 외(2009)의 dark channel 안개 제거를 적용하고 다시 반전.

    대기광 A: dark channel 상위 0.1% 픽셀 중 가장 밝은 픽셀
    t = 1 - omega * dark(R / A),  J = (R - A) / (P * t) + A,  P = 2t (t < 0.5), 1 (그 외)
    t는 guided filter로 다듬는다 (He 외 안개 제거의 투과율 다듬기 단계. 없으면 물체 가장자리에 네모 얼룩이 생김).
    영상용 프레임 간 매개변수 재사용은 쓰지 않음 (이미지 단위 평가).
    """

    def __init__(self, omega: float = 0.8, patch: int = 7, t_min: float = 0.1, gf_radius: int = 30, gf_eps: float = 1e-3):
        self.omega, self.patch, self.t_min = float(omega), int(patch), float(t_min)
        self.gf_radius, self.gf_eps = int(gf_radius), float(gf_eps)
        self.name = "dong"
        self.params = {"omega": self.omega, "patch": self.patch, "t_min": self.t_min,
                       "gf_radius": self.gf_radius, "gf_eps": self.gf_eps}

    def infer(self, img: np.ndarray) -> np.ndarray:
        r = 1.0 - _to_float(img)
        dark = _dark_channel(r, self.patch)
        n = max(1, int(dark.size * 0.001))
        idx = np.argpartition(dark.ravel(), -n)[-n:]
        cand = r.reshape(-1, 3)[idx]
        A = np.maximum(cand[np.argmax(cand.sum(axis=1))], _EPS)
        t = 1.0 - self.omega * _dark_channel(r / A, self.patch)
        t = _guided_filter(r.mean(axis=2), t, self.gf_radius, self.gf_eps)
        t = np.clip(t, self.t_min, 1.0)
        pt = np.where(t < 0.5, 2.0 * t * t, t)          # P*t
        j = (r - A) / pt[..., None] + A
        return _to_uint8(1.0 - np.clip(j, 0.0, 1.0))


# ---------------------------------------------------------------- BIMEF
_BTF_A, _BTF_B = -0.3293, 1.1258   # 논문의 카메라 응답 모델 매개변수


def bimef_btf(p: np.ndarray, k: float) -> np.ndarray:
    """노출을 k배 했을 때의 밝기 g(P, k) = e^{b(1 - k^a)} * P^{k^a}."""
    ka = k ** _BTF_A
    return np.exp(_BTF_B * (1.0 - ka)) * np.power(p, ka)


def _entropy(v: np.ndarray) -> float:
    hist = np.bincount(np.clip(np.round(v * 255), 0, 255).astype(int), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


class BimefEnhancer(PassThroughEnhancer):
    """조명 지도로 가중치 W = T^mu 를 만들고, 엔트로피가 가장 큰 노출 비율 k로 만든 사진과 섞는다.

    공개 MATLAB 코드처럼 조명 지도는 절반 크기에서 다듬고 원래 크기로 늘린다.
    """

    def __init__(self, mu: float = 0.5, lam: float = 0.5, sigma: float = 5.0, k_range=(1.0, 7.0)):
        self.mu, self.lam, self.sigma = float(mu), float(lam), float(sigma)
        self.k_range = (float(k_range[0]), float(k_range[1]))
        self.name = "bimef"
        self.params = {"mu": self.mu, "lambda": self.lam, "sigma": self.sigma, "k_range": list(self.k_range)}

    def _best_k(self, x: np.ndarray, t: np.ndarray) -> float:
        y = np.cbrt(np.prod(np.maximum(x, 0.0), axis=2))   # 세 채널 기하평균 밝기
        bad = y[t < 0.5]
        if bad.size < 10:
            return 1.0
        bad = bad[:: max(1, bad.size // 2500)]
        res = minimize_scalar(lambda k: -_entropy(np.clip(bimef_btf(bad, k), 0, 1)),
                              bounds=self.k_range, method="bounded")
        return float(res.x)

    def infer(self, img: np.ndarray) -> np.ndarray:
        x = _to_float(img)
        h, w = x.shape[:2]
        t_b = x.max(axis=2)
        small = cv2.resize(t_b, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        t = cv2.resize(refine_illumination(small, self.lam, "bimef", self.sigma), (w, h), interpolation=cv2.INTER_LINEAR)
        t = np.clip(t, _EPS, 1.0)
        k = self._best_k(x, t)
        wgt = (t ** self.mu)[..., None]
        j = np.clip(bimef_btf(np.maximum(x, 1e-6), k), 0.0, 1.0)
        return _to_uint8(x * wgt + j * (1.0 - wgt))
