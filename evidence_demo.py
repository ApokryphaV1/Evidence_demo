"""
Interactive Laplace Approximation Demo (Improved)
-------------------------------------------------
Streamlit web app that:
  • Simulates data from a sine model with user-controlled parameters (A, B, C, D, σ, n, seed).
  • Fits quadratic, cubic, 7th-order polynomial, and sine models (nonlinear) via MLE/MAP.
  • Uses safer initialization (FFT-based frequency guess) and bounded, robust optimization for the sine model.
  • Computes Laplace-approximated log marginal likelihoods with an adaptive finite-difference Hessian and optional ridge regularization.
  • Displays pairwise log Bayes factors and interactive plots.

Run locally:
    pip install streamlit numpy scipy matplotlib pandas
    streamlit run Interactive_Sine_vs_Polynomial_Laplace_App.py

"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import slogdet
from scipy.optimize import curve_fit, least_squares
import streamlit as st

# =============================================================
# Core model & helpers
# =============================================================

def sine_model(x: np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
    return A * np.sin(B * x + C) + D


def _neg_log_post_common(theta: np.ndarray, model_vals: np.ndarray, y: np.ndarray,
                          sigma: float, prior_std: float) -> float:
    n = y.size
    # Gaussian likelihood
    ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - np.sum((y - model_vals) ** 2) / (2 * sigma**2)
    # Independent Gaussian prior N(0, prior_std^2)
    lp = -len(theta) * np.log(np.sqrt(2 * np.pi) * prior_std) - np.sum(theta ** 2) / (2 * prior_std ** 2)
    return -(ll + lp)


def neg_log_post_poly(theta, x, y, sigma, prior_std):
    a, b, c = theta
    model = a * x**2 + b * x + c
    return _neg_log_post_common(theta, model, y, sigma, prior_std)


def neg_log_post_cubic(theta, x, y, sigma, prior_std):
    a, b, c, d = theta
    model = a * x**3 + b * x**2 + c * x + d
    return _neg_log_post_common(theta, model, y, sigma, prior_std)


def neg_log_post_poly7(theta, x, y, sigma, prior_std):
    model = np.polyval(theta, x)
    return _neg_log_post_common(theta, model, y, sigma, prior_std)


def neg_log_post_sine(theta, x, y, sigma, prior_std):
    A, B, C, D = theta
    model = sine_model(x, A, B, C, D)
    return _neg_log_post_common(theta, model, y, sigma, prior_std)


# -------------------------
# Adaptive finite-difference Hessian with ridge
# -------------------------

def compute_hessian(fun, theta, eps_scale=1e-4, ridge=0.0):
    """Compute Hessian via central finite differences with parameter-wise step sizes.
    fun(theta) -> scalar.  ridge adds ridge*I for numerical stability.
    """
    theta = np.array(theta, dtype=float)
    d = theta.size
    H = np.zeros((d, d))
    # Adaptive epsilon per coordinate
    eps = eps_scale * np.maximum(np.abs(theta), 1.0) + 1e-8
    for i in range(d):
        for j in range(d):
            ei, ej = eps[i], eps[j]
            th = theta.copy(); th[i] += ei; th[j] += ej; f_ijp = fun(th)
            th = theta.copy(); th[i] += ei; th[j] -= ej; f_ijm = fun(th)
            th = theta.copy(); th[i] -= ei; th[j] += ej; f_jip = fun(th)
            th = theta.copy(); th[i] -= ei; th[j] -= ej; f_jim = fun(th)
            H[i, j] = (f_ijp - f_ijm - f_jip + f_jim) / (4 * ei * ej)
    if ridge > 0.0:
        H += ridge * np.eye(d)
    return H


def laplace_approx(neg_log_post_fun, theta_hat, x, y, sigma, prior_std,
                   eps_scale=1e-4, ridge=1e-8):
    """Return log marginal likelihood via Laplace approximation.
    Adds ridge to Hessian if needed for PD.
    """
    nlp = neg_log_post_fun(theta_hat, x, y, sigma, prior_std)
    d = len(theta_hat)

    def wrapper(th):
        return neg_log_post_fun(th, x, y, sigma, prior_std)

    H = compute_hessian(wrapper, theta_hat, eps_scale=eps_scale, ridge=ridge)
    sign, logdet = slogdet(H)
    if sign <= 0:
        # try increasing ridge to enforce PD
        H = compute_hessian(wrapper, theta_hat, eps_scale=eps_scale, ridge=max(ridge*10, 1e-6))
        sign, logdet = slogdet(H)
        if sign <= 0:
            raise ValueError("Hessian not positive definite even after ridge regularization.")
    log_marg = -nlp + 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet
    return log_marg


# =============================================================
# Sine model fitting helpers (robust & bounded)
# =============================================================

def fft_guess_B(x, y):
    """Crude frequency guess using FFT peak (ignoring zero freq)."""
    dx = x[1] - x[0]
    freqs = np.fft.rfftfreq(x.size, d=dx)
    Yf = np.abs(np.fft.rfft(y - y.mean()))
    if Yf.size <= 1:
        return 1.0
    idx = np.argmax(Yf[1:]) + 1  # skip the zero frequency
    B0 = 2 * np.pi * freqs[idx]
    return max(B0, 1e-3)


def fit_sine_nonlin(x, y, A0=None, B0=None, C0=0.0, D0=None,
                    bounds=((0, 1e-3, -2*np.pi, -np.inf), (np.inf, 50, 2*np.pi, np.inf))):
    """
    Use curve_fit with bounds + decent initial guesses. Falls back to least_squares with robust loss if needed.
    Returns params, yhat
    """
    if B0 is None:
        B0 = fft_guess_B(x, y)
    if D0 is None:
        D0 = np.mean(y)
    if A0 is None:
        A0 = np.std(y) * np.sqrt(2)

    p0 = [A0, B0, C0, D0]

    try:
        params, _ = curve_fit(sine_model, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return params, sine_model(x, *params)
    except Exception:
        # robust fallback
        def resid(p):
            return sine_model(x, *p) - y
        res = least_squares(resid, p0, bounds=bounds, loss='soft_l1')
        return res.x, sine_model(x, *res.x)


# =============================================================
# Streamlit UI
# =============================================================

st.set_page_config(page_title="Laplace Approximation Demo", layout="wide")
st.title("Interactive Laplace Approximation: Sine vs Polynomial Models (Improved)")

with st.sidebar:
    st.header("Simulation Controls")
    seed = st.number_input("Random Seed", value=42, step=1)
    n = st.slider("Number of data points (n)", min_value=50, max_value=10000, value=1000, step=50)
    x_max = st.slider("x max (range is 0 to x_max)", min_value=2.0, max_value=50.0, value=float(2 * np.pi), step=0.5)
    sigma = st.slider("Noise std (σ)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)

    st.subheader("True Sine Parameters")
    true_A = st.slider("A (amplitude)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    true_B = st.slider("B (frequency)", min_value=0.1, max_value=30.0, value=4.0, step=0.1)
    true_C = st.slider("C (phase shift)", min_value=-np.pi, max_value=np.pi, value=0.0, step=0.1)
    true_D = st.slider("D (offset)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

    st.subheader("Prior Settings")
    prior_std = st.slider("Prior std (Gaussian, mean=0)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

    st.subheader("Laplace / Hessian Settings")
    eps_scale = st.slider("Finite diff eps scale", min_value=1e-6, max_value=1e-2, value=1e-4, step=1e-6, format="%.0e")
    ridge = st.slider("Hessian ridge", min_value=0.0, max_value=1e-3, value=1e-8, step=1e-8, format="%.0e")

    st.markdown("---")
    show_quadratic = st.checkbox("Fit Quadratic", value=True)
    show_cubic = st.checkbox("Fit Cubic", value=True)
    show_poly7 = st.checkbox("Fit 7th Order Polynomial", value=True)
    show_sine = st.checkbox("Fit Sine", value=True)

# ------------- Data Generation -------------
np.random.seed(seed)
x = np.linspace(0, x_max, int(n))
y_true = sine_model(x, true_A, true_B, true_C, true_D)
y = y_true + np.random.normal(scale=sigma, size=int(n))

# ------------- Model Fitting (MLE/MAP) -------------
results = []
fits = {}

if show_quadratic:
    coeffs_poly = np.polyfit(x, y, 2)
    y_poly = np.polyval(coeffs_poly, x)
    log_marg_poly = laplace_approx(neg_log_post_poly, coeffs_poly, x, y, sigma, prior_std,
                                   eps_scale=eps_scale, ridge=ridge)
    results.append({"Model": "Quadratic", "Params": coeffs_poly, "Log Marginal": log_marg_poly})
    fits["Quadratic"] = y_poly

if show_cubic:
    coeffs_cubic = np.polyfit(x, y, 3)
    y_cubic = np.polyval(coeffs_cubic, x)
    log_marg_cubic = laplace_approx(neg_log_post_cubic, coeffs_cubic, x, y, sigma, prior_std,
                                    eps_scale=eps_scale, ridge=ridge)
    results.append({"Model": "Cubic", "Params": coeffs_cubic, "Log Marginal": log_marg_cubic})
    fits["Cubic"] = y_cubic

if show_poly7:
    coeffs_poly7 = np.polyfit(x, y, 7)
    y_poly7 = np.polyval(coeffs_poly7, x)
    log_marg_poly7 = laplace_approx(neg_log_post_poly7, coeffs_poly7, x, y, sigma, prior_std,
                                    eps_scale=eps_scale, ridge=ridge)
    results.append({"Model": "7th Poly", "Params": coeffs_poly7, "Log Marginal": log_marg_poly7})
    fits["7th Poly"] = y_poly7

if show_sine:
    params_sine, y_sine = fit_sine_nonlin(x, y)
    log_marg_sine = laplace_approx(neg_log_post_sine, params_sine, x, y, sigma, prior_std,
                                   eps_scale=eps_scale, ridge=ridge)
    results.append({"Model": "Sine", "Params": params_sine, "Log Marginal": log_marg_sine})
    fits["Sine"] = y_sine

# ------------- Organize results -------------
if results:
    df = pd.DataFrame(results)
    best_log_marg = df["Log Marginal"].max()
    df["Δ Log Marginal vs Best"] = df["Log Marginal"] - best_log_marg

    st.subheader("Log Marginal Likelihoods & Δ vs Best")
    st.dataframe(df[["Model", "Log Marginal", "Δ Log Marginal vs Best"]].set_index("Model"))

    # Pairwise log Bayes factors (row vs column)
    models = df["Model"].tolist()
    bf_mat = pd.DataFrame(index=models, columns=models, dtype=float)
    for mi in models:
        for mj in models:
            bf_mat.loc[mi, mj] = float(df.loc[df.Model == mi, "Log Marginal"].values[0] -
                                       df.loc[df.Model == mj, "Log Marginal"].values[0])
    st.subheader("Pairwise Log Bayes Factors (row − column)")
    st.dataframe(bf_mat)

# ------------- Plotting -------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, s=12, alpha=0.5, label=f"Data (n={n})")
ax.plot(x, y_true, linewidth=2, alpha=0.6, label="True Sine (noise-free)")
for name, yhat in fits.items():
    ax.plot(x, yhat, linewidth=2, label=f"{name} fit")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Model Fits")
ax.legend()
st.pyplot(fig)

st.markdown(
    """
---
**Notes & Tips**
- Sine fit now uses FFT-based initialization + bounds. If it still misbehaves, widen the x-range or increase n.
- Laplace: adjust `eps_scale` and `ridge` if the Hessian is near-singular.
- Positive log Bayes factor (row − column) favors the row model.
- Extend easily: add new models, heavier-tailed likelihoods, different priors, or automatic differentiation.
"""
)
