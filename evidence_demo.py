"""
Interactive Laplace Approximation Demo
--------------------------------------
Streamlit web app that:
  • Simulates data from a sine model with user-controlled parameters (A, B, C, D, sigma, n, seed).
  • Fits quadratic, cubic, 7th-order polynomial, and sine models via MLE.
  • Computes Laplace-approximated log marginal likelihoods and (log) Bayes factors.
  • Plots data and fitted curves.

Run locally:
    pip install streamlit numpy scipy matplotlib pandas
    streamlit run Interactive_Sine_vs_Polynomial_Laplace_App.py

Author: (your name)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import slogdet
from scipy.optimize import curve_fit
import streamlit as st

# -------------------------
# Utility functions
# -------------------------

def sine_model(x, A, B, C, D):
    return A * np.sin(B * x + C) + D


def neg_log_post_poly(theta, x, y, sigma, prior_std):
    # Quadratic: theta = [a, b, c]
    a, b, c = theta
    model = a * x**2 + b * x + c
    return _neg_log_post_common(theta, model, y, sigma, prior_std)


def neg_log_post_cubic(theta, x, y, sigma, prior_std):
    # Cubic: theta = [a, b, c, d]
    a, b, c, d = theta
    model = a * x**3 + b * x**2 + c * x + d
    return _neg_log_post_common(theta, model, y, sigma, prior_std)


def neg_log_post_poly7(theta, x, y, sigma, prior_std):
    # 7th Order Polynomial: theta has 8 coefficients
    model = np.polyval(theta, x)
    return _neg_log_post_common(theta, model, y, sigma, prior_std)


def neg_log_post_sine(theta, x, y, sigma, prior_std):
    # Sine: theta = [A, B, C, D]
    A, B, C, D = theta
    model = sine_model(x, A, B, C, D)
    return _neg_log_post_common(theta, model, y, sigma, prior_std)


def _neg_log_post_common(theta, model, y, sigma, prior_std):
    n = y.size
    # log-likelihood
    ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - np.sum((y - model) ** 2) / (2 * sigma**2)
    # independent Gaussian priors N(0, prior_std^2)
    lp = -len(theta) * np.log(np.sqrt(2 * np.pi) * prior_std) - np.sum(theta ** 2) / (2 * prior_std ** 2)
    return -(ll + lp)


def compute_hessian(fun, theta, epsilon=1e-5):
    """Finite-difference Hessian. Central differences on each pair of dimensions.
    fun: function that takes theta -> scalar
    theta: 1D array-like
    """
    theta = np.array(theta, dtype=float)
    d = theta.size
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            th = theta.copy()
            th[i] += epsilon; th[j] += epsilon
            f_ijp = fun(th)
            th = theta.copy()
            th[i] += epsilon; th[j] -= epsilon
            f_ijm = fun(th)
            th = theta.copy()
            th[i] -= epsilon; th[j] += epsilon
            f_jip = fun(th)
            th = theta.copy()
            th[i] -= epsilon; th[j] -= epsilon
            f_jim = fun(th)
            H[i, j] = (f_ijp - f_ijm - f_jip + f_jim) / (4 * epsilon ** 2)
    return H


def laplace_approx(neg_log_post_fun, theta_hat, x, y, sigma, prior_std):
    """Return log marginal likelihood via Laplace approximation."""
    nlp = neg_log_post_fun(theta_hat, x, y, sigma, prior_std)
    d = len(theta_hat)

    def wrapper(th):
        return neg_log_post_fun(th, x, y, sigma, prior_std)

    H = compute_hessian(wrapper, theta_hat)
    sign, logdet = slogdet(H)
    if sign <= 0:
        raise ValueError("Hessian is not positive definite. Try different starting values or epsilon.")
    log_marg = -nlp + 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet
    return log_marg


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Laplace Approximation Demo", layout="wide")
st.title("Interactive Laplace Approximation: Sine vs Polynomial Models")

with st.sidebar:
    st.header("Simulation Controls")
    seed = st.number_input("Random Seed", value=42, step=1)
    n = st.slider("Number of data points (n)", min_value=50, max_value=5000, value=1000, step=50)
    x_max = st.slider("x max (range is 0 to x_max)", min_value=2.0, max_value=20.0, value=2 * np.pi, step=0.5)
    sigma = st.slider("Noise std (sigma)", min_value=0.05, max_value=2.0, value=0.5, step=0.05)

    st.subheader("True Sine Parameters")
    true_A = st.slider("A (amplitude)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    true_B = st.slider("B (frequency)", min_value=0.1, max_value=10.0, value=4.0, step=0.1)
    true_C = st.slider("C (phase shift)", min_value=-np.pi, max_value=np.pi, value=0.0, step=0.1)
    true_D = st.slider("D (offset)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    st.subheader("Prior Settings")
    prior_std = st.slider("Prior std (Gaussian, mean=0)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

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

# ------------- Model Fitting (MLE) -------------
results = []

if show_quadratic:
    coeffs_poly = np.polyfit(x, y, 2)
    y_poly = np.polyval(coeffs_poly, x)
    log_marg_poly = laplace_approx(neg_log_post_poly, coeffs_poly, x, y, sigma, prior_std)
    results.append({"Model": "Quadratic", "Params": coeffs_poly, "Log Marginal": log_marg_poly})

if show_cubic:
    coeffs_cubic = np.polyfit(x, y, 3)
    y_cubic = np.polyval(coeffs_cubic, x)
    log_marg_cubic = laplace_approx(neg_log_post_cubic, coeffs_cubic, x, y, sigma, prior_std)
    results.append({"Model": "Cubic", "Params": coeffs_cubic, "Log Marginal": log_marg_cubic})

if show_poly7:
    coeffs_poly7 = np.polyfit(x, y, 7)
    y_poly7 = np.polyval(coeffs_poly7, x)
    log_marg_poly7 = laplace_approx(neg_log_post_poly7, coeffs_poly7, x, y, sigma, prior_std)
    results.append({"Model": "7th Poly", "Params": coeffs_poly7, "Log Marginal": log_marg_poly7})

if show_sine:
    # initial guess can be important for curve_fit
    init = [true_A * 0.9, true_B * 0.5, 0.0, 0.0]
    params_sine, _ = curve_fit(sine_model, x, y, p0=init)
    y_sine = sine_model(x, *params_sine)
    log_marg_sine = laplace_approx(neg_log_post_sine, params_sine, x, y, sigma, prior_std)
    results.append({"Model": "Sine", "Params": params_sine, "Log Marginal": log_marg_sine})

# ------------- Organize results -------------
if results:
    df = pd.DataFrame(results)
    # Compute Bayes factors relative to best model (highest log marginal likelihood)
    best_log_marg = df["Log Marginal"].max()
    df["Δ Log Marginal vs Best"] = df["Log Marginal"] - best_log_marg
    # Log Bayes factors between models: we'll create a pairwise table too
    st.subheader("Log Marginal Likelihoods & Δ vs Best")
    st.dataframe(df[["Model", "Log Marginal", "Δ Log Marginal vs Best"]].set_index("Model"))

    # Pairwise log Bayes factors
    bf_mat = pd.DataFrame(index=df["Model"], columns=df["Model"], dtype=float)
    for i, mi in df.iterrows():
        for j, mj in df.iterrows():
            bf_mat.loc[mi["Model"], mj["Model"]] = mi["Log Marginal"] - mj["Log Marginal"]
    st.subheader("Pairwise Log Bayes Factors (row vs column)")
    st.dataframe(bf_mat)

# ------------- Plotting -------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, s=12, alpha=0.5, label=f"Data (n={n})")
ax.plot(x, y_true, linewidth=2, alpha=0.6, label="True Sine (no noise)")

if show_quadratic:
    ax.plot(x, y_poly, linestyle='--', linewidth=2, label="Quadratic fit")
if show_cubic:
    ax.plot(x, y_cubic, linestyle='-.', linewidth=2, label="Cubic fit")
if show_poly7:
    ax.plot(x, y_poly7, linestyle=':', linewidth=2, label="7th Poly fit")
if show_sine:
    ax.plot(x, y_sine, linewidth=2, label="Sine fit")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Model Fits")
ax.legend()
st.pyplot(fig)

st.markdown("""
---
**Notes**
- Laplace approximation assumes a well-behaved posterior around the MLE/MAP. If the Hessian isn't positive-definite, try changing epsilon, the prior_std, or initial guesses.
- Bayes factors are reported on the log scale. Positive values favor the row model over the column model.
- You can extend this app with additional models or priors (e.g., hierarchical priors, different noise structures).
""")
