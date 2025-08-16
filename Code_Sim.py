import numpy as np
from scipy.stats import kendalltau, spearmanr
from scipy.optimize import bisect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import os
import torch

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tf.random.set_seed(seed)
    # Optional GPU determinism:
    # tf.config.experimental.enable_op_determinism()

# -------------------------
# Copula generators (θ ≥ 1)
# -------------------------
def phi_gumbel(t, theta):      return (-np.log(t))**theta
def dphi_gumbel(t, theta):     return - (theta / t) * ((-np.log(t))**(theta-1))
def phi_gumbel_inv(x, theta):  return np.exp(- x**(1.0/theta))

def phi_joe(t, theta):         return -np.log(1.0 - (1.0 - t)**theta)
def dphi_joe(t, theta):
    num = theta * (1.0 - t)**(theta-1)
    den = 1.0 - (1.0 - t)**theta
    return num / (den + 1e-15)
def phi_joe_inv(x, theta):     return 1.0 - (1.0 - np.exp(-x))**(1.0/theta)

def phi_A1(t, theta):
    f = t**(1.0/theta) + t**(-1.0/theta) - 2.0
    return f**theta
def dphi_A1(t, theta):
    f = t**(1.0/theta) + t**(-1.0/theta) - 2.0
    fp = (1.0/theta)*t**(1.0/theta - 1) - (1.0/theta)*t**(-1.0/theta - 1)
    return theta * (f**(theta-1)) * fp
def phi_A1_inv(y, theta):
    a = y**(1.0/theta) + 2.0
    inner = (a - np.sqrt(a*a - 4.0)) / 2.0
    return inner**theta

def phi_A2(t, theta):          return ((1.0/t) * (1.0 - t)**2)**theta
def dphi_A2(t, theta):
    g = (1.0 - t)**2 / t
    num = -(1.0 - t) * (1.0 + t)
    den = t * t
    gprime = num / den
    return theta * (g**(theta-1)) * gprime
def phi_A2_inv(y, theta):
    a = 2.0 + y**(1.0/theta)
    return (a - np.sqrt(a*a - 4.0)) / 2.0

copulas = {
    "Gumbel": {"phi": phi_gumbel, "dphi": dphi_gumbel, "phi_inv": phi_gumbel_inv},
    "Joe":    {"phi": phi_joe,    "dphi": dphi_joe,    "phi_inv": phi_joe_inv},
    "A1":     {"phi": phi_A1,     "dphi": dphi_A1,     "phi_inv": phi_A1_inv},
    "A2":     {"phi": phi_A2,     "dphi": dphi_A2,     "phi_inv": phi_A2_inv},
}

# -------------------------
# K-function, inverse, sampling
# -------------------------
def K_function(x, phi, dphi, theta):
    return x - (phi(x, theta) / (dphi(x, theta) + 1e-15))

def K_inverse(t, phi, dphi, theta, tol=1e-9):
    left, right = 1e-14, 1.0 - 1e-14
    def f(x): return K_function(x, phi, dphi, theta) - t
    return bisect(f, left, right, xtol=tol)

def sample_archimedean_copula_alg1(phi, dphi, phi_inv, theta, n=3000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    s_vals = np.random.rand(n)
    t_vals = np.random.rand(n)
    u = np.empty(n); v = np.empty(n)
    for i in range(n):
        w = K_inverse(t_vals[i], phi, dphi, theta)
        phi_w = phi(w, theta)
        u[i] = phi_inv(s_vals[i] * phi_w, theta)
        v[i] = phi_inv((1.0 - s_vals[i]) * phi_w, theta)
    return u, v

# -------------------------
# Features
# -------------------------
def compute_summary_features(U, V):
    tau_emp, _ = kendalltau(U, V)
    rho_emp, _ = spearmanr(U, V)
    upper_thr, lower_thr = 0.95, 0.05
    upper_tail_dep = np.mean((U > upper_thr) & (V > upper_thr))
    lower_tail_dep = np.mean((U < lower_thr) & (V < lower_thr))
    corr = np.corrcoef(U, V)[0,1]
    return np.array([tau_emp, rho_emp, upper_tail_dep, lower_tail_dep, corr])

# -------------------------
# Simulated training data
# -------------------------
def generate_training_data(n_samples=500, sample_size=5000, theta_range=(1.0,20.0)):
    X_list, y_list, copula_list = [], [], []
    types = list(copulas.keys())
    for name in types:
        funcs = copulas[name]
        thetas = np.linspace(theta_range[0], theta_range[1], n_samples)
        for theta in thetas:
            U, V = sample_archimedean_copula_alg1(funcs["phi"], funcs["dphi"], funcs["phi_inv"],
                                                  theta, n=sample_size)
            feat = compute_summary_features(U, V)
            X_list.append(feat)
            y_list.append(theta)
            one_hot = np.zeros(len(types))
            one_hot[types.index(name)] = 1.0
            copula_list.append(one_hot)
    X = np.hstack([np.array(X_list), np.array(copula_list)])
    y = np.array(y_list)
    return X, y, types

# -------------------------
# Train model
# -------------------------
set_seed(123)
X, y, copula_types = generate_training_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)

def create_theta_estimator_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(64,  activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(1,   activation='softplus'),
        layers.Lambda(lambda x: x + 1.0)  # θ ≥ 1
    ])
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss='mse')
    return model

input_dim = X_train_scaled.shape[1]
theta_model = create_theta_estimator_model(input_dim)
theta_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
    verbose=1
)

# -------------------------
# Rigorous Simulation Study
# -------------------------
thetas_to_test = [2.0, 5.0, 10.0, 15.0, 20.0]
n_test_samples = 5000
n_runs = 100

print("--- Running Rigorous Simulation Study ---")
for true_theta in thetas_to_test:
    print(f"\n----- True Theta = {true_theta:.1f} -----")
    for name in copula_types:
        funcs = copulas[name]
        estimates_for_this_setting = []
        for i in range(n_runs):
            U, V = sample_archimedean_copula_alg1(
                funcs["phi"], funcs["dphi"], funcs["phi_inv"],
                true_theta, n=n_test_samples
            )
            feat = compute_summary_features(U, V)
            one_hot = np.zeros(len(copula_types))
            one_hot[copula_types.index(name)] = 1.0
            model_input = np.hstack([feat, one_hot]).reshape(1, -1)
            scaled_input = scaler.transform(model_input)
            est = theta_model.predict(scaled_input, verbose=0)[0, 0]
            estimates_for_this_setting.append(est)

        estimates = np.array(estimates_for_this_setting)
        mean_estimate = np.mean(estimates)
        std_dev_estimate = np.std(estimates)
        bias = mean_estimate - true_theta
        rmse = np.sqrt(np.mean((estimates - true_theta) ** 2))

        print(f"{name:<10s} | Est. θ: {mean_estimate:.2f} | Bias: {bias:.2f} | "
              f"Std. Dev.: {std_dev_estimate:.2f} | RMSE: {rmse:.2f}")

# =============================================================================
# APPENDIX : LOG-LIKELIHOOD COMPARISON CODE (IGNIS vs MoM for A1 & A2)
# =============================================================================

# ---- Analytic second derivatives for A1 & A2 + stable LL via inverse-function theorem ----
EPS = 1e-12

def d2phi_A1(t, theta):
    # f(t) = t^(1/theta) + t^(-1/theta) - 2
    a   = 1.0/theta
    f   = t**a + t**(-a) - 2.0
    fp  = a*t**(a-1.0) - a*t**(-a-1.0)
    fpp = a*(a-1.0)*t**(a-2.0) + a*(a+1.0)*t**(-a-2.0)
    return theta*(theta-1.0)*(f**(theta-2.0))*(fp**2) + theta*(f**(theta-1.0))*fpp

def d2phi_A2(t, theta):
    # g(t) = ((1-t)^2)/t
    g   = ((1.0 - t)**2) / t
    gp  = 1.0 - 1.0/(t*t)        # = -(1-t)(1+t)/t^2
    gpp = 2.0/(t**3)
    return theta*(theta-1.0)*(g**(theta-2.0))*(gp**2) + theta*(g**(theta-1.0))*gpp

_phi2 = {"A1": d2phi_A1, "A2": d2phi_A2}

def calculate_log_likelihood(uv_data, theta, copula_name, copulas_dict):
    """
    Stable 2D Archimedean copula log-likelihood.
      ψ = φ^{-1}, s = φ(u)+φ(v), w = ψ(s)
      ψ''(s) = -φ''(w) / [φ'(w)]^3
      c(u,v) = ψ''(s) * φ'(u) * φ'(v)
    """
    phi     = lambda t: copulas_dict[copula_name]["phi"](t, theta)
    dphi    = lambda t: copulas_dict[copula_name]["dphi"](t, theta)
    phi_inv = lambda s: copulas_dict[copula_name]["phi_inv"](s, theta)
    d2phi   = _phi2[copula_name]  # only A1/A2 here

    total_ll = 0.0
    valid = 0

    for u, v in uv_data:
        # keep within (0,1)
        if not (0.0 < u < 1.0 and 0.0 < v < 1.0):
            continue
        u = min(max(u, EPS), 1.0 - EPS)
        v = min(max(v, EPS), 1.0 - EPS)

        s = phi(u) + phi(v)        # > 0
        w = phi_inv(s)             # in (0,1)

        # derivatives
        phi_p_u  = dphi(u)
        phi_p_v  = dphi(v)
        phi_p_w  = dphi(w)
        phi_pp_w = d2phi(w, theta)

        # ψ''(s) via inverse-function theorem
        denom = (phi_p_w**3) if phi_p_w != 0.0 else (np.sign(phi_p_w) * EPS)
        psi_pp = - phi_pp_w / (denom + EPS)

        dens = psi_pp * phi_p_u * phi_p_v
        if np.isfinite(dens) and dens > 0.0:
            total_ll += np.log(dens + EPS)
            valid += 1

    return total_ll if valid > 0 else -np.inf


estimates_theta_5 = {
    "MoM":   {"A1": 9.52,  "A2": 4.94},   # <--  MoM estimates at θ=5
    "IGNIS": {"A1": 5.10,  "A2": 4.97},   # <--  IGNIS estimates at θ=5
}

# Multiple-test averaging for stabler comparison
true_theta_test = 5.0
n_test = 5000
seeds = [101,102,103,104,105,201,202,203,204,205]  # 10 test draws

print("\n--- Appendix D: Averaged Log-L Comparison (θ_true = 5.0) ---")
print(f"{'Copula':<6s} | {'Mean LL MoM':>12s} | {'Mean LL IGNIS':>14s} | {'Δ(IGNIS-MoM)':>12s} | {'Wins(IGNIS)':>11s}")
print("-"*70)

for name in ["A1","A2"]:
    ll_mom_list, ll_ignis_list, wins = [], [], 0

    for sd in seeds:
        U, V = sample_archimedean_copula_alg1(
            copulas[name]["phi"], copulas[name]["dphi"], copulas[name]["phi_inv"],
            true_theta_test, n=n_test, seed=sd
        )
        test_data = np.column_stack([U, V])

        theta_mom   = float(estimates_theta_5["MoM"][name])
        theta_ignis = float(estimates_theta_5["IGNIS"][name])

        ll_mom   = calculate_log_likelihood(test_data, theta_mom,   name, copulas)
        ll_ignis = calculate_log_likelihood(test_data, theta_ignis, name, copulas)

        ll_mom_list.append(ll_mom)
        ll_ignis_list.append(ll_ignis)
        if ll_ignis > ll_mom:
            wins += 1

    mean_mom   = np.mean(ll_mom_list)
    mean_ignis = np.mean(ll_ignis_list)
    diff       = mean_ignis - mean_mom

    print(f"{name:<6s} | {mean_mom:12.2f} | {mean_ignis:14.2f} | {diff:12.2f} | {wins:11d}/{len(seeds)}")
