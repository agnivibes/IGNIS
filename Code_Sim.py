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
# LOG-LIKELIHOOD COMPARISON (IGNIS vs MoM) — NUMERICALLY STABLE
# =============================================================================

EPS       = 1e-12
EPS_A1    = 1e-10   # slightly stronger clip for A1 near boundaries
TINY      = 1e-300  # floor for logs to avoid -inf

# ---- exact second derivatives (do NOT add EPS inside the bases) ----
def d2phi_A1(t, theta):
    a   = 1.0 / theta
    f   = t**a + t**(-a) - 2.0
    fp  = a * t**(a - 1.0) - a * t**(-a - 1.0)
    fpp = a * (a - 1.0) * t**(a - 2.0) + a * (a + 1.0) * t**(-a - 2.0)
    return theta * (theta - 1.0) * (f**(theta - 2.0)) * (fp**2) + theta * (f**(theta - 1.0)) * fpp

def d2phi_A2(t, theta):
    # g(t) = ((1-t)^2)/t = 1/t - 2 + t
    g   = (1.0 - t)**2 / t
    gp  = 1.0 - 1.0 / (t * t)
    gpp = 2.0 / (t ** 3)
    return theta * (theta - 1.0) * (g**(theta - 2.0)) * (gp**2) + theta * (g**(theta - 1.0)) * gpp

_phi2 = {"A1": d2phi_A1, "A2": d2phi_A2}

# ---- safe inverses (floor discriminant; clamp outputs to [0,1]) ----
def phi_A1_inv_safe(y, theta):
    x = np.power(np.maximum(y, 0.0), 1.0/theta)
    disc = np.maximum((x + 2.0)**2 - 4.0, 0.0)
    base = (x + 2.0 - np.sqrt(disc)) / 2.0
    base = np.clip(base, 0.0, 1.0)
    return np.power(base, theta)

def phi_A2_inv_safe(y, theta):
    x = np.power(np.maximum(y, 0.0), 1.0/theta)
    disc = np.maximum((x + 2.0)**2 - 4.0, 0.0)
    w = (x + 2.0 - np.sqrt(disc)) / 2.0
    return np.clip(w, 0.0, 1.0)

_phi_inv_safe = {"A1": phi_A1_inv_safe, "A2": phi_A2_inv_safe}

# ---- stable, vectorized log-likelihood (log-space; clip inputs only) ----
def archimedean_loglik_stable(U, V, theta, copula_name):
    if theta is None or np.isnan(theta):
        return -np.inf, 0.0
    if copula_name not in ("A1", "A2"):
        raise ValueError("This block compares A1/A2 only.")

    phi  = lambda t: copulas[copula_name]["phi"](t, theta)
    dphi = lambda t: copulas[copula_name]["dphi"](t, theta)
    d2phi = _phi2[copula_name]
    inv  = _phi_inv_safe[copula_name]

    # clip u,v (A1 slightly stronger)
    eps_here = EPS_A1 if copula_name == "A1" else EPS
    U = np.clip(U, eps_here, 1.0 - eps_here)
    V = np.clip(V, eps_here, 1.0 - eps_here)

    # s = φ(u)+φ(v), w = φ^{-1}(s)
    Su = phi(U); Sv = phi(V)
    S  = Su + Sv
    W  = inv(S, theta)
    W  = np.clip(W, eps_here, 1.0 - eps_here)

    # psi''(W) = -φ''(W)/[φ'(W)]^3, and c(u,v)=psi''(W) * (-φ'(u)) * (-φ'(v))
    d1u = -dphi(U)   # φ'(U) < 0 => -φ'(U) > 0
    d1v = -dphi(V)
    d1w = -dphi(W)
    d2w =  d2phi(W, theta)

    # floors for logs
    d1u = np.maximum(d1u, TINY)
    d1v = np.maximum(d1v, TINY)
    d1w = np.maximum(d1w, TINY)
    d2w = np.maximum(d2w, TINY)

    logc = np.log(d2w) - 3.0*np.log(d1w) + np.log(d1u) + np.log(d1v)

    mask = np.isfinite(logc)
    valid_ratio = mask.mean()
    total_ll = np.sum(logc[mask]) if mask.any() else -np.inf
    return total_ll, valid_ratio

# ---- use your fixed point estimates (as before) ----
estimates_for_comparison = {
    2.0:  {"MoM": {"A1": 2.05,  "A2": 1.90},  "IGNIS": {"A1": 1.97,  "A2": 1.91}},
    5.0:  {"MoM": {"A1": 4.99,  "A2": 4.94},  "IGNIS": {"A1": 5.10,  "A2": 4.97}},
    10.0: {"MoM": {"A1": 10.03, "A2": 9.44},  "IGNIS": {"A1": 10.10, "A2": 10.05}},
}

# ---- run the comparison on both A1 and A2 (same data for both estimators) ----
thetas_to_test = [2.0, 5.0, 10.0]
n_test = 5000
n_replications = 100

print("\n--- Appendix E: Log-Likelihood Comparison ---")
print(f"{'True θ':<8} | {'Copula':<3} | {'Mean LL (MoM)':>15} | {'Mean LL (IGNIS)':>16} | {'Δ(IGNIS-MoM)':>12} | {'valid%':>7}")
print("-" * 95)

for true_theta in thetas_to_test:
    for name in ["A1", "A2"]:
        ll_mom_list, ll_ignis_list, val_m_list, val_i_list = [], [], [], []
        for i in range(n_replications):
            U, V = sample_archimedean_copula_alg1(
                copulas[name]["phi"], copulas[name]["dphi"], copulas[name]["phi_inv"],
                true_theta, n=n_test, seed=i
            )
            th_mom   = estimates_for_comparison[true_theta]["MoM"][name]
            th_ignis = estimates_for_comparison[true_theta]["IGNIS"][name]

            ll_mom,   vr_m = archimedean_loglik_stable(U, V, th_mom,   name)
            ll_ignis, vr_i = archimedean_loglik_stable(U, V, th_ignis, name)

            ll_mom_list.append(ll_mom);     val_m_list.append(vr_m)
            ll_ignis_list.append(ll_ignis); val_i_list.append(vr_i)

        mean_mom   = np.mean(ll_mom_list)
        mean_ignis = np.mean(ll_ignis_list)
        diff       = mean_ignis - mean_mom
        valid_pct  = 100.0 * min(np.mean(val_m_list), np.mean(val_i_list))
        print(f"{true_theta:<8.1f} | {name:<3} | {mean_mom:15.2f} | {mean_ignis:16.2f} | {diff:12.2f} | {valid_pct:6.1f}%")
