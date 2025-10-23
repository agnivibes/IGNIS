!pip install ucimlrepo



# =============================================================================
# CDC Dataset Analysis (Real Data with Bootstrap SE)
# Structure mirrors the simulation code; only the final evaluation differs.
# =============================================================================
import os
os.environ["PYTHONHASHSEED"] = "123"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from scipy.optimize import bisect
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from numpy.random import default_rng
import torch
from ucimlrepo import fetch_ucirepo

# -------------------------
# Reproducibility (same as simulation code)
# -------------------------
def set_seed(seed=123):
    # Base Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # TensorFlow
    tf.random.set_seed(seed)
    # Optional: full determinism on GPU (may impact performance)
    # tf.config.experimental.enable_op_determinism()

# -------------------------
# Copula generators (θ ≥ 1) — identical
# -------------------------
def phi_gumbel(t, theta):      return (-np.log(t))**theta
def dphi_gumbel(t, theta):     return - (theta / t) * ((-np.log(t))**(theta-1))
def phi_gumbel_inv(x, theta):  return np.exp(- x**(1.0/theta))

def phi_joe(t, theta):         return -np.log(1.0 - (1.0 - t)**theta)
def dphi_joe(t, theta):
    num = theta * (1.0 - t)**(theta-1)
    den = 1.0 - (1.0 - t)**theta
    return - num / (den + 1e-15) # Correctly returns a NEGATIVE value
def phi_joe_inv(x, theta):     return 1.0 - (1.0 - np.exp(-x))**(1.0/theta)

def phi_A1(t, theta):
    f = t**(1.0/theta) + t**(-1.0/theta) - 2.0
    return f**theta
def dphi_A1(t, theta):
    f  = t**(1.0/theta) + t**(-1.0/theta) - 2.0
    fp = (1.0/theta)*t**(1.0/theta - 1) - (1.0/theta)*t**(-1.0/theta - 1)
    return theta * (f**(theta-1)) * fp
def phi_A1_inv(y, theta):
    a = y**(1.0/theta) + 2.0
    inner = (a - np.sqrt(a*a - 4.0)) / 2.0
    return inner**theta

def phi_A2(t, theta):          return ((1.0/t) * (1.0 - t)**2)**theta
def dphi_A2(t, theta):
    g = (1.0 - t)**2 / t
    num, den = -(1.0 - t) * (1.0 + t), t * t
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
# K-function, inverse, sampling — identical
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
# Features — identical (5 features)
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
# Simulated training data — identical
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
# Neural network — identical
# -------------------------
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

# -------------------------
# Train (on simulated data) — identical
# -------------------------
set_seed(123)
print("--- Generating Simulated Data & Training IGNIS Model ---")
X_sim, y_sim, copula_types = generate_training_data(sample_size=5000)
X_train, X_val, y_train, y_val = train_test_split(X_sim, y_sim, test_size=0.2, random_state=123)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)

model = create_theta_estimator_model(X_train_s.shape[1])
model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
    verbose=1
)

# -------------------------
# Load & preprocess REAL data (CDC)
# -------------------------
print("\n--- Loading and Preprocessing CDC Diabetes Data ---")
cdc = fetch_ucirepo(id=891)
df  = cdc.data.features[["GenHlth", "PhysHlth"]].sort_index()  # lock row order
n   = len(df)
u_real = (df["GenHlth"].rank(method="average") / (n + 1)).to_numpy()
v_real = (df["PhysHlth"].rank(method="average") / (n + 1)).to_numpy()


# -------------------------
# Bootstrap uncertainty
# -------------------------
print("\n--- Applying Trained Model to Real CDC Data ---")
print("Copula    | θ Estimate | Bootstrap SE")
print("----------|------------|--------------")

B = 1000  # bootstrap iterations (deterministic)
rng = default_rng(2025)  # fixed seed → identical SEs across runs

for name in copula_types:
    one_hot = np.zeros(len(copula_types))
    one_hot[copula_types.index(name)] = 1.0

    # point estimate on full sample
    feats_full = compute_summary_features(u_real, v_real)
    iv_full = np.hstack([feats_full, one_hot]).reshape(1, -1)
    theta_hat_full = model.predict(scaler.transform(iv_full), verbose=0)[0, 0]

    # bootstrap SE (pairs bootstrap)
    boot_preds = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)  # deterministic resamples
        feats_b = compute_summary_features(u_real[idx], v_real[idx])
        iv_b = np.hstack([feats_b, one_hot]).reshape(1, -1)
        pred_b = model.predict(scaler.transform(iv_b), verbose=0)[0, 0]
        boot_preds.append(pred_b)

    se = np.std(boot_preds, ddof=1)  # classic unbiased SE
    print(f"{name:<9} | {theta_hat_full:>10.4f} | {se:.4f}")







# =============================================================================
# Finance Dataset Analysis (Real Data with Bootstrap SE)
# Structure mirrors the simulation code; only the final evaluation differs.
# =============================================================================
import os
os.environ["PYTHONHASHSEED"] = "123"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, rankdata
from scipy.optimize import bisect
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from numpy.random import default_rng
import torch
import yfinance as yf

# -------------------------
# Reproducibility (same as simulation code)
# -------------------------
def set_seed(seed=123):
    # Base Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # TensorFlow
    tf.random.set_seed(seed)
    # Optional: full determinism on GPU (may impact performance)
    # tf.config.experimental.enable_op_determinism()

# -------------------------
# Copula generators (θ ≥ 1) — identical
# -------------------------
def phi_gumbel(t, theta):      return (-np.log(t))**theta
def dphi_gumbel(t, theta):     return - (theta / t) * ((-np.log(t))**(theta-1))
def phi_gumbel_inv(x, theta):  return np.exp(- x**(1.0/theta))

def phi_joe(t, theta):         return -np.log(1.0 - (1.0 - t)**theta)
def dphi_joe(t, theta):
    num = theta * (1.0 - t)**(theta-1)
    den = 1.0 - (1.0 - t)**theta
    return -num / (den + 1e-15)
def phi_joe_inv(x, theta):     return 1.0 - (1.0 - np.exp(-x))**(1.0/theta)

def phi_A1(t, theta):
    f = t**(1.0/theta) + t**(-1.0/theta) - 2.0
    return f**theta
def dphi_A1(t, theta):
    f  = t**(1.0/theta) + t**(-1.0/theta) - 2.0
    fp = (1.0/theta)*t**(1.0/theta - 1) - (1.0/theta)*t**(-1.0/theta - 1)
    return theta * (f**(theta-1)) * fp
def phi_A1_inv(y, theta):
    a = y**(1.0/theta) + 2.0
    inner = (a - np.sqrt(a*a - 4.0)) / 2.0
    return inner**theta

def phi_A2(t, theta):          return ((1.0/t) * (1.0 - t)**2)**theta
def dphi_A2(t, theta):
    g = (1.0 - t)**2 / t
    num, den = -(1.0 - t) * (1.0 + t), t * t
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
# K-function, inverse, sampling — identical
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
# Features — identical (5 features)
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
# Simulated training data — identical
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
# Neural network — identical
# -------------------------
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

# -------------------------
# Train (on simulated data) — identical
# -------------------------
set_seed(123)
print("--- Generating Simulated Data & Training IGNIS Model ---")
X_sim, y_sim, copula_types = generate_training_data(sample_size=5000)
X_train, X_val, y_train, y_val = train_test_split(X_sim, y_sim, test_size=0.2, random_state=123)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)

model = create_theta_estimator_model(X_train_s.shape[1])
model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
    verbose=1
)

# -------------------------
# Load & preprocess REAL data (Finance) — new final section
# -------------------------
print("\n--- Loading and Preprocessing Financial Data ---")
TICKERS = ['AAPL', 'MSFT']

# Force auto_adjust=False so 'Adj Close' exists; silence progress + warning
data = yf.download(TICKERS, start='2020-01-01', end='2023-12-31',
                   auto_adjust=False, progress=False)

# Handle MultiIndex (common for multiple tickers) vs single-index
if isinstance(data.columns, pd.MultiIndex):
    prices = data['Adj Close'][TICKERS]  # columns: AAPL, MSFT
else:
    # Fallback: if single-index, keep just the two adj close columns by name
    prices = data[['Adj Close']]  # unlikely when multiple tickers, but safe

# Clean and compute log returns
prices = prices.dropna(how='any')
CACHE = "aapl_msft_2020_2023_adjclose.csv"
if os.path.exists(CACHE):
    prices = pd.read_csv(CACHE, index_col=0, parse_dates=True)
else:
    # (use the prices computed above)
    prices.to_csv(CACHE)

log_returns = np.log(prices / prices.shift(1)).dropna()

def pit_transform(series):
    return rankdata(series, method='average') / (len(series) + 1)

u_real = pit_transform(log_returns['AAPL'].to_numpy())
v_real = pit_transform(log_returns['MSFT'].to_numpy())
n = len(u_real)


# -------------------------
# Bootstrap uncertainty 
# -------------------------

print("\n--- Applying Trained Model to Real Financial Data ---")
print("Copula    | θ Estimate | Bootstrap SE")
print("----------|------------|--------------")

B = 1000
rng = default_rng(2025)  # fixed seed → identical SEs across runs

for name in copula_types:
    one_hot = np.zeros(len(copula_types))
    one_hot[copula_types.index(name)] = 1.0

    # point estimate on full sample
    feats_full = compute_summary_features(u_real, v_real)
    iv_full = np.hstack([feats_full, one_hot]).reshape(1, -1)
    theta_hat_full = model.predict(scaler.transform(iv_full), verbose=0)[0, 0]

    # pairs bootstrap with deterministic RNG
    boot_preds = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        feats_b = compute_summary_features(u_real[idx], v_real[idx])
        iv_b = np.hstack([feats_b, one_hot]).reshape(1, -1)
        pred_b = model.predict(scaler.transform(iv_b), verbose=0)[0, 0]
        boot_preds.append(pred_b)

    se = np.std(boot_preds, ddof=1)
    print(f"{name:<9} | {theta_hat_full:>10.4f} | {se:.4f}")
