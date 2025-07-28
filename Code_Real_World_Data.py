#CDC Dataset


!pip install ucimlrepo
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from scipy.optimize import bisect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import os
import torch
import time
from ucimlrepo import fetch_ucirepo

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
    # Optional: For full determinism on GPU, but may impact performance
    # tf.config.experimental.enable_op_determinism()
set_seed(123)

print("--- Loading and Preprocessing CDC Diabetes Data ---")

def compute_summary_features(U, V):
    """This function is defined here because it's needed for the real data right away.
    It is the standardized 5-feature function."""
    tau_emp, _        = kendalltau(U, V)
    rho_emp, _        = spearmanr(U, V)
    upper_tail_dep    = np.mean((U > 0.95) & (V > 0.95))
    lower_tail_dep    = np.mean((U < 0.05) & (V < 0.05))
    corr              = np.corrcoef(U, V)[0, 1]
    return np.array([tau_emp, rho_emp, upper_tail_dep, lower_tail_dep, corr])


# 1. Fetch CDC dataset (ID 891) and grab features
cdc = fetch_ucirepo(id=891)
df  = cdc.data.features

# 2. Compute pseudo‐values from GenHlth & PhysHlth
n       = len(df)
gen_pu  = df["GenHlth"] .rank(method="average") / (n + 1)
phys_pu = df["PhysHlth"].rank(method="average") / (n + 1)

# 3. Plug into your pipeline
u_real        = gen_pu .values
v_real        = phys_pu.values
features_real = compute_summary_features(u_real, v_real)

print(f"Computed Real Data Features: {features_real}")


# Copula generators, derivatives, and inverses
def phi_gumbel(t, theta): return (-np.log(t)) ** theta


def dphi_gumbel(t, theta): return - (theta / t) * ((-np.log(t)) ** (theta - 1))


def phi_gumbel_inv(x, theta): return np.exp(-x ** (1.0 / theta))


def phi_joe(t, theta): return -np.log(1.0 - (1.0 - t) ** theta)


def dphi_joe(t, theta): return theta * (1.0 - t) ** (theta - 1) / (1.0 - (1.0 - t) ** theta + 1e-15)


def phi_joe_inv(x, theta): return 1.0 - (1.0 - np.exp(-x)) ** (1.0 / theta)


def phi_A1(t, theta): return (t ** (1.0 / theta) + t ** (-1.0 / theta) - 2.0) ** theta


def dphi_A1(t, theta):
    f = t ** (1.0 / theta) + t ** (-1.0 / theta) - 2.0
    fp = (1.0 / theta) * t ** (1.0 / theta - 1) - (1.0 / theta) * t ** (-1.0 / theta - 1)
    return theta * (f ** (theta - 1)) * fp


def phi_A1_inv(y, theta):
    a = y ** (1.0 / theta) + 2.0
    inner = (a - np.sqrt(a * a - 4.0)) / 2.0
    return inner ** theta


def phi_A2(t, theta): return (((1.0 - t) ** 2) / t) ** theta


def dphi_A2(t, theta):
    g = (1.0 - t) ** 2 / t
    gprime = -(1.0 - t) * (1.0 + t) / (t * t)
    return theta * (g ** (theta - 1)) * gprime


def phi_A2_inv(y, theta):
    a = 2.0 + y ** (1.0 / theta)
    return (a - np.sqrt(a * a - 4.0)) / 2.0


copulas = {
    "Gumbel": {"phi": phi_gumbel, "dphi": dphi_gumbel, "phi_inv": phi_gumbel_inv},
    "Joe": {"phi": phi_joe, "dphi": dphi_joe, "phi_inv": phi_joe_inv},
    "A1": {"phi": phi_A1, "dphi": dphi_A1, "phi_inv": phi_A1_inv},
    "A2": {"phi": phi_A2, "dphi": dphi_A2, "phi_inv": phi_A2_inv},
}


# Sampling functions
def K_function(x, phi, dphi, theta):
    return x - (phi(x, theta) / (dphi(x, theta) + 1e-15))


def K_inverse(t, phi, dphi, theta, tol=1e-9):
    return bisect(lambda x: K_function(x, phi, dphi, theta) - t, 1e-14, 1.0 - 1e-14, xtol=tol)


def sample_archimedean_copula_alg1(phi, dphi, phi_inv, theta, n=3000, seed=None):
    if seed is not None: np.random.seed(seed)
    s, t = np.random.rand(n), np.random.rand(n)
    U, V = np.empty(n), np.empty(n)
    for i in range(n):
        w = K_inverse(t[i], phi, dphi, theta)
        phi_w = phi(w, theta)
        U[i] = phi_inv(s[i] * phi_w, theta)
        V[i] = phi_inv((1 - s[i]) * phi_w, theta)
    return U, V


# simulated data generation 
def generate_training_data(n_thetas=500, sample_size=5000, theta_range=(1.0, 20.0)):
    X_list, y_list, oh_list = [], [], []
    families = list(copulas.keys())
    for name in families:
        funcs = copulas[name]
        thetas = np.linspace(theta_range[0], theta_range[1], n_thetas)
        for θ in thetas:
            U, V = sample_archimedean_copula_alg1(funcs["phi"], funcs["dphi"], funcs["phi_inv"], θ, n=sample_size)
            feats = compute_summary_features(U, V)
            X_list.append(feats)
            y_list.append(θ)
            one_hot = np.zeros(len(families))
            one_hot[families.index(name)] = 1.0
            oh_list.append(one_hot)
    X = np.hstack([np.array(X_list), np.array(oh_list)])
    y = np.array(y_list)
    return X, y, families


print("\n--- Generating Simulated Data for Model Training ---")
X_sim, y_sim, copula_types = generate_training_data(sample_size=5000)
print(f"Generated {X_sim.shape[0]} training samples.")

# model trianing 

print("\n--- Training the IGNIS Network ---")
X_train, X_val, y_train, y_val = train_test_split(X_sim, y_sim, test_size=0.2, random_state=123)
scaler = StandardScaler().fit(X_train)
X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)


def create_theta_estimator_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(1, activation='softplus'),
        layers.Lambda(lambda x: x + 1.0)
    ])
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss='mse')
    return model


input_dim = X_train_s.shape[1]
model = create_theta_estimator_model(input_dim)
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=200, batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
    verbose=1
)


# uncertainty quantification

def bootstrap_theta_se(U, V, model, scaler, one_hot, B=100):
    n = len(U)
    boot_preds = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        feats = compute_summary_features(U[idx], V[idx])
        iv = np.hstack([feats, one_hot]).reshape(1, -1)
        pred = model.predict(scaler.transform(iv), verbose=0)[0, 0]
        boot_preds.append(pred)
    return np.std(boot_preds)



# This section applies the trained model to the real data from Section 1.


print("\n--- Applying Trained Model to Real CDC Data ---")
results, se_results = {}, {}

for i, fam_name in enumerate(copula_types):
    one_hot = np.zeros(len(copula_types))
    one_hot[i] = 1.0
    input_vector = np.hstack([features_real, one_hot]).reshape(1, -1)
    input_vector_scaled = scaler.transform(input_vector)

    pred_theta = model.predict(input_vector_scaled, verbose=0)[0, 0]
    results[fam_name] = pred_theta
    se_results[fam_name] = bootstrap_theta_se(u_real, v_real, model, scaler, one_hot, B=100)

print("\nIGNIS Network Estimates for CDC Diabetes Data:")
print("Copula    | θ Estimate | SE")
print("----------|------------|---------")
for fam_name in copula_types:
    print(f"{fam_name:<9} | {results[fam_name]:>10.4f} | {se_results[fam_name]:.4f}")




#Finance Dataset Code

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, rankdata
from scipy.optimize import bisect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import os
import torch
import time


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
    # Optional: For full determinism on GPU, but may impact performance
    tf.config.experimental.enable_op_determinism()

set_seed(123)
# -------------------------------
# 1. Real-Life Data Preprocessing (Financial Data)
# -------------------------------
def download_stock_data(tickers, start, end, retries=3, delay=5):
    """Download adjusted closing prices with retries."""
    for i in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, auto_adjust=True)
            # Use 'Adj Close' if available; otherwise, 'Close'
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                raise ValueError("No suitable price column found.")
            # Check that each ticker has data
            if data.empty or any(data[ticker].empty for ticker in tickers):
                raise ValueError("Data is empty for one or more tickers.")
            return data.dropna()
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(delay)
    return pd.DataFrame()

# Download data for AAPL and MSFT from 2020 to end of 2023
tickers = ['AAPL', 'MSFT']
data = download_stock_data(tickers, start='2020-01-01', end='2023-12-31')
if data.empty:
    raise ValueError("Failed to download stock data for the specified tickers.")

print("Downloaded Financial Data:")
print(data.head())

# Compute daily log returns (log returns are typically stationary)
log_returns = np.log(data / data.shift(1)).dropna()

# Form bivariate observations by pairing same-day log returns
def pit_transform(series):
    ranks = rankdata(series, method='average')
    return ranks / (len(ranks) + 1)

u_aapl = pit_transform(log_returns['AAPL'].values)
v_msft = pit_transform(log_returns['MSFT'].values)

features = compute_summary_features(u_aapl, v_msft)

# Compute summary features (SAME AS SIMULATION/CDC)

def compute_summary_features(U, V):
    tau_emp, _ = kendalltau(U, V)
    rho_emp, _ = spearmanr(U, V)

    # Upper tail dependence
    upper_thr = 0.95
    upper_tail_dep = np.mean((U > upper_thr) & (V > upper_thr))

    # Lower tail dependence (the new feature)
    lower_thr = 0.05
    lower_tail_dep = np.mean((U < lower_thr) & (V < lower_thr))

    corr = np.corrcoef(U, V)[0,1]

    # Return array with all 5 features
    return np.array([tau_emp, rho_emp, upper_tail_dep, lower_tail_dep, corr])

# -------------------------------
# 2. Copula Functions (SAME AS SIMULATION/CDC)
# -------------------------------
## Gumbel Copula
def phi_gumbel(t, theta): return (-np.log(t))**theta
def dphi_gumbel(t, theta): return - (theta / t) * ((-np.log(t))**(theta-1))
def phi_gumbel_inv(x, theta): return np.exp(-x**(1.0/theta))

## Joe Copula (ADDED)
def phi_joe(t, theta): return -np.log(1.0 - (1.0 - t)**theta)
def dphi_joe(t, theta):
    num = theta * (1.0 - t)**(theta - 1)
    den = 1.0 - (1.0 - t)**theta + 1e-15
    return num / den
def phi_joe_inv(x, theta): return 1.0 - (1.0 - np.exp(-x))**(1.0/theta)

## A1 Copula
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

## A2 Copula
def phi_A2(t, theta): return ((1.0/t) * (1.0 - t)**2)**theta
def dphi_A2(t, theta):
    g = (1.0 - t)**2 / t
    num = -(1.0 - t)*(1.0 + t)
    den = t*t
    gprime = num/den
    return theta * (g**(theta-1)) * gprime
def phi_A2_inv(y, theta):
    a = 2.0 + y**(1.0/theta)
    return (a - np.sqrt(a*a - 4.0)) / 2.0

# Copula dictionary (SAME 4 AS SIMULATION/CDC)
copulas = {
    "Gumbel": {"phi": phi_gumbel, "dphi": dphi_gumbel, "phi_inv": phi_gumbel_inv},
    "Joe":    {"phi": phi_joe,    "dphi": dphi_joe,    "phi_inv": phi_joe_inv},
    "A1":     {"phi": phi_A1,     "dphi": dphi_A1,     "phi_inv": phi_A1_inv},
    "A2":     {"phi": phi_A2,     "dphi": dphi_A2,     "phi_inv": phi_A2_inv},
}

# -------------------------------
# 3. Sampling Functions (SAME AS SIMULATION/CDC)
# -------------------------------
def K_function(x, phi, dphi, theta):
    return x - (phi(x, theta) / (dphi(x, theta) + 1e-15))

def K_inverse(t, phi, dphi, theta, tol=1e-9):
    left, right = 1e-14, 1.0 - 1e-14
    def f(x): return K_function(x, phi, dphi, theta) - t
    return bisect(f, left, right, xtol=tol)

def sample_archimedean_copula_alg1(phi, dphi, phi_inv, theta, n=3000, seed=None):
    if seed: np.random.seed(seed)
    s = np.random.rand(n)
    t = np.random.rand(n)
    U, V = np.empty(n), np.empty(n)
    for i in range(n):
        w = K_inverse(t[i], phi, dphi, theta)
        phi_w = phi(w, theta)
        U[i] = phi_inv(s[i]*phi_w, theta)
        V[i] = phi_inv((1-s[i])*phi_w, theta)
    return U, V

# -------------------------------
# 4. Data Generation (SAME AS SIMULATION/CDC)
# -------------------------------
def generate_training_data(n_samples=500, sample_size=5000, theta_range=(1.0,20.0)):
    X_list, y_list, oh_list = [], [], []
    families = list(copulas.keys())
    for name in families:
        funcs = copulas[name]
        thetas = np.linspace(theta_range[0], theta_range[1], n_samples)
        for θ in thetas:
            U, V = sample_archimedean_copula_alg1(
                funcs["phi"], funcs["dphi"], funcs["phi_inv"], θ, n=sample_size
            )
            feats = compute_summary_features(U, V)
            X_list.append(feats)
            y_list.append(θ)
            one_hot = np.zeros(len(families))
            one_hot[families.index(name)] = 1.0
            oh_list.append(one_hot)
    X = np.hstack([np.array(X_list), np.array(oh_list)])
    y = np.array(y_list)
    return X, y, families

# Generate simulated data
X_sim, y_sim, copula_types = generate_training_data(sample_size=5000)

# -------------------------------
# 5. Model Training (SAME AS SIMULATION/CDC)
# -------------------------------
# Split simulated data
X_train, X_val, y_train, y_val = train_test_split(
    X_sim, y_sim, test_size=0.2, random_state=123
)

# Scale using training data only
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)

# Define model (SAME ARCHITECTURE)
def create_theta_estimator_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(64,  activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(1, activation='softplus'),
        layers.Lambda(lambda x: x + 1.0)  # θ ≥ 1
    ])
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss='mse')
    return model

# Create and train model
input_dim = X_train_s.shape[1]
model = create_theta_estimator_model(input_dim)

# Train with early stopping (SAME AS CDC)
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# -------------------------------
# 6. Bootstrap SE (SAME AS SIMULATION/CDC)
# -------------------------------
def bootstrap_theta_se(U, V, model, scaler, one_hot, B=100):
    n = len(U)
    boot = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        feats = compute_summary_features(U[idx], V[idx])
        iv = np.hstack([feats, one_hot]).reshape(1, -1)
        pred = model.predict(scaler.transform(iv), verbose=0)[0,0]
        boot.append(pred)
    return np.std(boot)

# -------------------------------
# 7. Estimation on Real Data
# -------------------------------
results = {}
se_results = {}
for i, fam in enumerate(copula_types):
    one_hot = np.zeros(len(copula_types))
    one_hot[i] = 1.0
    iv = np.hstack([features, one_hot]).reshape(1, -1)
    iv_scaled = scaler.transform(iv)
    pred_theta = model.predict(iv_scaled, verbose=0)[0,0]
    results[fam] = pred_theta
    se_results[fam] = bootstrap_theta_se(u_aapl, v_msft, model, scaler, one_hot, B=100)

# Print results
print("\nIGNIS Network Estimates for Financial Data:")
print("Copula    | θ Estimate | SE")
print("--------- | ---------- | ---------")
for fam in copula_types:
    print(f"{fam:<8} | {results[fam]:>9.4f} | {se_results[fam]:.4f}")


