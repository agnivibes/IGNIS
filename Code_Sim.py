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

# --- Copula generators ---

## Gumbel–Hougaard Copula (θ ≥ 1)
def phi_gumbel(t, theta):
    return (-np.log(t))**theta

def dphi_gumbel(t, theta):
    return - (theta / t) * ((-np.log(t))**(theta-1))

def phi_gumbel_inv(x, theta):
    return np.exp(- x**(1.0/theta))

## Joe Copula (θ ≥ 1)
def phi_joe(t, theta):
    return -np.log(1.0 - (1.0 - t)**theta)

def dphi_joe(t, theta):
    num = theta * (1.0 - t)**(theta-1)
    den = 1.0 - (1.0 - t)**theta
    return num / (den + 1e-15)

def phi_joe_inv(x, theta):
    return 1.0 - (1.0 - np.exp(-x))**(1.0/theta)

## A1 Copula (θ ≥ 1)
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

## A2 Copula (θ ≥ 1)
def phi_A2(t, theta):
    return ((1.0/t) * (1.0 - t)**2)**theta

def dphi_A2(t, theta):
    g = (1.0 - t)**2 / t
    num = -(1.0 - t) * (1.0 + t)
    den = t * t
    gprime = num / den
    return theta * (g**(theta-1)) * gprime

def phi_A2_inv(y, theta):
    a = 2.0 + y**(1.0/theta)
    return (a - np.sqrt(a*a - 4.0)) / 2.0

# copulas dictionary
copulas = {
    "Gumbel": {"phi": phi_gumbel, "dphi": dphi_gumbel, "phi_inv": phi_gumbel_inv},
    "Joe":    {"phi": phi_joe,    "dphi": dphi_joe,    "phi_inv": phi_joe_inv},
    "A1":     {"phi": phi_A1,     "dphi": dphi_A1,     "phi_inv": phi_A1_inv},
    "A2":     {"phi": phi_A2,     "dphi": dphi_A2,     "phi_inv": phi_A2_inv},
}

# K‐function and its inverse
def K_function(x, phi, dphi, theta):
    return x - (phi(x, theta) / (dphi(x, theta) + 1e-15))

def K_inverse(t, phi, dphi, theta, tol=1e-9):
    left, right = 1e-14, 1.0 - 1e-14
    def f(x):
        return K_function(x, phi, dphi, theta) - t
    return bisect(f, left, right, xtol=tol)

# Sampling
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

# Features
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

# Data generation
def generate_training_data(n_samples=500, sample_size=5000, theta_range=(1.0,20.0)):
    X_list, y_list, copula_list = [], [], []
    types = list(copulas.keys())
    for name in types:
        funcs = copulas[name]
        thetas = np.linspace(theta_range[0], theta_range[1], n_samples)
        for θ in thetas:
            U, V = sample_archimedean_copula_alg1(
                funcs["phi"], funcs["dphi"], funcs["phi_inv"],
                θ, n=sample_size
            )
            feat = compute_summary_features(U, V)
            X_list.append(feat)
            y_list.append(θ)
            one_hot = np.zeros(len(types))
            one_hot[types.index(name)] = 1.0
            copula_list.append(one_hot)
    X = np.hstack([np.array(X_list), np.array(copula_list)])
    y = np.array(y_list)
    return X, y, types

# Generate & split
set_seed(123)
X, y, copula_types = generate_training_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# Neural network 
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

# Train
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

# Bootstrap SE & evaluation 
def bootstrap_theta_se(U, V, model, scaler, one_hot, B=100):
    n = len(U)
    arr = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        feat = compute_summary_features(U[idx], V[idx])
        iv = np.hstack([feat, one_hot]).reshape(1,-1)
        arr.append(model.predict(scaler.transform(iv), verbose=0)[0,0])
    return np.std(arr)

for name in copula_types:
    funcs = copulas[name]
    true_theta = 10.0
    U, V = sample_archimedean_copula_alg1(
        funcs["phi"], funcs["dphi"], funcs["phi_inv"], true_theta, n=3000, seed=42
    )
    feat = compute_summary_features(U, V)
    one_hot = np.zeros(len(copula_types))
    one_hot[copula_types.index(name)] = 1.0
    est = theta_model.predict(scaler.transform(np.hstack([feat, one_hot]).reshape(1,-1)), verbose=0)[0,0]
    se = bootstrap_theta_se(U, V, theta_model, scaler, one_hot)
    print(f"{name:<10} True θ: {true_theta:.2f}  Est θ: {est:.2f}  SE: {se:.2f}")
