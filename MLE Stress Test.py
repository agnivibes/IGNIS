# mle_stress_test.py
# -------------------------------------------------------------
# Archimedean copulas (A1, A2): MLE stress tests in [1,20]
# - Raw MLE (untrimmed)
# - θ-dependent trimmed objective (safe evaluator)
# - Fixed-mask MLE (trim once; not the true likelihood)
#
# Requirements: numpy, scipy, matplotlib (optional for plots)
# -------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize, bisect
import math
import warnings
warnings.filterwarnings("ignore")

# ------------- utilities / reproducibility -------------------
def set_seed(seed=123):
    np.random.seed(seed)

# ------------- copula generators (θ ≥ 1) ---------------------
def phi_A1(t, theta):
    f = t**(1.0/theta) + t**(-1.0/theta) - 2.0
    return f**theta

def dphi_A1(t, theta):
    a = 1.0/theta
    f  = t**a + t**(-a) - 2.0
    fp = a*t**(a-1.0) - a*t**(-a-1.0)
    return theta * (f**(theta-1.0)) * fp

def d2phi_A1(t, theta):
    a   = 1.0/theta
    f   = t**a + t**(-a) - 2.0
    fp  = a*t**(a-1.0) - a*t**(-a-1.0)
    fpp = a*(a-1.0)*t**(a-2.0) + a*(a+1.0)*t**(-a-2.0)
    return theta*(theta-1.0)*(f**(theta-2.0))*(fp**2) + theta*(f**(theta-1.0))*fpp

def phi_A1_inv(y, theta):
    # exact inverse; assumes y >= 0
    a = np.power(np.maximum(y, 0.0), 1.0/theta) + 2.0
    disc = np.maximum(a*a - 4.0, 0.0)
    inner = (a - np.sqrt(disc)) / 2.0
    return np.power(np.clip(inner, 0.0, 1.0), theta)

def phi_A2(t, theta):
    return ((1.0 - t)**2 / t)**theta

def dphi_A2(t, theta):
    g = (1.0 - t)**2 / t
    gp = 1.0 - 1.0/(t*t)
    return theta * (g**(theta-1.0)) * gp

def d2phi_A2(t, theta):
    g   = (1.0 - t)**2 / t
    gp  = 1.0 - 1.0/(t*t)
    gpp = 2.0/(t**3)
    return theta*(theta-1.0)*(g**(theta-2.0))*(gp**2) + theta*(g**(theta-1.0))*gpp

def phi_A2_inv(y, theta):
    x = np.power(np.maximum(y, 0.0), 1.0/theta)
    disc = np.maximum((x + 2.0)**2 - 4.0, 0.0)
    w = (x + 2.0 - np.sqrt(disc)) / 2.0
    return np.clip(w, 0.0, 1.0)

COP = {
    "A1": {"phi": phi_A1, "dphi": dphi_A1, "d2phi": d2phi_A1, "inv": phi_A1_inv},
    "A2": {"phi": phi_A2, "dphi": dphi_A2, "d2phi": d2phi_A2, "inv": phi_A2_inv},
}

# ------------- sampling (McNeil's Alg. 1 via K^{-1}) ---------
def K_function(x, phi, dphi, theta):
    return x - (phi(x, theta) / (dphi(x, theta) + 1e-300))

def K_inverse(t, phi, dphi, theta, tol=1e-10):
    left, right = 1e-14, 1.0 - 1e-14
    f = lambda x: K_function(x, phi, dphi, theta) - t
    return bisect(f, left, right, xtol=tol)

def sample_archimedean_copula(phi, dphi, inv, theta, n=5000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    s_vals = np.random.rand(n)
    t_vals = np.random.rand(n)
    u = np.empty(n); v = np.empty(n)
    for i in range(n):
        w = K_inverse(t_vals[i], phi, dphi, theta)
        phi_w = phi(w, theta)
        u[i] = inv(s_vals[i]*phi_w, theta)
        v[i] = inv((1.0 - s_vals[i])*phi_w, theta)
    return u, v

# ---------- log-density via inverse-function theorem ----------
def log_density_raw(U, V, theta, name):
    """Return log c(u,v;theta) termwise (no floors), may contain ±inf/NaN."""
    phi  = COP[name]["phi"]; d1 = COP[name]["dphi"]; d2 = COP[name]["d2phi"]; inv = COP[name]["inv"]
    # do not clip U,V here (raw)
    Su = phi(U, theta); Sv = phi(V, theta)
    W  = inv(Su + Sv, theta)

    phi_p_u = d1(U, theta)
    phi_p_v = d1(V, theta)
    phi_p_w = d1(W, theta)
    phi_pp_w = d2(W, theta)

    # psi''(w) = -φ''(w)/[φ'(w)]^3 ; c = psi''(w)*(-φ'(u))*(-φ'(v))
    with np.errstate(all="ignore"):
        psi_pp = -phi_pp_w / np.power(phi_p_w, 3.0)
        dens   = psi_pp * (-phi_p_u) * (-phi_p_v)
        logc   = np.log(dens)
    return logc  # may be nan/±inf

# --------- stabilized evaluator (θ-dependent trimming) --------
def loglik_theta_trimmed(U, V, theta, name, eps=1e-10, tiny=1e-300):
    """Clip inputs; compute raw density; use mask for finite/positive terms only.
       Returns: total_loglik (using floor for invalid terms) and valid_ratio."""
    # A1 benefits from a slightly stronger clip
    eps_here = max(eps, 1e-10 if name == "A1" else 1e-12)
    Uc = np.clip(U, eps_here, 1.0 - eps_here)
    Vc = np.clip(V, eps_here, 1.0 - eps_here)

    raw = log_density_raw(Uc, Vc, theta, name)
    mask = np.isfinite(raw)
    valid_ratio = float(np.mean(mask))
    # floor invalids to keep the scalar objective finite (this is *not* MLE)
    logc = np.where(mask, raw, np.log(tiny))
    return float(np.sum(logc)), valid_ratio

# -------------- fixed-mask evaluator (not MLE) ----------------
def build_fixed_mask(U, V, eps=1e-4):
    return (U > eps) & (U < 1.0 - eps) & (V > eps) & (V < 1.0 - eps)

def loglik_fixed_mask(U, V, mask, theta, name, tiny=1e-300):
    Uc, Vc = U[mask], V[mask]
    raw = log_density_raw(Uc, Vc, theta, name)
    # floor remaining invalids just so the objective is finite
    logc = np.where(np.isfinite(raw), raw, np.log(tiny))
    return float(np.sum(logc))

# -------------- finite-diff gradient (single param) -----------
def fd_grad(fun, theta, h=1e-3):
    t1 = max(1.0, theta - h)
    t2 = theta + h
    f1 = fun(t1)
    f2 = fun(t2)
    if not np.isfinite(f1) or not np.isfinite(f2):
        return np.nan
    return (f2 - f1) / (t2 - t1)

# ------------------------ ADAM for 1-D ------------------------
def adam_1d(fun, theta0=2.0, iters=500, lr=0.05, h=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, bounds=(1.0, 200.0)):
    m = 0.0; v = 0.0
    theta = float(theta0)
    best = (-np.inf, theta0)
    for t in range(1, iters+1):
        g = fd_grad(fun, theta, h=h)
        if not np.isfinite(g):
            return {"success": False, "reason": "nan_grad", "theta": theta, "best": best}
        m = beta1*m + (1.0 - beta1)*g
        v = beta2*v + (1.0 - beta2)*(g*g)
        mhat = m/(1.0 - beta1**t); vhat = v/(1.0 - beta2**t)
        theta += lr * mhat / (math.sqrt(vhat) + eps)
        theta = min(max(theta, bounds[0]), bounds[1])
        val = fun(theta)
        if np.isfinite(val) and val > best[0]:
            best = (val, theta)
        # tiny improvement break
        if t > 10 and np.isfinite(val) and abs(g) < 1e-5:
            return {"success": True, "reason": "small_grad", "theta": theta, "best": best}
    return {"success": True, "reason": "max_iters", "theta": theta, "best": best}

# ------------------ MLE wrappers (3 variants) -----------------
def mle_raw(U, V, name, theta0=2.0):
    # objective is NEGATIVE log-lik to minimize; we add a small penalty if inf/nan
    def obj(t):
        t = float(t[0])
        ll = np.sum(log_density_raw(U, V, t, name))
        if not np.isfinite(ll):
            return 1e50  # drive optimizer away / report failure rate
        return -ll
    res = minimize(obj, x0=[theta0], method="L-BFGS-B", bounds=[(1.0, 200.0)])
    return res

def mle_theta_trimmed(U, V, name, theta0=2.0):
    def ll(t):
        return loglik_theta_trimmed(U, V, float(t), name)[0]
    res = minimize(lambda x: -ll(x[0]), x0=[theta0], method="L-BFGS-B", bounds=[(1.0, 200.0)])
    return res

def mle_fixed_mask(U, V, name, eps=1e-4, theta0=2.0):
    mask = build_fixed_mask(U, V, eps=eps)
    def ll(t):
        return loglik_fixed_mask(U, V, mask, float(t), name)
    res = minimize(lambda x: -ll(x[0]), x0=[theta0], method="L-BFGS-B", bounds=[(1.0, 200.0)])
    return res

# ---------------------- experiment runner ---------------------
def run_once(name="A2", true_theta=5.0, n=3000, seed=0):
    phi = COP[name]["phi"]; dphi = COP[name]["dphi"]; inv = COP[name]["inv"]
    U, V = sample_archimedean_copula(phi, dphi, inv, true_theta, n=n, seed=seed)

    # 1) raw MLE (untrimmed)
    res_raw = mle_raw(U, V, name, theta0=2.0)
    outcome_raw = ("success" if res_raw.success and np.isfinite(res_raw.fun) else "fail",
                   float(res_raw.x[0]), res_raw.message)

    # 2) θ-dependent trimmed (safe) with L-BFGS-B
    res_trim = mle_theta_trimmed(U, V, name, theta0=2.0)
    outcome_trim = ("success" if res_trim.success else "fail",
                    float(res_trim.x[0]), res_trim.message)

    # ADAM on the same θ-dependent trimmed objective
    def ll_trim(t):  # maximize
        return loglik_theta_trimmed(U, V, t, name)[0]
    adam_trim = adam_1d(ll_trim, theta0=2.0, iters=500, lr=0.05, bounds=(1.0, 200.0))
    outcome_adam = ("success" if adam_trim["success"] else "fail",
                    float(adam_trim["theta"]), adam_trim["reason"])

    # 3) fixed masks (two choices) to show mask dependence
    res_mask1 = mle_fixed_mask(U, V, name, eps=1e-4, theta0=2.0)
    res_mask2 = mle_fixed_mask(U, V, name, eps=1e-3, theta0=2.0)
    out_m1 = float(res_mask1.x[0])
    out_m2 = float(res_mask2.x[0])

    return {
        "raw": outcome_raw,
        "trim_lbfgs": outcome_trim,
        "trim_adam": outcome_adam,
        "mask_eps1e-4": out_m1,
        "mask_eps1e-3": out_m2,
        "valid_ratio_at_true": loglik_theta_trimmed(U, V, true_theta, name)[1]
    }

def summary(name="A2", thetas=(2,5,10,15,20), reps=50, n=3000):
    print(f"\n=== {name}: MLE stress in [1,20] (n={n}, reps={reps}) ===")
    for th in thetas:
        fails_raw = 0
        boundary_raw = 0
        fails_trim = 0
        fails_adam = 0
        ests_mask1 = []
        ests_mask2 = []
        valid_ratios = []
        for r in range(reps):
            out = run_once(name=name, true_theta=float(th), n=n, seed=r)
            # raw:
            if out["raw"][0] != "success" or not np.isfinite(out["raw"][1]):
                fails_raw += 1
            if np.isfinite(out["raw"][1]) and abs(out["raw"][1] - 1.0) < 1e-6:
                boundary_raw += 1
            # trimmed:
            if out["trim_lbfgs"][0] != "success":
                fails_trim += 1
            if out["trim_adam"][0] != "success":
                fails_adam += 1
            ests_mask1.append(out["mask_eps1e-4"])
            ests_mask2.append(out["mask_eps1e-3"])
            valid_ratios.append(out["valid_ratio_at_true"])

        print(f"\n-- true θ = {th}")
        print(f"raw MLE:    fail%={100*fails_raw/reps:5.1f} | boundary@θ=1.0%={100*boundary_raw/reps:5.1f}")
        print(f"trim L-BFGS: fail%={100*fails_trim/reps:5.1f}")
        print(f"trim ADAM:   fail%={100*fails_adam/reps:5.1f}")
        m1 = np.median(ests_mask1); m2 = np.median(ests_mask2)
        print(f"fixed-mask MLE: median θ (ε=1e-4)={m1:5.2f} vs (ε=1e-3)={m2:5.2f}  Δ={abs(m1-m2):.2f}")
        print(f"valid% at true θ (trimmed evaluator): mean={100*np.mean(valid_ratios):.1f}%, std={100*np.std(valid_ratios):.1f}%")

# ---------------- optional: tiny plots ------------------------
def tiny_plots(name="A2"):
    import matplotlib.pyplot as plt
    # (1) valid% vs θ on one dataset
    set_seed(7)
    true_theta = 10.0
    phi = COP[name]["phi"]; dphi = COP[name]["dphi"]; inv = COP[name]["inv"]
    U, V = sample_archimedean_copula(phi, dphi, inv, true_theta, n=4000, seed=7)
    thetas = np.linspace(1.0, 20.0, 100)
    valid = []
    for t in thetas:
        _, v = loglik_theta_trimmed(U, V, t, name)
        valid.append(v)
    plt.figure(figsize=(5,3))
    plt.plot(thetas, np.array(valid)*100.0)
    plt.xlabel(r"$\theta$"); plt.ylabel("valid %")
    plt.title(f"{name}: θ-dependent valid fraction")
    plt.tight_layout()
    plt.show()

    # (2) objective kinks via finite-diff jump
    def LL(t): return loglik_theta_trimmed(U, V, t, name)[0]
    delta = 1e-3
    jumps = [abs(LL(t+delta) - LL(t-delta)) for t in thetas]
    plt.figure(figsize=(5,3))
    plt.plot(thetas, jumps)
    plt.xlabel(r"$\theta$"); plt.ylabel(r"|Δℓ| (δ=1e-3)")
    plt.title(f"{name}: finite-diff jump (kinks)")
    plt.tight_layout()
    plt.show()

# ------------------------------ main -------------------------
if __name__ == "__main__":
    set_seed(123)
    # run summaries for A2 (the reviewer’s focus); A1 available too
    summary(name="A2", thetas=(2,5,10,15,20), reps=50, n=3000)
    # Uncomment to check A1 as well:
    summary(name="A1", thetas=(2,5,10,15,20), reps=30, n=3000)

    tiny_plots("A2")
    tiny_plots("A1")
