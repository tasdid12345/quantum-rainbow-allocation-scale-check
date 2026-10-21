"""Does a class-free, feature-engineered LightGBM Q-regressor match or beat a classical
MLP Q-network (Rainbow-DQN stand-in) on sequential multi-dimensional knapsack allocation
as the number of items grows? Everything is computed at run time."""
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # seeds for models, behaviour data and exploration
SIZES = [20, 50, 100]         # number of items
M = 3                         # resource constraints
N_INST = 200                  # synthetic instances per size
TEST_FRAC = 0.25              # instance-level hold-out fraction
N_TRANS = 8000                # transition budget (identical for both models)
FQI_ITERS = 5                 # fitted Q-iteration sweeps
GAMMA = 0.9
EPS = 0.3                     # behaviour-policy exploration rate
TOL = 1e-9
# name -> (model family, feature set)
METHODS = {"baseline": ("mlp", "raw"), "improved": ("lgbm", "eng")}
QUESTION = ("Does a class-free, feature-engineered gradient-boosting Q-function match or beat "
            "a classical MLP Q-network on multi-resource allocation as the number of items grows?")


def make_instances(n, rng):
    """Random multi-dimensional knapsack instances; capacity = 50% of total demand per resource."""
    v = rng.integers(1, 101, size=(N_INST, n)).astype(float)
    w = rng.integers(1, 101, size=(N_INST, M, n)).astype(float)
    return v, w, 0.5 * w.sum(axis=2)


def reference(v, w, cap):
    """Greedy by value density, then 1-add / 1-swap local search (reference heuristic)."""
    dens = v / (w / cap[:, None]).sum(0)
    sel, used = np.zeros(len(v), bool), np.zeros(M)
    for j in np.argsort(-dens):
        if np.all(used + w[:, j] <= cap):
            sel[j], used = True, used + w[:, j]
    improved = True
    while improved:
        improved = False
        for j in np.where(~sel)[0]:                      # 1-add
            if np.all(used + w[:, j] <= cap):
                sel[j], used, improved = True, used + w[:, j], True
        for i in np.where(sel)[0]:                       # 1-swap: i out, better j in
            base = used - w[:, i]
            ok = ~sel & (v > v[i]) & np.all(base[:, None] + w <= cap[:, None], axis=0)
            if ok.any():
                j = np.where(ok)[0][np.argmax(v[ok])]
                sel[i], sel[j], used, improved = False, True, base + w[:, j], True
                break
    return v[sel].sum()


def cand_feats(vn, wn, rem, sel, feat):
    """Features for every (state, candidate item) pair.
    vn (T,n) normalised values, wn (T,M,n) normalised weights, rem (T,M) remaining capacity,
    sel (T,n) bool selected mask. Returns (T,n,F) float32 features and (T,n) feasibility."""
    T, n = vn.shape
    wt = wn.transpose(0, 2, 1)                                       # (T,n,M)
    remB = np.broadcast_to(rem[:, None, :], wt.shape)                # (T,n,M)
    feas = np.all(wn <= rem[:, :, None] + TOL, axis=1) & ~sel        # (T,n)
    kk = np.broadcast_to((sel.sum(1) / n)[:, None, None], (T, n, 1))  # fraction chosen
    cols = [vn[..., None], wt, remB, kk]                             # raw: 8 features
    if feat == "eng":
        inv = 1.0 / np.maximum(wt, 1e-6)
        tight = (wt / np.maximum(remB, 1e-6)).sum(-1)                 # resource tightness
        dens = vn / np.maximum(tight, 1e-6)                           # value density
        nf = feas.sum(1, keepdims=True)
        drank = np.argsort(np.argsort(-np.where(feas, dens, -np.inf), 1), 1) / np.maximum(nf, 1)
        vrank = np.argsort(np.argsort(-vn, 1), 1) / n                 # static value rank
        cols += [np.clip(vn[..., None] * inv, 0, 50),                 # value / weight ratios
                 np.clip(remB * inv, 0, 50),                          # slack ratios
                 tight[..., None], dens[..., None], vrank[..., None], drank[..., None],
                 np.broadcast_to(nf / n, (T, n))[..., None]]
    return np.concatenate(cols, -1).astype(np.float32), feas


def make_model(family, seed):
    """Baseline: standardised MLP Q-network. Improved: LightGBM Q-regressor."""
    if family == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=200,
                                          random_state=seed))
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=seed, verbose=-1)


def collect(vn, wn, dens, pool, rng):
    """Fixed-budget transitions from training instances under an epsilon-greedy density policy."""
    rows = []
    while len(rows) < N_TRANS:
        for i in pool:
            rem, sel = np.ones(M), np.zeros(vn.shape[1], bool)
            while len(rows) < N_TRANS:
                feas = np.all(wn[i] <= rem[:, None] + TOL, axis=0) & ~sel
                if not feas.any():
                    break
                cand = np.where(feas)[0]
                a = rng.choice(cand) if rng.random() < EPS else cand[np.argmax(dens[i, cand])]
                rem_a = rem - wn[i][:, a]
                sel_a = sel.copy()
                sel_a[a] = True
                done = not np.any(np.all(wn[i] <= rem_a[:, None] + TOL, axis=0) & ~sel_a)
                rows.append((i, rem, sel, a, vn[i, a], rem_a, sel_a, done))
                rem, sel = rem_a, sel_a
    col = lambda k: np.array([r[k] for r in rows])
    I = col(0)
    return dict(Vt=vn[I], Wt=wn[I], REMB=np.stack(col(1)), SELB=np.stack(col(2)), A=col(3),
                R=col(4), REMA=np.stack(col(5)), SELA=np.stack(col(6)), DONE=col(7).astype(bool))


def fqi(family, feat, d, seed):
    """Fitted Q-iteration: refit a fresh regressor on Bellman targets for FQI_ITERS sweeps."""
    Xb, _ = cand_feats(d["Vt"], d["Wt"], d["REMB"], d["SELB"], feat)
    Xsa = Xb[np.arange(len(d["A"])), d["A"]]                         # features of taken actions
    Xn, feas_n = cand_feats(d["Vt"], d["Wt"], d["REMA"], d["SELA"], feat)
    Xn_rows = Xn[feas_n]
    model = make_model(family, seed).fit(Xsa, d["R"])
    for _ in range(FQI_ITERS - 1):
        qn = np.full(feas_n.shape, -np.inf)
        qn[feas_n] = model.predict(Xn_rows)
        maxq = np.where(feas_n.any(1), qn.max(1), 0.0)               # 0 when no item fits
        y = d["R"] + GAMMA * maxq * ~d["DONE"]
        model = make_model(family, seed).fit(Xsa, y)
    return model


def q_values(model, feat, vn, wn, rem, sel):
    """Q for every item in the current state; -inf for infeasible or already chosen items."""
    X, feas = cand_feats(vn, wn, rem, sel, feat)
    Q = np.full(feas.shape, -np.inf)
    if feas.any():
        Q[feas] = model.predict(X[feas])
    return Q, feas


def rollout(model, feat, vn, wn):
    """Greedy sequential allocation for all test instances at once (argmax Q, stop when nothing fits)."""
    K, n = vn.shape
    rem, sel, active = np.ones((K, M)), np.zeros((K, n), bool), np.ones(K, bool)
    for _ in range(n):
        Q, feas = q_values(model, feat, vn, wn, rem, sel)
        active &= feas.any(1)
        if not active.any():
            break
        rows = np.where(active)[0]
        a = Q[rows].argmax(1)
        rem[rows] -= wn[rows, :, a]
        sel[rows, a] = True
    return sel


def main():
    rng = np.random.default_rng(SEED)
    shape = (len(SEEDS), len(SIZES))
    res = {nm: {k: np.zeros(shape) for k in ("score", "feas", "time")} for nm in METHODS}
    n_train = n_test = 0
    for j, n in enumerate(SIZES):
        v, w, cap = make_instances(n, rng)
        perm = rng.permutation(N_INST)                               # split ONCE per size
        n_te = int(TEST_FRAC * N_INST)
        te, tr = perm[:n_te], perm[n_te:]
        n_train, n_test = n_train + len(tr), n_test + len(te)
        vn = v / v.max(1, keepdims=True)                             # per-instance normalisation
        wn = w / cap[:, :, None]
        dens = vn / wn.sum(1)
        ref = np.array([reference(v[i], w[i], cap[i]) for i in te])  # test-set reference values
        print(f"n={n}: {len(tr)} train / {len(te)} test instances, reference computed")
        for k, s in enumerate(SEEDS):
            d = collect(vn, wn, dens, tr, np.random.default_rng(s))  # same transitions for both
            for nm, (family, feat) in METHODS.items():
                t0 = time.time()
                model = fqi(family, feat, d, s)
                tt = time.time() - t0
                sel = rollout(model, feat, vn[te], wn[te])
                val = (v[te] * sel).sum(1)
                ok = np.all((w[te] * sel[:, None, :]).sum(2) <= cap[te] + 1e-9, axis=1)
                res[nm]["score"][k, j] = np.mean(val / ref)
                res[nm]["feas"][k, j] = ok.mean()
                res[nm]["time"][k, j] = tt
                print(f"  seed={s} {nm}: ratio={res[nm]['score'][k, j]:.4f} "
                      f"feas={ok.mean():.2f} train_time={tt:.1f}s")

    m = {}
    for nm in METHODS:
        head = res[nm]["score"].mean(1)                              # per-seed mean over sizes
        m[f"{nm}_score"] = float(head.mean())
        m[f"{nm}_score_std"] = float(head.std())
        m[f"{nm}_feasibility_rate"] = float(res[nm]["feas"].mean())
        m[f"{nm}_train_time_s"] = float(res[nm]["time"].sum(1).mean())
        for j, n in enumerate(SIZES):
            m[f"{nm}_score_n{n}"] = float(res[nm]["score"][:, j].mean())
    m["improvement"] = m["improved_score"] - m["baseline_score"]
    if not all(np.isfinite(x) for x in m.values()):
        print("Non-finite metrics; failing.")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    x, wd = np.arange(len(SIZES)), 0.35
    plt.figure(figsize=(6, 4))
    for off, nm in enumerate(METHODS):
        s = res[nm]["score"]
        plt.bar(x + off * wd, s.mean(0), wd, yerr=s.std(0), capsize=3, label=nm)
    plt.xticks(x + wd / 2, [f"n={n}" for n in SIZES])
    plt.ylabel("test value / reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/score_by_size.png", dpi=120)

    out = {
        "question": QUESTION,
        "method": ("Synthetic multi-dimensional knapsack MDP solved sequentially; fitted Q-iteration "
                   "with a fixed transition budget; both Q-functions trained on identical transitions; "
                   "evaluated by greedy rollout on held-out instances against a greedy+local-search reference."),
        "dataset": {
            "name": "Seeded synthetic multi-dimensional knapsack instances (numpy generator)",
            "source_url": "numpy synthetic generator (no download); models: https://scikit-learn.org, https://lightgbm.readthedocs.io",
            "n_rows": N_INST * len(SIZES), "n_train": n_train, "n_test": n_test,
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "mean achieved value / greedy+local-search reference value on test instances",
        "higher_is_better": True,
        "baseline": "sklearn MLPRegressor (64,64) on standardised raw state-action features; classical Rainbow-DQN stand-in without distributional, dueling or quantum parts.",
        "improved": "LightGBM regressor Q-function on engineered features (value-to-resource ratios, slack ratios, tightness, value and density ranks, feasible-item fraction, remaining capacity, item rank), same transitions and budget.",
        "metrics": m,
        "notes": ("Synthetic instances, not the paper's data; the paper's quantum model is not reproduced. "
                  "Hyper-parameters are fixed a priori and nothing was tuned on the test set. "
                  "Feasibility is enforced by action masking, so feasibility rate is a sanity check. "
                  "Reference is a heuristic, so ratios can exceed 1. Only 3 seeds and 50 test instances per size: differences are noisy."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
