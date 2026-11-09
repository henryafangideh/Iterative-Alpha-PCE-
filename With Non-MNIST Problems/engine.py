"""
Batched statevector engine for IαPCE-QNN benchmarking.

Models
  PCE : k-body Pauli-string encoding on n = min{n : 3*C(n,k) >= m} qubits,
        readout z_i = tanh(alpha <Pi_i>)                    (this work)
  AE  : "ordinary" entangled angle encoding, N qubits for N features,
        readout z_j = <Z_j>                                 (baseline)
Both share: HEA (Ry + CZ chain, p layers), logistic head refit in closed form
for every candidate theta, COBYLA outer optimiser, identical evaluation budget.
"""
import warnings
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def min_qubits(m, k):
    n = k
    while 3 * comb(n, k) < m:
        n += 1
    return n


def pauli_labels(m, n, k):
    labels = []
    for P in ("X", "Y", "Z"):
        for idx in combinations(range(n), k):
            s = ["I"] * n
            for q in idx:
                s[q] = P
            labels.append("".join(s))
    assert len(labels) >= m
    return labels[:m]


def pauli_matrix(label):
    """label[q] acts on qubit q; qubit 0 is least significant (Qiskit order)."""
    M = np.array([[1.0 + 0j]])
    for q in range(len(label) - 1, -1, -1):
        M = np.kron(M, _PAULI[label[q]])
    return M


def ry(psi, theta, q, n):
    """Apply Ry(theta) on qubit q to a batch psi of shape (S, 2**n).
    theta may be a scalar or a per-sample array of shape (S,)."""
    S = psi.shape[0]
    t = psi.reshape((S,) + (2,) * n)
    ax = 1 + (n - 1 - q)
    a, b = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
    th = np.asarray(theta, dtype=float)
    if th.ndim == 0:
        c, s = np.cos(th / 2), np.sin(th / 2)
    else:
        sh = (S,) + (1,) * (n - 1)
        c, s = np.cos(th / 2).reshape(sh), np.sin(th / 2).reshape(sh)
    return np.stack([c * a - s * b, s * a + c * b], axis=ax).reshape(S, -1)


def cz_chain_diag(n):
    idx = np.arange(2 ** n)
    sign = np.ones(2 ** n)
    for q in range(n - 1):
        sign *= 1 - 2 * (((idx >> q) & 1) & ((idx >> (q + 1)) & 1))
    return sign


class PCEModel:
    def __init__(self, m, k=2, p=3, beta=np.pi / 4):
        self.m, self.k, self.p, self.beta = m, k, p, beta
        self.n = min_qubits(m, k)
        self.dim = 2 ** self.n
        self.P = np.stack([pauli_matrix(l) for l in pauli_labels(m, self.n, k)])
        self.cz = cz_chain_diag(self.n)
        psi = np.zeros((1, self.dim), dtype=complex)
        psi[0, 0] = 1.0
        for q in range(self.n):                # symmetry-breaking preparation
            psi = ry(psi, np.pi / 4, q, self.n)
        self.psi0 = (psi * self.cz)[0]
        self.n_params = p * self.n

    def raw_features(self, X, theta):
        n = self.n
        psi = np.tile(self.psi0, (X.shape[0], 1))
        for i in range(self.m):                # exp(-i beta x_i Pi_i)
            phi = X[:, i] * self.beta
            Pp = psi @ self.P[i].T
            psi = np.cos(phi)[:, None] * psi - 1j * np.sin(phi)[:, None] * Pp
        for layer in np.asarray(theta).reshape(self.p, n):
            for q in range(n):
                psi = ry(psi, layer[q], q, n)
            psi = psi * self.cz
        Ppsi = np.einsum("mkd,sd->smk", self.P, psi)
        return np.real(np.einsum("sk,smk->sm", psi.conj(), Ppsi))

    def features(self, X, theta, alpha):
        return np.tanh(alpha * self.raw_features(X, theta))


class AEModel:
    """Ordinary entangled-angle encoding: one qubit per feature."""
    def __init__(self, N, p=3):
        self.N, self.n, self.p = N, N, p
        self.dim = 2 ** N
        self.cz = cz_chain_diag(N)
        idx = np.arange(self.dim)
        self.zs = np.stack([1 - 2 * ((idx >> q) & 1) for q in range(N)]).astype(float)
        self.n_params = p * N

    def features(self, X, theta, alpha=None):
        n = self.n
        psi = np.full((X.shape[0], self.dim), 1 / np.sqrt(self.dim), dtype=complex)
        for q in range(n):
            psi = ry(psi, X[:, q] * np.pi / 2, q, n)
        psi = psi * self.cz
        for layer in np.asarray(theta).reshape(self.p, n):
            for q in range(n):
                psi = ry(psi, layer[q], q, n)
            psi = psi * self.cz
        return (np.abs(psi) ** 2) @ self.zs.T


def _head(Z, y):
    return LogisticRegression(max_iter=500).fit(Z, y)


def make_loss(model, Xtr, ytr, classes):
    def loss(tv, alpha=None):
        Z = model.features(Xtr, tv, alpha)
        h = _head(Z, ytr)
        return log_loss(ytr, h.predict_proba(Z), labels=classes)
    return loss


def evaluate(model, tv, alpha, Xtr, ytr, Xte, yte):
    Ztr = model.features(Xtr, tv, alpha)
    h = _head(Ztr, ytr)
    pred = h.predict(model.features(Xte, tv, alpha))
    return {
        "acc": float(np.mean(pred == yte)),
        "f1": float(f1_score(yte, pred, average="macro")),
        "train_acc": float(h.score(Ztr, ytr)),
    }


def train_schedule(model, loss, theta0, schedule, iters):
    v = np.array(theta0, dtype=float)
    for a in schedule:
        v = minimize(loss, v, args=(a,), method="COBYLA",
                     options={"maxiter": iters, "rhobeg": 0.4}).x
    return v


# ----------------------------------------------------------------- data
def load_task(name):
    if name == "wdbc":
        d = load_breast_cancer()
        return d.data, d.target
    if name == "wine":
        d = load_wine()
        return d.data, d.target
    if name.startswith("digits"):
        d = load_digits()
        X, y = d.data, d.target
        spec = name.split("_")[1] if "_" in name else "all"
        if "v" in spec:
            a, b = (int(s) for s in spec.split("v"))
            mask = (y == a) | (y == b)
            X, y = X[mask], (y[mask] == b).astype(int)
        return X, y
    raise ValueError(name)


def preprocess(Xtr, Xte, m=12):
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=m, random_state=0).fit(sc.transform(Xtr))
    Ptr, Pte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    mm = MinMaxScaler((-1, 1)).fit(Ptr)
    return mm.transform(Ptr), np.clip(mm.transform(Pte), -1, 1)


SCHED_SAT = [0.5, 2.0, 6.0, 16.0]
SCHED_MOD = [0.5, 1.0, 2.0, 4.0]
ITERS = 70


def run_one(task, seed, ntr, nte, m=12, k=2, p=3, methods=None):
    """One (task, seed): fresh stratified split, train-only preprocessing,
    shared theta0, identical total COBYLA budget (4 x 70 = 280) for every method."""
    X, y = load_task(task)
    nte = min(nte, len(y) - ntr)
    Xtr0, Xte0, ytr, yte = train_test_split(
        X, y, train_size=ntr, test_size=nte, stratify=y, random_state=seed)
    Xtr, Xte = preprocess(Xtr0, Xte0, m)
    classes = np.unique(ytr)
    rng = np.random.default_rng(seed)
    out = {"task": task, "seed": seed, "n_classes": int(len(classes))}

    pce = PCEModel(m, k, p)
    th0 = rng.uniform(-0.4, 0.4, pce.n_params)
    loss = make_loss(pce, Xtr, ytr, classes)
    out["n_qubits_pce"] = pce.n
    total = ITERS * len(SCHED_SAT)

    runs = {
        "IaPCE_sat16": (SCHED_SAT, ITERS, 16.0),
        "fixed_sat16": ([16.0], total, 16.0),
        "IaPCE_mod4": (SCHED_MOD, ITERS, 4.0),
        "fixed_mod4": ([4.0], total, 4.0),
    }
    for name, (sched, it, a_final) in runs.items():
        if methods and name not in methods:
            continue
        v = train_schedule(pce, loss, th0, sched, it)
        r = evaluate(pce, v, a_final, Xtr, ytr, Xte, yte)
        r["final_loss"] = float(loss(v, a_final))
        out[name] = r

    for N in (4, 8):
        name = f"AE_{N}q"
        if methods and name not in methods:
            continue
        ae = AEModel(N, p)
        th0a = np.random.default_rng(seed).uniform(-0.4, 0.4, ae.n_params)
        la = make_loss(ae, Xtr[:, :N], ytr, classes)
        v = train_schedule(ae, la, th0a, [None], total)
        r = evaluate(ae, v, None, Xtr[:, :N], ytr, Xte[:, :N], yte)
        r["final_loss"] = float(la(v))
        out[name] = r

    # classical reference on the same PCA features (not a competitor; a ceiling)
    for N in (4, 8, 12):
        h = LogisticRegression(max_iter=2000).fit(Xtr[:, :N], ytr)
        out[f"classical_LR_{N}f"] = {"acc": float(h.score(Xte[:, :N], yte))}
    return out


def run_legacy_digits01(seed):
    """Exact replica of the original notebook protocol (leaky PCA/MinMax fit on
    all data, fixed split random_state=0, 60 train / 40 test) used ONLY to
    validate this engine against the numbers already in the paper."""
    d = load_digits(n_class=2)
    X = PCA(n_components=12, random_state=0).fit_transform(d.data)
    X = MinMaxScaler((-1, 1)).fit_transform(X)
    y = d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Xtr, ytr, Xte, yte = Xtr[:60], ytr[:60], Xte[:40], yte[:40]
    pce = PCEModel(12, 2, 3)
    th0 = np.random.default_rng(seed).uniform(-0.4, 0.4, pce.n_params)
    loss = make_loss(pce, Xtr, ytr, np.array([0, 1]))
    res = {}
    for name, sched, a_f, tot in [
        ("IaPCE_sat16", [0.5, 2, 6, 16], 16.0, 70),
        ("fixed_sat16", [16.0], 16.0, 280),
        ("IaPCE_mod4", [0.5, 1, 2, 4], 4.0, 70),
        ("fixed_mod4", [4.0], 4.0, 280),
    ]:
        v = train_schedule(pce, loss, th0, sched, tot)
        res[name] = evaluate(pce, v, a_f, Xtr, ytr, Xte, yte)["acc"]
    return res
