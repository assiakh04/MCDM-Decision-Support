
import numpy as np
import pandas as pd
from scipy.optimize import linprog

EPS = 1e-12

RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56
}

def _as_array(df):
    return np.asarray(df, dtype=float)

def _weight_series(weights, columns):
    w = np.asarray(weights, dtype=float)
    if len(w) != len(columns):
        raise ValueError("Le nombre de poids doit être égal au nombre de critères.")
    if np.any(w < 0):
        raise ValueError("Les poids doivent être positifs ou nuls.")
    s = w.sum()
    if s <= EPS:
        w = np.ones(len(columns)) / len(columns)
    else:
        w = w / s
    return pd.Series(w, index=columns, name="Poids")

def minmax_normalize(df, directions):
    """Normalisation [0,1] pour bénéfice/coût."""
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        x = df[col].astype(float)
        xmin, xmax = x.min(), x.max()
        den = xmax - xmin
        if abs(den) <= EPS:
            out[col] = 1.0
        elif directions[col] == "Bénéfice (+)":
            out[col] = (x - xmin) / den
        else:
            out[col] = (xmax - x) / den
    return out

def simple_ratio_normalize(df, directions):
    """
    Normalisation WSM/WPM/WASPAS suivant le cours :
    bénéfice -> x/max(x), coût -> min(x)/x.
    Les valeurs doivent être strictement positives.
    """
    if (df.astype(float) <= 0).any().any():
        raise ValueError(
            "WSM/WPM/WASPAS utilisent ici la normalisation du cours "
            "(x/max ou min/x), donc les valeurs doivent être strictement positives."
        )
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        x = df[col].astype(float)
        if directions[col] == "Bénéfice (+)":
            out[col] = x / max(x.max(), EPS)
        else:
            out[col] = max(x.min(), EPS) / x
    return out

def equal_weights(columns):
    return pd.Series(
        np.ones(len(columns)) / len(columns),
        index=columns,
        name="Poids"
    )

def entropy_weights(df, directions):
    """
    Le PDF donne la formule d'entropie à partir d'une matrice déjà normalisée.
    L'application utilise explicitement une normalisation min-max bénéfice/coût
    avant l'étape entropie.
    """
    R = minmax_normalize(df, directions)
    X = R.to_numpy(dtype=float)

    col_sums = X.sum(axis=0)
    P = np.zeros_like(X, dtype=float)

    for j in range(X.shape[1]):
        if col_sums[j] <= EPS:
            P[:, j] = 1.0 / X.shape[0]
        else:
            P[:, j] = X[:, j] / col_sums[j]

    m = X.shape[0]
    k = 1.0 / np.log(m) if m > 1 else 1.0
    plogp = np.where(P > 0, P * np.log(P), 0.0)
    E = -k * plogp.sum(axis=0)
    D = 1 - E

    if D.sum() <= EPS:
        w = np.ones(len(D)) / len(D)
    else:
        w = D / D.sum()

    weights = pd.Series(w, index=df.columns, name="Poids")
    details = pd.DataFrame({
        "Entropie E_j": E,
        "Diversification 1-E_j": D,
        "Poids": w
    }, index=df.columns)
    return weights, R, details

def critic_weights(df, directions):
    R = minmax_normalize(df, directions)

    sigma = R.std(axis=0, ddof=1).fillna(0.0)
    corr = R.corr().fillna(0.0)
    conflict = (1 - corr.abs()).sum(axis=1)
    C = sigma * conflict

    if C.sum() <= EPS:
        w = np.ones(len(C)) / len(C)
    else:
        w = (C / C.sum()).to_numpy()

    weights = pd.Series(w, index=df.columns, name="Poids")
    details = pd.DataFrame({
        "Écart-type σ_j": sigma,
        "Indépendance Σ(1-|ρ|)": conflict,
        "Information C_j": C,
        "Poids": weights
    }, index=df.columns)

    return weights, R, corr, details

def ahp_weights(pairwise):
    A = np.asarray(pairwise, dtype=float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("La matrice AHP doit être carrée.")
    if np.any(A <= 0):
        raise ValueError("Les comparaisons AHP doivent être strictement positives.")

    col_sums = A.sum(axis=0)
    norm = A / col_sums
    w = norm.mean(axis=1)
    w = w / w.sum()

    n = A.shape[0]
    lambda_max = float(np.dot(col_sums, w))
    ci = 0.0 if n <= 2 else (lambda_max - n) / (n - 1)
    ri = RI_TABLE.get(n, np.nan)
    cr = 0.0 if n <= 2 else (ci / ri if ri and not np.isnan(ri) else np.nan)

    return w, {
        "lambda_max": lambda_max,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "normalized_matrix": norm
    }

def bwm_weights(criteria, best, worst, best_to_others, others_to_worst):
    n = len(criteria)
    if best == worst:
        raise ValueError("Le critère Best et le critère Worst doivent être différents.")

    idx = {c: i for i, c in enumerate(criteria)}
    B = idx[best]
    W = idx[worst]

    # variables : w_1,...,w_n, xi
    c = np.zeros(n + 1)
    c[-1] = 1.0

    A_ub = []
    b_ub = []

    for j, crit in enumerate(criteria):
        aBj = float(best_to_others[crit])
        ajW = float(others_to_worst[crit])

        row = np.zeros(n + 1)
        row[B] = 1
        row[j] -= aBj
        row[-1] = -1
        A_ub.append(row); b_ub.append(0)

        row = np.zeros(n + 1)
        row[B] = -1
        row[j] += aBj
        row[-1] = -1
        A_ub.append(row); b_ub.append(0)

        row = np.zeros(n + 1)
        row[j] = 1
        row[W] -= ajW
        row[-1] = -1
        A_ub.append(row); b_ub.append(0)

        row = np.zeros(n + 1)
        row[j] = -1
        row[W] += ajW
        row[-1] = -1
        A_ub.append(row); b_ub.append(0)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * (n + 1)

    res = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        raise RuntimeError("Le programme linéaire BWM n'a pas convergé : " + res.message)

    w = res.x[:n]
    xi = float(res.x[-1])

    return pd.Series(w, index=criteria, name="Poids"), xi

def wsm(df, directions, weights):
    w = _weight_series(weights, df.columns)
    R = simple_ratio_normalize(df, directions)
    scores = R.mul(w, axis=1).sum(axis=1)
    return scores.rename("Score WSM"), R

def wpm(df, directions, weights):
    w = _weight_series(weights, df.columns)
    R = simple_ratio_normalize(df, directions).clip(lower=EPS)
    scores = np.exp(np.log(R).mul(w, axis=1).sum(axis=1))
    return scores.rename("Score WPM"), R

def waspas(df, directions, weights, lam=0.5):
    if not (0 <= lam <= 1):
        raise ValueError("λ doit être compris entre 0 et 1.")
    q1, R = wsm(df, directions, weights)
    q2, _ = wpm(df, directions, weights)
    q = lam * q1 + (1 - lam) * q2
    return q.rename("Score WASPAS"), q1, q2, R

def topsis(df, directions, weights):
    w = _weight_series(weights, df.columns)
    X = _as_array(df)

    den = np.sqrt((X ** 2).sum(axis=0))
    den = np.where(den <= EPS, 1.0, den)
    R = X / den
    V = R * w.to_numpy()

    ideal_plus = np.zeros(df.shape[1])
    ideal_minus = np.zeros(df.shape[1])

    for j, col in enumerate(df.columns):
        if directions[col] == "Bénéfice (+)":
            ideal_plus[j] = V[:, j].max()
            ideal_minus[j] = V[:, j].min()
        else:
            ideal_plus[j] = V[:, j].min()
            ideal_minus[j] = V[:, j].max()

    s_plus = np.sqrt(((V - ideal_plus) ** 2).sum(axis=1))
    s_minus = np.sqrt(((V - ideal_minus) ** 2).sum(axis=1))
    rc = s_minus / np.where((s_plus + s_minus) <= EPS, 1.0, s_plus + s_minus)

    details = pd.DataFrame({
        "S+": s_plus,
        "S-": s_minus,
        "RC": rc
    }, index=df.index)

    return pd.Series(rc, index=df.index, name="RC TOPSIS"), details, pd.DataFrame(V, index=df.index, columns=df.columns)

def vikor(df, directions, weights, v=0.5):
    w = _weight_series(weights, df.columns)
    X = _as_array(df)
    f_star = np.zeros(df.shape[1])
    f_minus = np.zeros(df.shape[1])

    for j, col in enumerate(df.columns):
        if directions[col] == "Bénéfice (+)":
            f_star[j] = X[:, j].max()
            f_minus[j] = X[:, j].min()
        else:
            f_star[j] = X[:, j].min()
            f_minus[j] = X[:, j].max()

    den = f_star - f_minus
    terms = np.zeros_like(X, dtype=float)

    for j in range(df.shape[1]):
        if abs(den[j]) <= EPS:
            terms[:, j] = 0.0
        else:
            terms[:, j] = w.iloc[j] * (f_star[j] - X[:, j]) / den[j]

    S = terms.sum(axis=1)
    R = terms.max(axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    sden = S_minus - S_star
    rden = R_minus - R_star

    q_s = np.zeros_like(S) if abs(sden) <= EPS else (S - S_star) / sden
    q_r = np.zeros_like(R) if abs(rden) <= EPS else (R - R_star) / rden

    Q = v * q_s + (1 - v) * q_r

    details = pd.DataFrame({
        "S": S,
        "R": R,
        "Q": Q
    }, index=df.index)

    return pd.Series(Q, index=df.index, name="Q VIKOR"), details

def ranking_from_scores(scores, higher_is_better=True):
    return scores.rank(
        ascending=not higher_is_better,
        method="min"
    ).astype(int)

def compare_rankings(df, directions, weights, lam=0.5, v=0.5):
    out = {}

    s, _ = wsm(df, directions, weights)
    out["WSM"] = ranking_from_scores(s, True)

    s, _ = wpm(df, directions, weights)
    out["WPM"] = ranking_from_scores(s, True)

    s, *_ = waspas(df, directions, weights, lam)
    out["WASPAS"] = ranking_from_scores(s, True)

    s, *_ = topsis(df, directions, weights)
    out["TOPSIS"] = ranking_from_scores(s, True)

    s, _ = vikor(df, directions, weights, v)
    out["VIKOR"] = ranking_from_scores(s, False)

    ranks = pd.DataFrame(out)
    return ranks

def spearman_matrix(ranks):
    # Corrélation de Pearson appliquée aux rangs = Spearman
    return ranks.corr(method="pearson")
