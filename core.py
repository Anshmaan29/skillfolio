"""CareerTwin core maths. No Streamlit in here, so every function can be unit-tested."""
import json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
D = json.loads((ROOT / 'data' / 'careertwin.json').read_text())
for k in ('emb', 'xy', 'xyz', 'coef', 'intercept', 'role_profile', 'freq', 'clusters'): D[k] = np.array(D[k])
V, ROLES = D['vocab'], D['roles']; N = len(V)
IX = {s: i for i, s in enumerate(V)}
T = np.zeros((len(D['twins']['ix']), N), dtype=np.float32)          # 3,000 real postings for the KNN tab
for r, ix in enumerate(D['twins']['ix']): T[r, ix] = 1

def norm(s):
    """Clean a raw skill string the same way the training script did."""
    s = re.sub(r'\s+', ' ', s.lower().strip(' .,;:'))
    s = re.sub(r' (skills|programming|language)$', '', s)
    return D['alias'].get(s, s)

def probs(M):
    """Softmax of the trained logistic regression. M = rows of 0/1 skill flags."""
    z = M @ D['coef'].T + D['intercept']; z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def flags(idx):
    m = np.zeros(N); m[list(idx)] = 1; return m

def extract(text):
    """Fallback skill finder (exact phrases). Used only if the NER model files are missing."""
    t = ' ' + re.sub(r'\s+', ' ', text.lower()) + ' '
    return [s for s in V if len(s) > 2 and re.search(r'(?<![a-z0-9+#])' + re.escape(s) + r'(?![a-z0-9+#])', t)]

def effort(have):
    """Learning effort of every skill = distance from the nearest skill you already know (embedding cosine)."""
    return 1 - (D['emb'] @ D['emb'][list(have)].T).max(1)

def evolve(have, target, k, lam, gens=40, pop=60, seed=7, *, profile=None):
    """Genetic algorithm. A chromosome = k new skills. Fitness = role probability - lam * learning effort.
    Returns (best set, best fitness per generation, best set per generation)."""
    rng = np.random.default_rng(seed)
    cand = np.array([i for i in range(N) if i not in have]); cost = effort(have)
    base = flags(have) if profile is None else profile
    def fitness(P):
        M = np.repeat(base[None], len(P), 0); np.put_along_axis(M, P, 1, 1)
        return probs(M)[:, target] - lam * cost[P].mean(1)
    P = np.array([rng.choice(cand, k, replace=False) for _ in range(pop)]); hist, bests = [], []
    for _ in range(gens):
        f = fitness(P); order = np.argsort(-f); hist.append(float(f[order[0]])); bests.append(P[order[0]].tolist())
        nxt = [P[order[0]].copy(), P[order[1]].copy()]                        # elitism: keep the best two
        while len(nxt) < pop:
            a, b = (P[max(rng.choice(pop, 3), key=lambda i: f[i])] for _ in range(2))   # tournament selection
            child = rng.choice(np.union1d(a, b), k, replace=False)            # crossover: mix both parents' skills
            if rng.random() < 0.3:                                            # mutation: swap in a random skill
                c = rng.choice(cand)
                if c not in child: child[rng.integers(k)] = c
            nxt.append(child)
        P = np.array(nxt)
    return P[np.argmax(fitness(P))].tolist(), hist, bests

def path(have, new):
    """Order the new skills: always learn next the one closest to what you know so far."""
    known, todo, out = list(have), list(new), []
    while todo:
        nxt = min(todo, key=lambda i: effort(known)[i]); todo.remove(nxt); known.append(nxt); out.append(nxt)
    return out

def test_set():
    """The 20% of postings the model never saw during training (exported from Kaggle)."""
    X = np.zeros((len(D['test']['ix']), N), dtype=np.float32)
    for r, ix in enumerate(D['test']['ix']): X[r, ix] = 1
    return X, np.array(D['test']['y'])

def audit(seed=0):
    """Recompute accuracy right now from the stored weights. Also a control: scramble which column is which skill."""
    X, y = test_set(); rng = np.random.default_rng(seed)
    return dict(n=len(y), accuracy=float((probs(X).argmax(1) == y).mean()),
                scrambled=float((probs(X[:, rng.permutation(N)]).argmax(1) == y).mean()), chance=1 / len(ROLES))
