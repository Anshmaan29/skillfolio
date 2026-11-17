# CareerTwin knowledge-tracing models (Kaggle GPU). Real student answers:
#   Riiid! (100M answers to 4-option multiple-choice questions) and ASSISTments 2009 (named math skills).
# Task: predict whether a student answers the next question correctly, from their earlier answers.
# Models: item average (baseline), IRT 2PL, IRT 4PL (guess + slip), IRT with only 3 difficulty levels (= how our app works),
#         Bayesian Knowledge Tracing (hidden Markov model), Deep Knowledge Tracing (LSTM).
# Also: adaptive-test simulation, forgetting curve, learning curve. Output: kt.json for the app.
import glob, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from scipy.optimize import curve_fit
dev = 'cuda'; RNG = 42; torch.manual_seed(RNG); rng = np.random.default_rng(RNG)
MAXLEN = 100                                   # first 100 answers of each student
find = lambda n: glob.glob(f'/kaggle/input/**/{n}', recursive=True)
def log(*a): print(*a, flush=True); open('progress.txt', 'a').write(' '.join(map(str, a)) + '\n')
OUT = {}
def save(): json.dump(OUT, open('kt.json', 'w'), default=float)
def score(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return dict(auc=roc_auc_score(y, p), acc=accuracy_score(y, p > .5), logloss=log_loss(y, p))

def grid(d, U, col, dtype):
    g = np.zeros((U, MAXLEN), dtype=dtype); g[d.uix.values, d.pos.values] = d[col].values; return g

# ---------------- IRT ----------------
def irt_prob(theta, a, b, g, s): return g + (1 - g - s) * torch.sigmoid(a * (theta - b))
def fit_irt(d, n_items, four):
    u = torch.tensor(d.uix.values, device=dev); i = torch.tensor(d.item.values, device=dev); y = torch.tensor(d.y.values, device=dev, dtype=torch.float32)
    th = torch.zeros(d.uix.max() + 1, device=dev, requires_grad=True); b = torch.zeros(n_items, device=dev, requires_grad=True)
    la = torch.zeros(n_items, device=dev, requires_grad=True); gs = torch.tensor([-1.0, -3.0], device=dev, requires_grad=four)
    opt = torch.optim.Adam([th, b, la] + ([gs] if four else []), lr=0.05); n = len(y)
    for step in range(400):
        g, s = (0.5 * torch.sigmoid(gs[0]), 0.3 * torch.sigmoid(gs[1])) if four else (torch.tensor(0., device=dev),) * 2
        p = irt_prob(th[u], la.exp()[i], b[i], g, s).clamp(1e-5, 1 - 1e-5)
        loss = nn.functional.binary_cross_entropy(p, y) + (0.5 * (th ** 2).sum() + 0.125 * (b ** 2).sum() + 2 * (la ** 2).sum()) / n   # MAP priors
        opt.zero_grad(); loss.backward(); opt.step()
    g, s = (0.5 * torch.sigmoid(gs[0]), 0.3 * torch.sigmoid(gs[1])) if four else (torch.tensor(0., device=dev),) * 2
    return la.exp().detach(), b.detach(), float(g), float(s)

TH = torch.linspace(-4, 4, 81, device=dev)
@torch.no_grad()
def irt_online(I, Y, M, a, b, g, s):
    """The app's method: start from a normal prior over ability, predict each answer, then update with Bayes' theorem."""
    lp = (-0.5 * TH ** 2).repeat(I.shape[0], 1); out = []
    for t in range(I.shape[1]):
        P = irt_prob(TH[None], a[I[:, t], None], b[I[:, t], None], g, s).clamp(1e-5, 1 - 1e-5)
        post = torch.softmax(lp, 1); out.append((post * P).sum(1))
        yt = Y[:, t, None].float(); lp = lp + M[:, t, None] * (yt * P.log() + (1 - yt) * (1 - P).log())
    return torch.stack(out, 1)

# ---------------- BKT ----------------
def bkt_params(p): return torch.sigmoid(p[:, 0]), torch.sigmoid(p[:, 1]), 0.4 * torch.sigmoid(p[:, 2]), 0.4 * torch.sigmoid(p[:, 3])  # L0, T, G, S
def bkt_forward(par, sk, Y, M):
    L0, T, G, S = bkt_params(par); L0, T, G, S = L0[sk], T[sk], G[sk], S[sk]; pL = L0; out = []
    for t in range(Y.shape[1]):
        pc = (pL * (1 - S) + (1 - pL) * G).clamp(1e-5, 1 - 1e-5); out.append(pc); y = Y[:, t]
        post = torch.where(y > 0, pL * (1 - S) / pc, pL * S / (1 - pc))
        pL = torch.where(M[:, t], post + (1 - post) * T, pL)
    return torch.stack(out, 1)
def skill_seqs(d):
    d = d.copy(); d['k'] = d.groupby(['u', 'skill']).cumcount(); d['sid'] = d.groupby(['u', 'skill']).ngroup()
    n, L = d.sid.max() + 1, d.k.max() + 1
    Y = np.zeros((n, L), np.float32); M = np.zeros((n, L), bool); sk = np.zeros(n, np.int64)
    Y[d.sid, d.k] = d.y; M[d.sid, d.k] = True; sk[d.sid] = d.skill
    return d, torch.tensor(Y, device=dev), torch.tensor(M, device=dev), torch.tensor(sk, device=dev)
def fit_bkt(d, n_skills):
    _, Y, M, sk = skill_seqs(d)
    if len(sk) > 300000:
        keep = torch.randperm(len(sk), device=dev)[:300000]; Y, M, sk = Y[keep], M[keep], sk[keep]
    Y, M = Y[:, :50], M[:, :50]
    par = torch.tensor([[-1.0, -2.0, -1.0, -1.5]], device=dev).repeat(n_skills, 1).requires_grad_(True)
    opt = torch.optim.Adam([par], lr=0.05)
    for ep in range(150):
        p = bkt_forward(par, sk, Y, M); loss = nn.functional.binary_cross_entropy(p[M], Y[M])
        opt.zero_grad(); loss.backward(); opt.step()
    return par.detach()

# ---------------- DKT (LSTM) ----------------
class DKT(nn.Module):
    def __init__(s, n_items, d=200):
        super().__init__(); s.inp = nn.Embedding(2 * n_items + 1, d); s.lstm = nn.LSTM(d, d, batch_first=True)
        s.out = nn.Embedding(n_items, d); s.bias = nn.Embedding(n_items, 1); s.drop = nn.Dropout(0.2)
    def forward(s, I, Y):
        tok = I * 2 + Y + 1; tok = torch.cat([torch.zeros_like(tok[:, :1]), tok[:, :-1]], 1)   # 0 = start of sequence
        h, _ = s.lstm(s.drop(s.inp(tok)))
        return (s.drop(h) * s.out(I)).sum(-1) + s.bias(I).squeeze(-1)

# ---------------- one full benchmark on one dataset ----------------
def benchmark(df, name, n_items, n_skills, skill_names=None):
    df = df.astype({'item': 'int64', 'skill': 'int64', 'y': 'int64', 'pos': 'int64'})
    users = df.u.unique(); rng.shuffle(users); cut = int(.8 * len(users))
    dtr = df[df.u.isin(users[:cut])].copy(); dte = df[df.u.isin(users[cut:])].copy()
    for d in (dtr, dte): d['uix'] = pd.factorize(d.u)[0]
    Ute = dte.uix.max() + 1
    T = lambda d, U, c, t: torch.tensor(grid(d, U, c, t), device=dev)
    dte['one'] = True; Ite, Yte, Mte = T(dte, Ute, 'item', np.int64), T(dte, Ute, 'y', np.int64), T(dte, Ute, 'one', bool)
    y = Yte[Mte].cpu().numpy(); R, P = [], {}
    log(name, 'train users', cut, 'test users', Ute, 'test answers', len(y))
    # 1. baseline
    st = dtr.groupby('item').y.agg(['sum', 'count']); gm = dtr.y.mean(); pi = np.full(n_items, gm, np.float32)
    pi[st.index] = (st['sum'] + 5 * gm) / (st['count'] + 5)
    P['Item average (baseline)'] = torch.tensor(pi, device=dev)[Ite]
    # 2-4. IRT
    for four in (False, True):
        t0 = time.time(); a, b, g, s = fit_irt(dtr, n_items, four)
        nm = 'IRT 4PL (guess + slip)' if four else 'IRT 2PL'
        P[nm] = irt_online(Ite, Yte, Mte, a, b, g, s); log(nm, 'fit', round(time.time() - t0), 's, guess', g, 'slip', s)
    seen = torch.zeros(n_items, dtype=torch.bool, device=dev); seen[torch.tensor(st.index[st['count'] >= 20].values, device=dev)] = True
    q = torch.quantile(b[seen], torch.tensor([1 / 3, 2 / 3], device=dev)); lvl = torch.bucketize(b, q)
    b3 = torch.stack([b[seen & (lvl == k)].median() for k in range(3)]); a3 = a[seen].median()
    P['IRT, only 3 difficulty levels (our app)'] = irt_online(Ite, Yte, Mte, torch.full_like(a, a3), b3[lvl], g, s)
    irt = dict(guess=g, slip=s, a_median=float(a3), b_levels=b3.tolist(), b_quantiles=torch.quantile(b[seen], torch.linspace(0, 1, 21, device=dev)).tolist())
    # 5. BKT
    t0 = time.time(); par = fit_bkt(dtr, n_skills)
    d2, Y2, M2, sk2 = skill_seqs(dte); p2 = bkt_forward(par, sk2, Y2, M2)
    pb = torch.zeros_like(Yte, dtype=torch.float32); pb[torch.tensor(d2.uix.values, device=dev), torch.tensor(d2.pos.values, device=dev)] = p2[torch.tensor(d2.sid.values, device=dev), torch.tensor(d2.k.values, device=dev)]
    P['Bayesian Knowledge Tracing'] = pb; log('BKT', round(time.time() - t0), 's')
    L0, Tt, G, S = [x.cpu().numpy() for x in bkt_params(par)]; cnt = dtr.skill.value_counts()
    used = cnt.index[cnt >= 200].values
    bkt = dict(median=dict(L0=float(np.median(L0[used])), T=float(np.median(Tt[used])), G=float(np.median(G[used])), S=float(np.median(S[used]))),
               skills=[dict(skill=skill_names[k] if skill_names else int(k), n=int(cnt[k]), L0=float(L0[k]), T=float(Tt[k]), G=float(G[k]), S=float(S[k])) for k in cnt.index[:12]])
    # 6. DKT
    Utr = dtr.uix.max() + 1; dtr['one'] = True
    Itr, Ytr, Mtr = T(dtr, Utr, 'item', np.int64), T(dtr, Utr, 'y', np.int64), T(dtr, Utr, 'one', bool)
    m = DKT(n_items).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3); curve = []; t0 = time.time()
    for ep in range(10):
        m.train(); perm = torch.randperm(Utr, device=dev); tot = 0
        for k in range(0, Utr, 256):
            ix = perm[k:k + 256]; z = m(Itr[ix], Ytr[ix])
            loss = nn.functional.binary_cross_entropy_with_logits(z[Mtr[ix]], Ytr[ix][Mtr[ix]].float())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step(); tot += loss.item()
        m.eval()
        with torch.no_grad(): pd_ = torch.cat([torch.sigmoid(m(Ite[k:k + 512], Yte[k:k + 512])) for k in range(0, Ute, 512)])
        curve.append(dict(epoch=ep + 1, train_loss=tot / (Utr / 256), test_auc=roc_auc_score(y, pd_[Mte].cpu().numpy()))); log('DKT', curve[-1])
    P['Deep Knowledge Tracing (LSTM)'] = pd_
    nparams = sum(p.numel() for p in m.parameters())
    for nm, p in P.items():
        r = score(p[Mte].cpu().numpy(), y); r['model'] = nm; R.append(r); log(name, r)
    pi4 = P['IRT 4PL (guess + slip)'][Mte].cpu().numpy(); bins = np.minimum((pi4 * 10).astype(int), 9)
    calib = [dict(conf=float(pi4[bins == k].mean()), acc=float(y[bins == k].mean()), n=int((bins == k).sum())) for k in range(10) if (bins == k).any()]
    # accuracy by how many answers the model has already seen from this student (cold start -> warm)
    steps = []
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100)]:
        mm = Mte.clone(); mm[:, :lo] = False; mm[:, hi:] = False; yy = Yte[mm].cpu().numpy()
        steps.append(dict(window=f'{lo + 1}-{hi}', **{nm: roc_auc_score(yy, p[mm].cpu().numpy()) for nm, p in P.items()}))
    return dict(n_users=int(len(users)), n_answers=int(len(df)), n_test_answers=int(len(y)), n_items=int(n_items), n_skills=int(n_skills),
                results=R, irt=irt, bkt=bkt, dkt=dict(curve=curve, params=int(nparams), seconds=time.time() - t0), calibration=calib, by_step=steps), (a, b, g, s)

# ================= Riiid =================
t0 = time.time()
tr = pd.read_parquet(find('train.parquet')[0], columns=['user_id', 'timestamp', 'content_id', 'content_type_id', 'answered_correctly'])
tr = tr[tr.content_type_id == 0].drop(columns='content_type_id'); total_answers, total_users = len(tr), tr.user_id.nunique()
qs = pd.read_parquet(find('questions.parquet')[0]); log('riiid loaded', total_answers, 'answers', total_users, 'users', round(time.time() - t0), 's')
skill_of = pd.to_numeric(qs.set_index('question_id').tags.astype(str).str.split().str[0], errors='coerce').fillna(-1).astype(int)
allu = tr.user_id.unique(); pick = rng.choice(allu, 60000, replace=False)
full = tr[tr.user_id.isin(pick)].sort_values(['user_id', 'timestamp'], kind='stable').reset_index(drop=True); del tr
full['skill'] = full.content_id.map(skill_of).fillna(-1).astype(int); full['skill'] = pd.factorize(full.skill)[0]
full = full.rename(columns={'user_id': 'u', 'content_id': 'item', 'answered_correctly': 'y', 'timestamp': 'ts'})
full['pos'] = full.groupby('u').cumcount()

# learning curve + forgetting curve (descriptive, full histories of the sampled students)
full['k'] = full.groupby(['u', 'skill']).cumcount()
lc = full[full.k < 30].groupby('k').y.mean()
prev = full.groupby(['u', 'skill']).shift(1)
pairs = full[prev.y.notna()].assign(prev_y=prev.y, gap_h=(full.ts - prev.ts) / 3.6e6)
pairs = pairs[pairs.prev_y == 1]
edges = [0, 0.1, 1, 6, 24, 72, 168, 720, 1e9]; labels = ['<6 min', '6 min-1 h', '1-6 h', '6-24 h', '1-3 days', '3-7 days', '1-4 weeks', '>4 weeks']
pairs['bucket'] = pd.cut(pairs.gap_h, edges, labels=labels, right=False)
fc = pairs.groupby('bucket', observed=False).agg(acc=('y', 'mean'), n=('y', 'size'), gap=('gap_h', 'median')).reset_index()
try:
    f = lambda g, pinf, p0, tau: pinf + (p0 - pinf) * np.exp(-g / tau)
    (pinf, p0, tau), _ = curve_fit(f, fc.gap.values / 24, fc.acc.values, p0=[.6, .8, 3], maxfev=20000)
except Exception as e:
    log('forgetting fit failed', e); pinf, p0, tau = float(fc.acc.min()), float(fc.acc.max()), 7.0
OUT['curves'] = dict(learning=[dict(attempt=int(k + 1), acc=float(v)) for k, v in lc.items()],
                     forgetting=[dict(gap=str(r.bucket), hours=float(r.gap), acc=float(r.acc), n=int(r.n)) for r in fc.itertuples()],
                     forgetting_fit=dict(p_inf=float(pinf), p0=float(p0), tau_days=float(tau)))
log('curves', OUT['curves']['forgetting_fit'])

df = full[full.pos < MAXLEN][['u', 'item', 'skill', 'y', 'pos']].copy()
df = df.astype({'item': 'int64', 'skill': 'int64', 'y': 'int64'})
riiid, (a, b, g, s) = benchmark(df, 'riiid', 13523, int(df.skill.max() + 1))
riiid.update(total_answers=int(total_answers), total_users=int(total_users), sampled_users=60000)
OUT['riiid'] = riiid; save()

# ================= adaptive testing simulation (uses the item parameters fitted on Riiid) =================
@torch.no_grad()
def simulate(A, B, g, s, n=4000, max_q=20, adaptive=True, sd_stop=None, min_q=5):
    th_true = torch.randn(n, device=dev); lp = (-0.5 * TH ** 2).repeat(n, 1); used = torch.zeros(n, len(A), dtype=torch.bool, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev); nq = torch.zeros(n, device=dev); err = []
    for q in range(max_q):
        post = torch.softmax(lp, 1); m = (post * TH).sum(1)
        if adaptive:
            P = irt_prob(m[:, None], A[None], B[None], g, s)
            info = A ** 2 * (P - g) ** 2 * (1 - s - P) ** 2 / ((1 - g - s) ** 2 * P * (1 - P))
            pick = info.masked_fill(used, -1).argmax(1)
        else:
            pick = torch.rand(n, len(A), device=dev).masked_fill(used, -1).argmax(1)
        used[torch.arange(n), pick] = True
        resp = (torch.rand(n, device=dev) < irt_prob(th_true, A[pick], B[pick], g, s)).float()[:, None]
        Pg = irt_prob(TH[None], A[pick, None], B[pick, None], g, s)
        act = (~done).float()[:, None]; lp = lp + act * (resp * Pg.log() + (1 - resp) * (1 - Pg).log()); nq += (~done).float()
        post = torch.softmax(lp, 1); m = (post * TH).sum(1); sd = ((post * (TH - m[:, None]) ** 2).sum(1)).sqrt()
        err.append(float(((m - th_true) ** 2).mean().sqrt()))
        if sd_stop is not None: done |= (nq >= min_q) & (sd < sd_stop)
    return err, float(nq.mean()), float(((m - th_true) ** 2).mean().sqrt())
cnt = df.groupby('item').size(); ok = torch.tensor(cnt.index[cnt >= 50].values, device=dev)
ok = ok[torch.randperm(len(ok), device=dev)[:1500]]
A_real, B_real = a[ok], b[ok]
ir = riiid['irt']; b3 = torch.tensor(ir['b_levels'], device=dev)
A_app = torch.full((18,), ir['a_median'], device=dev); B_app = b3.repeat_interleave(6)
sim = dict(real_random=simulate(A_real, B_real, g, s, adaptive=False)[0], real_adaptive=simulate(A_real, B_real, g, s)[0],
           app_random=simulate(A_app, B_app, g, s, max_q=10, adaptive=False)[0], app_adaptive=simulate(A_app, B_app, g, s, max_q=10)[0])
stop = []
for sd_ in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
    _, avg_n, rmse = simulate(A_app, B_app, g, s, max_q=10, sd_stop=sd_, min_q=5); stop.append(dict(sd_stop=sd_, avg_questions=avg_n, rmse=rmse))
OUT['cat'] = dict(curves=sim, stopping=stop); log('cat', stop); save()

# ================= ASSISTments 2009 =================
try:
    A = pd.read_csv(find('assistments_2009_2010.csv')[0], encoding='latin-1', low_memory=False)
    c = {k.lower(): k for k in A.columns}; log('assistments columns', list(A.columns)[:30])
    sk = c.get('skill_name') or c.get('skill_id')
    A = A.dropna(subset=[sk])
    if 'order_id' in c: A = A.drop_duplicates(c['order_id']).sort_values([c['user_id'], c['order_id']])
    A = A[A[c['correct']].isin([0, 1])]
    d = pd.DataFrame(dict(u=A[c['user_id']].values, item=pd.factorize(A[c['problem_id']])[0], y=A[c['correct']].astype(int).values))
    names, codes = pd.factorize(A[sk])[1], pd.factorize(A[sk])[0]; d['skill'] = codes
    d['pos'] = d.groupby('u').cumcount(); d = d[d.pos < MAXLEN]
    d = d[d.u.map(d.u.value_counts()) >= 5]
    assist, _ = benchmark(d, 'assistments', int(d.item.max() + 1), int(d.skill.max() + 1), [str(x) for x in names])
    OUT['assistments'] = assist
except Exception as e:
    log('assistments failed', type(e).__name__, e)
save(); log('done')
