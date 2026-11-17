# CareerTwin coder models (Kaggle, CPU + internet).
# 1. Collect real data from the official Codeforces API (rate limit: 1 call per 2 seconds).
# 2. Regression: predict a coder's rating from what they have solved (Module I: MSE, MAE, R2; Module III: regularisation).
# 3. K-means: group coders into styles by which topics they solve (Module IV).
# 4. Rating-band profiles: what coders one level above you have solved (the "success profile").
import requests, time, json, random, numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
RNG = 42; random.seed(RNG); np.random.seed(RNG)
S = requests.Session(); S.headers['User-Agent'] = 'CareerTwin student project'
last = [0.0]
def api(method, **params):
    for attempt in range(5):
        wait = 2.1 - (time.time() - last[0])
        if wait > 0: time.sleep(wait)
        last[0] = time.time()
        try:
            j = S.get(f'https://codeforces.com/api/{method}', params=params, timeout=60).json()
            if j.get('status') == 'OK': return j['result']
            if 'not found' in j.get('comment', ''): return None
        except Exception as e:
            print('retry', method, e)
        time.sleep(3 * (attempt + 1))
    return None

# ---------- 1. collect ----------
problems = api('problemset.problems')
P = pd.DataFrame(problems['problems']).merge(pd.DataFrame(problems['problemStatistics']), on=['contestId', 'index'], how='left')
print('problems:', len(P))
users = pd.DataFrame(api('user.ratedList', activeOnly='true', includeRetired='false'))
print('active rated users:', len(users))
ratings_all = users.rating.values
users = users[users.rating >= 800].copy()
users['band'] = (users.rating.clip(800, 3199) // 200 * 200).astype(int)
sample = pd.concat([g.sample(min(len(g), 150), random_state=RNG) for _, g in users.groupby('band')]).sample(frac=1, random_state=RNG)
print('sampling', len(sample), 'users across', sample.band.nunique(), 'bands', flush=True)
rows, t0 = [], time.time()
for k, u in enumerate(sample.itertuples()):
    if time.time() - t0 > 85 * 60: print('time budget reached'); break
    subs = api('user.status', handle=u.handle)
    if not subs: continue
    solved, tried = {}, set()
    for s in subs:
        p = s.get('problem', {}); key = (p.get('contestId'), p.get('index'))
        if key[0] is None: continue
        tried.add(key)
        if s.get('verdict') == 'OK' and key not in solved: solved[key] = (p.get('rating'), p.get('tags', []))
    rows.append(dict(handle=u.handle, rating=u.rating, maxRating=u.maxRating, n_subs=len(subs), n_tried=len(tried),
                     solved=[[r, t] for r, t in solved.values()]))
    if k % 100 == 0: print(k, 'users', round((time.time() - t0) / 60, 1), 'min', flush=True)
raw = pd.DataFrame(rows); raw.to_json('cf_raw.json.gz', orient='records')
print('collected', len(raw))

# ---------- 2. features (the app recomputes these exactly) ----------
TAGS = [t for t, c in Counter(t for s in raw.solved for _, ts in s for t in ts).most_common(24)]
EDGES = list(range(800, 3001, 200))       # difficulty buckets 800-999, 1000-1199, ... 3000+
def feats(solved, n_tried):
    r = np.array([x[0] for x in solved if x[0]], dtype=float)
    hist = np.histogram(r, bins=EDGES + [10000])[0] if len(r) else np.zeros(len(EDGES))
    top = np.sort(r)[-20:] if len(r) else np.array([0.0])
    tc = Counter(t for _, ts in solved for t in ts); n = len(solved)
    return np.concatenate([np.log1p(hist), [np.log1p(n), np.log1p(n_tried), n / max(n_tried, 1), top.mean() / 1000,
                                           (r.max() if len(r) else 0) / 1000, np.median(r) / 1000 if len(r) else 0],
                           [np.log1p(tc[t]) for t in TAGS]])
FEATS = [f'solved {e}-{e + 199}' if e < 3000 else 'solved 3000+' for e in EDGES] + ['log solved', 'log tried', 'solve ratio',
         'mean of top-20 solved rating', 'hardest solved', 'median solved rating'] + [f'tag: {t}' for t in TAGS]
data = raw[raw.solved.map(len) >= 5].reset_index(drop=True)
X = np.vstack([feats(s, t) for s, t in zip(data.solved, data.n_tried)]); y = data.rating.values.astype(float)
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, np.arange(len(y)), test_size=0.2, random_state=RNG)
print('model data', X.shape, flush=True)

# ---------- 3. regression: compare models ----------
MODELS = {
    'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
    'Ridge (L2)': make_pipeline(StandardScaler(), Ridge(alpha=3)),
    'Lasso (L1)': make_pipeline(StandardScaler(), Lasso(alpha=3, max_iter=20000)),
    'Elastic Net': make_pipeline(StandardScaler(), ElasticNet(alpha=3, l1_ratio=.5, max_iter=20000)),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=RNG),
    'Random Forest': RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=RNG),
    'Gradient Boosting': GradientBoostingRegressor(random_state=RNG),
    'KNN': make_pipeline(StandardScaler(), KNeighborsRegressor(15)),
    'Neural Network (MLP)': make_pipeline(StandardScaler(), MLPRegressor((64, 32), alpha=1e-3, max_iter=3000, early_stopping=True, random_state=RNG)),
}
res = []
for name, m in MODELS.items():
    m.fit(Xtr, ytr); p = m.predict(Xte); pt = m.predict(Xtr)
    cv = cross_val_score(m, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=RNG), scoring='neg_mean_absolute_error')
    res.append(dict(model=name, mae=mean_absolute_error(yte, p), rmse=mean_squared_error(yte, p) ** .5, r2=r2_score(yte, p),
                    train_mae=mean_absolute_error(ytr, pt), cv_mae=-cv.mean())); print(res[-1], flush=True)
base_mae = mean_absolute_error(yte, np.full(len(yte), ytr.mean()))
rid = MODELS['Ridge (L2)']; sc, lr = rid[0], rid[1]
mp = MODELS['Neural Network (MLP)']; msc, mlp = mp[0], mp[1]
best = min(('Ridge (L2)', 'Neural Network (MLP)'), key=lambda n: next(r['mae'] for r in res if r['model'] == n))
coef_order = np.argsort(-np.abs(lr.coef_)); gb = MODELS['Gradient Boosting']; pred_te = MODELS[best].predict(Xte)

# ---------- 4. K-means coder styles ----------
tagc = np.vstack([[sum(1 for _, ts in s if t in ts) for t in TAGS] for s in data.solved]).astype(float)
ok = data.solved.map(len).values >= 30
share = tagc[ok] / tagc[ok].sum(1, keepdims=True); mu, sd = share.mean(0), share.std(0) + 1e-9; Z = (share - mu) / sd
ks, sil, inertia = list(range(3, 9)), [], []
for k in ks:
    km = KMeans(k, n_init=20, random_state=RNG).fit(Z); sil.append(float(silhouette_score(Z, km.labels_))); inertia.append(float(km.inertia_))
bk = ks[int(np.argmax(sil))]; km = KMeans(bk, n_init=20, random_state=RNG).fit(Z)
styles = []
for c in range(bk):
    lift = share[km.labels_ == c].mean(0) / share.mean(0)
    styles.append(dict(top=[TAGS[i] for i in np.argsort(-lift)[:3]], low=[TAGS[i] for i in np.argsort(lift)[:2]], size=int((km.labels_ == c).sum()),
                       median_rating=float(np.median(data.rating.values[ok][km.labels_ == c]))))
print('styles', bk, styles, flush=True)

# ---------- 5. band profiles ----------
data['band'] = (data.rating.clip(800, 3199) // 200 * 200).astype(int)
bands = {}
for b, g in data.groupby('band'):
    idx = g.index.values
    bands[int(b)] = dict(n=len(g), tags=np.median(tagc[idx], 0).tolist(), solved=float(g.solved.map(len).median()),
                         diff=np.median(np.vstack([np.histogram([x[0] for x in s if x[0]], bins=EDGES + [10000])[0] for s in g.solved]), 0).tolist())

PP = P.dropna(subset=['rating'])[['contestId', 'index', 'name', 'rating', 'tags', 'solvedCount']].fillna(0)
OUT = dict(tags=TAGS, edges=EDGES, feats=FEATS, n_users=int(len(data)), n_active=int(len(ratings_all)),
           rating_quantiles=np.percentile(ratings_all, np.arange(101)).tolist(),
           models=res, baseline_mae=base_mae, deployed=best,
           ridge=dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_.tolist(), intercept=float(lr.intercept_)),
           mlp=dict(mean=msc.mean_.tolist(), scale=msc.scale_.tolist(), W=[w.tolist() for w in mlp.coefs_], b=[b.tolist() for b in mlp.intercepts_]),
           ridge_top=[dict(feature=FEATS[i], coef=float(lr.coef_[i])) for i in coef_order[:12]],
           gb_importance=sorted([dict(feature=f, imp=float(v)) for f, v in zip(FEATS, gb.feature_importances_)], key=lambda d: -d['imp'])[:12],
           scatter=dict(actual=yte.tolist(), pred=pred_te.tolist()),
           kmeans=dict(k=ks, silhouette=sil, inertia=inertia, best_k=bk, mu=mu.tolist(), sd=sd.tolist(), centers=km.cluster_centers_.tolist(), styles=styles),
           bands=bands,
           check=[dict(solved=data.solved[i], n_tried=int(data.n_tried[i]), pred=float(MODELS[best].predict(X[i:i + 1])[0])) for i in ite[:25]],
           problems=dict(id=[f'{int(a)}{b}' for a, b in zip(PP.contestId, PP['index'])], name=PP.name.tolist(), rating=PP.rating.astype(int).tolist(),
                         tags=PP.tags.tolist(), solved=PP.solvedCount.astype(int).tolist()))
json.dump(OUT, open('coders.json', 'w'), default=float)
print('done. deployed:', best, 'baseline MAE', base_mae)
