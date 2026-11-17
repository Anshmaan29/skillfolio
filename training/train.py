# CareerTwin training script (runs on Kaggle). Real data: 1.3M LinkedIn job postings (2024).
# Output: careertwin.json (everything the app needs) + mlflow_runs.csv
import pandas as pd, numpy as np, json, re, glob, time, warnings
from collections import Counter
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, top_k_accuracy_score)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')
RNG = 42
find = lambda n: glob.glob(f'/kaggle/input/**/{n}', recursive=True)[0]

# ---------- 1. Load postings and label each one with a role from its job title ----------
ROLES = [
    ('ML Engineer', r'machine learning|\bml engineer|\bai engineer|deep learning|\bnlp\b|computer vision'),
    ('Data Scientist', r'data scien'),
    ('Data Engineer', r'data engineer|\betl\b|big data'),
    ('Data Analyst', r'data analyst|business intelligence|\bbi (?:analyst|developer)'),
    ('DevOps / Cloud', r'devops|site reliability|\bsre\b|cloud engineer|cloud architect|platform engineer'),
    ('Cybersecurity', r'cyber|information security|security engineer|security analyst|penetration|infosec'),
    ('Full Stack Developer', r'full[- ]?stack'),
    ('Frontend Developer', r'front[- ]?end|ui developer|\breact\b|angular'),
    ('Mobile Developer', r'android|\bios\b|mobile (?:developer|engineer|app)|flutter'),
    ('QA / Testing', r'qa engineer|qa analyst|test engineer|sdet|software tester|qa automation|test automation|quality assurance engineer'),
    ('UI/UX Designer', r'\bux\b|user experience|ui/ux|product designer'),
    ('Embedded Engineer', r'embedded|firmware'),
    ('Software Engineer', r'software engineer|software developer|back[- ]?end|java developer|python developer|\.net developer|node\.?js'),
]
posts = pd.read_csv(find('linkedin_job_postings.csv'), usecols=['job_link', 'job_title', 'company'])
t = posts.job_title.fillna('').str.lower()
posts['role'] = None
for name, rx in ROLES:
    posts.loc[posts.role.isna() & t.str.contains(rx, regex=True), 'role'] = name
posts = posts.dropna(subset=['role'])
df = posts.merge(pd.read_csv(find('job_skills.csv')), on='job_link').dropna(subset=['job_skills'])
df = df.drop_duplicates(['job_title', 'company', 'job_skills'])   # no copies leaking into the test set
print('labelled tech postings:', len(df))

# ---------- 2. Clean the skill lists ----------
ALIAS = {'amazon web services': 'aws', 'microsoft excel': 'excel', 'ms excel': 'excel', 'js': 'javascript',
         'reactjs': 'react', 'react.js': 'react', 'nodejs': 'node.js', 'node': 'node.js', 'ml': 'machine learning',
         'k8s': 'kubernetes', 'postgres': 'postgresql', 'problemsolving': 'problem solving', 'golang': 'go',
         'ci/cd pipelines': 'ci/cd', 'cicd': 'ci/cd', 'microsoft azure': 'azure', 'google cloud platform': 'gcp',
         'google cloud': 'gcp', 'rest apis': 'rest api', 'restful apis': 'rest api', 'rest': 'rest api',
         'artificial intelligence': 'ai', 'natural language processing': 'nlp', 'team work': 'teamwork',
         'html5': 'html', 'css3': 'css', 'angularjs': 'angular', 'vue.js': 'vue', 'object-oriented': 'oop',
         'objectoriented': 'oop', 'object oriented': 'oop', 'agile methodologies': 'agile', 'agile development': 'agile'}
BAD = re.compile(r"degree|experience|years|bachelor|master'?s|diploma|certification|ability to|clearance|licen[sc]e|travel|citizen|salary|benefit|insurance|401|remote|hybrid|full ?time")
def norm(s):
    out = []
    for x in s.lower().split(','):
        x = re.sub(r'\s+', ' ', x.strip(' .'))
        x = re.sub(r' (skills|programming|language)$', '', x)
        x = ALIAS.get(x, x)
        if 1 < len(x) <= 30 and not BAD.search(x) and x not in out: out.append(x)
    return out
df['sk'] = df.job_skills.map(norm)
df = pd.concat([g.sample(min(len(g), 6000), random_state=RNG) for _, g in df.groupby('role')])
cnt = Counter(s for l in df.sk for s in l)
vocab = [s for s, c in cnt.most_common(500) if c >= 40]
ix = {s: i for i, s in enumerate(vocab)}
df['ix'] = df.sk.map(lambda l: [ix[s] for s in l if s in ix])
df = df[df.ix.map(len) >= 3].reset_index(drop=True)
roles = sorted(df.role.unique())
y = df.role.map({r: i for i, r in enumerate(roles)}).values
rows = np.repeat(np.arange(len(df)), df.ix.map(len)); cols = np.concatenate(df.ix.values)
X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(df), len(vocab)))
print('dataset:', X.shape, dict(Counter(df.role)))

# ---------- 3. Skill embeddings: co-occurrence -> PPMI -> SVD (Module IV/V) ----------
C = (X.T @ X).toarray(); np.fill_diagonal(C, 0)
tot = C.sum(); rs = C.sum(1, keepdims=True)
ppmi = np.maximum(np.log((C * tot + 1e-9) / (rs @ rs.T + 1e-9)), 0)
svd = TruncatedSVD(50, random_state=RNG)
emb = svd.fit_transform(ppmi); emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
xy = TSNE(2, perplexity=30, metric='cosine', init='random', random_state=RNG).fit_transform(emb)
xy = (xy - xy.mean(0)) / xy.std(0)
xyz = TSNE(3, perplexity=30, metric='cosine', init='random', random_state=RNG).fit_transform(emb)
xyz = (xyz - xyz.mean(0)) / xyz.std(0)

# ---------- 4. K-means on skills, pick k by silhouette (Module IV) ----------
ks, inertia, sil = list(range(4, 15)), [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=RNG).fit(emb)
    inertia.append(float(km.inertia_)); sil.append(float(silhouette_score(emb, km.labels_)))
best_k = ks[int(np.argmax(sil))]
clusters = KMeans(best_k, n_init=10, random_state=RNG).fit_predict(emb)
freq = np.asarray(X.sum(0)).ravel()
cl_names = [' · '.join(vocab[i] for i in np.argsort(-freq * (clusters == c))[:3]) for c in range(best_k)]

# ---------- 5. Compare classifiers: skills -> role (Modules I, II, III) ----------
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=RNG)
MODELS = {
    'Naive Bayes': BernoulliNB(),
    'Logistic Regression (softmax)': LogisticRegression(C=1, max_iter=400),
    'Decision Tree (CART, Gini)': DecisionTreeClassifier(max_depth=30, random_state=RNG),
    'Random Forest (bagging)': RandomForestClassifier(200, n_jobs=-1, random_state=RNG),
    'Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, random_state=RNG),
    'Linear SVM': LinearSVC(C=0.1),
    'K-Nearest Neighbours': KNeighborsClassifier(15, metric='cosine'),
    'Neural Network (MLP)': MLPClassifier((128,), early_stopping=True, max_iter=60, random_state=RNG),
}
results = []
for name, m in MODELS.items():
    d = 'Boost' in name
    A, B = (Xtr.toarray(), Xte.toarray()) if d else (Xtr, Xte)
    t0 = time.time(); m.fit(A, ytr); ft = time.time() - t0
    p = m.predict(B); pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average='macro', zero_division=0)
    auc = roc_auc_score(yte, m.predict_proba(B), multi_class='ovr') if hasattr(m, 'predict_proba') else None
    r = dict(model=name, accuracy=accuracy_score(yte, p), precision=pr, recall=rc, f1=f1, roc_auc=auc,
             train_accuracy=accuracy_score(ytr[:5000], m.predict(A[:5000])), fit_seconds=ft)
    results.append(r); print(r)

# ---------- 6. Final model: Logistic Regression (tiny, and every weight is readable) ----------
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=RNG)
Cs, vc = [0.01, 0.03, 0.1, 0.3, 1, 3, 10], []
for c in Cs:
    m = LogisticRegression(C=c, max_iter=400).fit(Xa, ya)
    vc.append(dict(C=c, train=m.score(Xa, ya), val=m.score(Xv, yv)))
bestC = max(vc, key=lambda d: d['val'])['C']
lr = LogisticRegression(C=bestC, max_iter=600).fit(Xtr, ytr)
cv = cross_val_score(LogisticRegression(C=bestC, max_iter=400), Xtr, ytr, cv=5)
P = lr.predict_proba(Xte); pred = P.argmax(1); conf = P.max(1)
bins = np.minimum((conf * 10).astype(int), 9)
calib = [dict(confidence=float(conf[bins == b].mean()), accuracy=float((pred == yte)[bins == b].mean()),
              n=int((bins == b).sum())) for b in range(10) if (bins == b).any()]
pr, rc, f1, sup = precision_recall_fscore_support(yte, pred, zero_division=0)
final = dict(C=bestC, accuracy=accuracy_score(yte, pred), top3_accuracy=top_k_accuracy_score(yte, P, k=3),
             roc_auc=roc_auc_score(yte, P, multi_class='ovr'), cv_mean=cv.mean(), cv_std=cv.std(),
             confusion=confusion_matrix(yte, pred).tolist(),
             per_role=[dict(role=r, precision=pr[i], recall=rc[i], f1=f1[i], support=int(sup[i])) for i, r in enumerate(roles)])
print('FINAL', {k: v for k, v in final.items() if k not in ('confusion', 'per_role')})

# ---------- 7. Export one small file for the app ----------
tw = df.sample(3000, random_state=RNG)
R = lambda a, n=4: np.round(np.asarray(a, dtype=float), n).tolist()
out = dict(
    roles=roles, vocab=vocab, freq=freq.astype(int).tolist(), emb=R(emb, 3), xy=R(xy, 3),
    clusters=clusters.tolist(), cluster_names=cl_names,
    kmeans=dict(k=ks, inertia=inertia, silhouette=sil, best_k=best_k),
    role_profile=R(np.vstack([np.asarray(X[y == i].mean(0)).ravel() for i in range(len(roles))]), 3),
    role_counts=[int((y == i).sum()) for i in range(len(roles))],
    coef=R(lr.coef_), intercept=R(lr.intercept_),
    models=results, validation_curve=vc, calibration=calib, final=final,
    twins=dict(title=tw.job_title.tolist(), company=tw.company.fillna('').tolist(), role=tw.role.tolist(), ix=tw.ix.tolist()),
    xyz=R(xyz, 3), alias=ALIAS, test=dict(ix=[df.ix[i] for i in ite], y=yte.tolist()),
    n_postings=int(len(df)), svd_variance=float(svd.explained_variance_ratio_.sum()),
)
json.dump(out, open('careertwin.json', 'w'), default=float)
print('saved careertwin.json')

# ---------- 8. MLflow experiment log (Module V) ----------
try:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mlflow'], check=True)
    import mlflow
    mlflow.set_experiment('careertwin')
    for r in results:
        with mlflow.start_run(run_name=r['model']):
            mlflow.log_param('model', r['model'])
            mlflow.log_metrics({k: float(v) for k, v in r.items() if k != 'model' and v is not None})
    mlflow.search_runs().to_csv('mlflow_runs.csv', index=False)
    import shutil; shutil.rmtree('mlruns', ignore_errors=True)
except Exception as e:
    print('mlflow skipped:', e)
print('done')
