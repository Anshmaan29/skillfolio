"""Coding profile: fetch public stats (LeetCode, Codeforces, GitHub) and analyse them with the models trained on real Codeforces users."""
import json, time
from collections import Counter
from pathlib import Path
import numpy as np, requests

ROOT = Path(__file__).parent
_cf = ROOT / 'models' / 'coders.json'
CF = json.loads(_cf.read_text()) if _cf.exists() else None
H = {'User-Agent': 'Skillfolio student project'}

# ---------------- fetchers ----------------
LC_QUERY = """query($u:String!){ matchedUser(username:$u){ username
  submitStatsGlobal{ acSubmissionNum{ difficulty count } }
  tagProblemCounts{ advanced{tagName problemsSolved} intermediate{tagName problemsSolved} fundamental{tagName problemsSolved} }
  problemsSolvedBeatsStats{ difficulty percentage } }
  userContestRanking(username:$u){ rating topPercentage attendedContestsCount } }"""

def leetcode(user):
    r = requests.post('https://leetcode.com/graphql', json=dict(query=LC_QUERY, variables=dict(u=user)),
                      headers=dict(H, Referer='https://leetcode.com'), timeout=20).json()
    m = (r.get('data') or {}).get('matchedUser')
    if not m: raise ValueError(f'LeetCode user "{user}" not found')
    solved = {d['difficulty']: d['count'] for d in m['submitStatsGlobal']['acSubmissionNum']}
    tags = {t['tagName']: t['problemsSolved'] for g in m['tagProblemCounts'].values() for t in g}
    beats = {d['difficulty']: d['percentage'] for d in (m.get('problemsSolvedBeatsStats') or []) if d['percentage'] is not None}
    return dict(user=m['username'], solved=solved, tags=tags, beats=beats, contest=r['data'].get('userContestRanking'))

def codeforces(handle):
    def api(method, **p):
        j = requests.get(f'https://codeforces.com/api/{method}', params=p, headers=H, timeout=30).json()
        if j.get('status') != 'OK': raise ValueError(j.get('comment', 'Codeforces error'))
        return j['result']
    info = api('user.info', handles=handle)[0]; time.sleep(2.1)
    subs = api('user.status', handle=handle)
    solved, tried = {}, set()
    for s in subs:
        p = s.get('problem', {}); key = (p.get('contestId'), p.get('index'))
        if key[0] is None: continue
        tried.add(key)
        if s.get('verdict') == 'OK' and key not in solved: solved[key] = [p.get('rating'), p.get('tags', [])]
    return dict(handle=info['handle'], rating=info.get('rating'), maxRating=info.get('maxRating'), rank=info.get('rank', 'unrated'),
                solved=list(solved.values()), solved_ids={f'{a}{b}' for a, b in solved}, n_tried=len(tried), n_subs=len(subs))

LANG_SKILL = {'Python': 'python', 'Jupyter Notebook': 'python', 'JavaScript': 'javascript', 'TypeScript': 'typescript', 'Java': 'java',
              'C++': 'c++', 'C#': 'c#', 'Go': 'go', 'HTML': 'html', 'CSS': 'css', 'Dockerfile': 'docker', 'HCL': 'terraform',
              'Shell': 'bash', 'Kotlin': 'kotlin', 'Swift': 'swift', 'Rust': 'rust', 'PHP': 'php', 'Ruby': 'ruby', 'Scala': 'scala', 'R': 'r'}
def github(user):
    repos = requests.get(f'https://api.github.com/users/{user}/repos', params=dict(per_page=100, sort='pushed'), headers=H, timeout=20)
    if repos.status_code == 404: raise ValueError(f'GitHub user "{user}" not found')
    repos = repos.json()
    if isinstance(repos, dict): raise ValueError(repos.get('message', 'GitHub error'))
    own = [r for r in repos if not r.get('fork')]
    langs = Counter()
    for r in own[:10]:                                   # byte counts for the 10 most recently pushed repos
        lb = requests.get(r['languages_url'], headers=H, timeout=20)
        if lb.status_code == 200: langs.update(lb.json())
    for r in own[10:]:
        if r.get('language'): langs[r['language']] += 1
    return dict(user=user, repos=len(own), stars=sum(r.get('stargazers_count', 0) for r in own), langs=dict(langs.most_common()),
                recent=[dict(name=r['name'], lang=r.get('language'), stars=r.get('stargazers_count', 0), pushed=r.get('pushed_at', '')[:10]) for r in own[:6]])

# ---------------- Codeforces analysis (trained models) ----------------
def features(solved, n_tried):
    """Exactly the features used in training (training_cf/train_cf.py)."""
    E, TAGS = CF['edges'], CF['tags']
    r = np.array([x[0] for x in solved if x[0]], dtype=float)
    hist = np.histogram(r, bins=E + [10000])[0] if len(r) else np.zeros(len(E))
    top = np.sort(r)[-20:] if len(r) else np.array([0.0])
    tc = Counter(t for _, ts in solved for t in ts); n = len(solved)
    return np.concatenate([np.log1p(hist), [np.log1p(n), np.log1p(n_tried), n / max(n_tried, 1), top.mean() / 1000,
                                           (r.max() if len(r) else 0) / 1000, np.median(r) / 1000 if len(r) else 0],
                           [np.log1p(tc[t]) for t in TAGS]])

def predict_rating(x):
    if CF['deployed'].startswith('Ridge'):
        m = CF['ridge']; return float(((x - m['mean']) / m['scale']) @ np.array(m['coef']) + m['intercept'])
    m = CF['mlp']; h = (x - np.array(m['mean'])) / np.array(m['scale'])
    for k, (W, b) in enumerate(zip(m['W'], m['b'])):
        h = h @ np.array(W) + np.array(b)
        if k < len(m['W']) - 1: h = np.maximum(h, 0)
    return float(h[0])

def percentile(rating):
    return float(np.interp(rating, CF['rating_quantiles'], np.arange(101)))

def style(solved):
    T = CF['tags']; c = np.array([sum(1 for _, ts in solved if t in ts) for t in T], dtype=float)
    if c.sum() < 30: return None
    k = CF['kmeans']; z = (c / c.sum() - np.array(k['mu'])) / np.array(k['sd'])
    i = int(np.argmin(((np.array(k['centers']) - z) ** 2).sum(1))); return dict(k['styles'][i], id=i)

def next_level(cf):
    """Compare your topic counts with the median coder one band (200 points) above you."""
    T = CF['tags']; base = cf['rating'] or 800
    band = str(int(min(max(base, 800) // 200 * 200 + 200, max(int(b) for b in CF['bands']))))
    prof = CF['bands'][band]; mine = np.array([sum(1 for _, ts in cf['solved'] if t in ts) for t in T], dtype=float)
    gaps = sorted(((T[i], prof['tags'][i] - mine[i], prof['tags'][i], mine[i]) for i in range(len(T))), key=lambda g: -g[1])
    return int(band), prof, [g for g in gaps if g[1] > 0][:6], mine

def recommend(cf, tags, n=6):
    P = CF['problems']; r = cf['rating'] or 1000; out = []
    for i in np.argsort(P['solved'])[::-1]:
        if P['id'][i] in cf['solved_ids'] or not (r + 100 <= P['rating'][i] <= r + 300) or not set(P['tags'][i]) & set(tags): continue
        out.append(dict(id=P['id'][i], name=P['name'][i], rating=P['rating'][i], tags=P['tags'][i], solved=P['solved'][i]))
        if len(out) == n: break
    return out

def leetcode_score(lc):
    """Transparent rule, not a trained model: hard problems count more."""
    s = lc['solved']; return min(100.0, (s.get('Easy', 0) * 1 + s.get('Medium', 0) * 2.5 + s.get('Hard', 0) * 5) / 4)

def github_skills(gh, vocab):
    return [LANG_SKILL[l] for l in gh['langs'] if l in LANG_SKILL and LANG_SKILL[l] in vocab]
