"""CareerTwin tests. Run:  .venv/bin/python tests/test_careertwin.py
Each test prints PASS/FAIL. They check that the models are real (learned from data), not hard-coded."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import core, deep
from core import D, V, ROLES, N, IX, probs, flags, evolve, path, effort

results = []
def check(name, ok, detail=''):
    results.append(ok); print(('PASS' if ok else 'FAIL'), '-', name, ('| ' + str(detail)) if detail else '')
top = lambda skills: ROLES[int(probs(flags([IX[s] for s in skills])[None])[0].argmax())]
P = lambda idx, t: float(probs(flags(idx)[None])[0, t])

# ---- 1. the skill model reproduces its Kaggle score on data it never saw ----
a = core.audit()
check('held-out accuracy recomputed locally equals the Kaggle number', abs(a['accuracy'] - D['final']['accuracy']) < 1e-6, f"{a['accuracy']:.4f} on {a['n']} postings")
check('accuracy collapses when skill columns are scrambled (so it is not hard-coded)', a['scrambled'] < 0.25, f"{a['scrambled']:.3f} vs chance {a['chance']:.3f}")
X, y = core.test_set(); pr = probs(X)
check('probabilities are valid (each row sums to 1)', np.allclose(pr.sum(1), 1))
check('every role gets predicted at least sometimes', len(set(pr.argmax(1))) == len(ROLES))

# ---- 2. it behaves sensibly on hand-made profiles ----
for skills, want in [(['python', 'sql', 'tableau', 'excel', 'data analysis'], 'Data Analyst'),
                     (['react', 'html', 'css', 'javascript', 'typescript'], 'Frontend Developer'),
                     (['kubernetes', 'docker', 'terraform', 'aws', 'ci/cd'], 'DevOps / Cloud'),
                     (['machine learning', 'pytorch', 'tensorflow', 'python', 'deep learning'], 'ML Engineer')]:
    check(f'{skills[:3]}... -> {want}', top(skills) == want, top(skills))
have = [IX[s] for s in ['python', 'sql', 'html', 'css', 'javascript', 'git', 'communication', 'java']]
t = ROLES.index('ML Engineer'); best = int(np.argmax(D['coef'][t] * ~np.isin(np.arange(N), have)))
check('adding the top-weighted missing skill raises the score', P(have + [best], t) > P(have, t), f'+{V[best]}: {P(have, t):.3f} -> {P(have + [best], t):.3f}')

# ---- 3. genetic algorithm ----
t0 = time.time(); best_set, hist, bests = evolve(have, t, 3, 0.2); dt = time.time() - t0
check('GA returns 3 distinct new skills', len(set(best_set)) == 3 and not set(best_set) & set(have), [V[i] for i in best_set])
check('GA fitness never gets worse (elitism)', all(b >= a_ - 1e-12 for a_, b in zip(hist, hist[1:])), f'{hist[0]:.3f} -> {hist[-1]:.3f}')
check('GA improves the role probability', P(have + best_set, t) > P(have, t), f'{P(have, t):.3f} -> {P(have + best_set, t):.3f}')
rng = np.random.default_rng(0); cand = [i for i in range(N) if i not in have]; cost = effort(have)
fit = lambda s: P(have + list(s), t) - 0.2 * cost[list(s)].mean()
rand = [fit(rng.choice(cand, 3, replace=False)) for _ in range(2000)]
check('GA beats 2,000 random skill sets', fit(best_set) >= max(rand), f'GA {fit(best_set):.3f} vs best random {max(rand):.3f}')
check('GA is repeatable with the same seed', evolve(have, t, 3, 0.2)[0] == best_set)
check('GA result changes when the target role changes', set(evolve(have, ROLES.index('Frontend Developer'), 3, 0.2)[0]) != set(best_set))
check('GA runs fast enough for a live demo', dt < 5, f'{dt:.2f}s')
check('learning path keeps the same skills, just reordered', sorted(path(have, best_set)) == sorted(best_set))

# ---- 4. skill embeddings learned real relationships ----
near = lambda s: [V[i] for i in np.argsort(-(D['emb'] @ D['emb'][IX[s]]))[1:8]]
for s, friends in [('pytorch', {'tensorflow', 'deep learning', 'keras'}), ('html', {'css', 'javascript'}), ('docker', {'kubernetes'})]:
    check(f'neighbours of {s} make sense', bool(set(near(s)) & friends), near(s))

# ---- 5. deep models (DistilBERT, int8 ONNX) ----
if deep.READY:
    R = deep.META['role']; check('deep role order matches the classical model', deep.META['roles'] == ROLES)
    t0 = time.time(); n = 150; pr = deep.role_probs(R['audit']['text'][:n]); acc = float((pr.argmax(1) == np.array(R['audit']['y'][:n])).mean())
    check('text model accuracy on unseen job descriptions (run on this laptop)', acc > 0.6, f'{acc:.3f} on {n} docs, {(time.time() - t0) / n * 1000:.0f} ms each, Kaggle said {R["quantised"]["accuracy"]:.3f}')
    for txt, want in [('Build responsive user interfaces with React, TypeScript and CSS. Work with designers on component libraries and web accessibility.', 'Frontend Developer'),
                      ('Train and deploy deep learning models with PyTorch. Build feature pipelines, run experiments and serve models in production.', 'ML Engineer'),
                      ('Monitor SIEM alerts, investigate incidents, run vulnerability scans and harden firewalls against threats.', 'Cybersecurity')]:
        g = ROLES[int(deep.role_probs([txt])[0].argmax())]; check(f'text model: "{txt[:38]}..." -> {want}', g == want, g)
    s = 'I built dashboards in Tableau and wrote SQL queries, then trained models with scikit-learn and deployed them using Docker.'
    got = [x[2].lower() for x in deep.ner(s)]
    check('skill tagger finds the skills in a sentence', {'tableau', 'sql', 'docker'} <= set(got), got)
    got2 = [x[2].lower() for x in deep.ner('We orchestrate workflows with Dagster and store vectors in Qdrant, and enjoy long walks on Sundays.')]
    check('skill tagger generalises to tools outside our 500-skill list', any(w in ' '.join(got2) for w in ('dagster', 'qdrant')), got2)
    check('skill tagger ignores ordinary words', not any('walk' in g or 'sunday' in g for g in got2), got2)
    imp = deep.sentence_impact('I trained neural networks in PyTorch for image classification. I enjoy playing cricket on weekends with friends. I deployed models with Docker on AWS.', t)
    check('occlusion: the PyTorch sentence matters more than the cricket one', imp and 'cricket' not in imp[0][0], [(s_[:30], round(d, 3)) for s_, d in imp])
else:
    print('SKIP - deep model files not in models/')

print(f'\n{sum(results)}/{len(results)} passed'); sys.exit(0 if all(results) else 1)
