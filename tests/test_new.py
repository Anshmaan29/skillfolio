"""Tests for the skill test engine and the coder models. Run: .venv/bin/python tests/test_new.py
Online checks (LeetCode, Codeforces, GitHub) run only with --online."""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import numpy as np
import assess, coding

results = []
def check(name, ok, detail=''):
    results.append(bool(ok)); print(('PASS' if ok else 'FAIL'), '-', name, ('| ' + str(detail)) if detail else '')

# ---- question bank ----
B = assess.BANK
check('30 skills, 18 questions each, 6 per difficulty', len(B) == 30 and all(sorted(q[1] for q in v['q']) == [1] * 6 + [2] * 6 + [3] * 6 for v in B.values()))
check('every question has 1 right + 3 distinct wrong options', all(len(set(q[4])) == 3 and q[3] not in q[4] for v in B.values() for q in v['q']))
check('every tested skill exists in the role model vocabulary', all(s in json.loads((ROOT / 'data' / 'careertwin.json').read_text())['vocab'] for s in B))
check('test parameters came from Kaggle training (not defaults)', assess.KT is not None, f'guess {assess.GUESS:.2f}, slip {assess.SLIP:.2f}, a {assess.A:.2f}, b {[round(b, 2) for b in assess.B]}, stop sd {assess.SD_STOP}')

# ---- adaptive test behaves like a real measurement ----
rng = np.random.default_rng(0)
def run(theta, seed, sure=True):
    t = assess.Test('docker', seed=seed)
    while (q := t.next_question()):
        right = rng.random() < assess.p_correct(theta, q['level'])
        t.answer(q['correct'] if right else q['wrong'][0], sure=sure)
    return t
sims = {th: [run(th, s) for s in range(200)] for th in (-2, 0, 2)}
lens = [len(t.answers) for v in sims.values() for t in v]
check('every test asks between 5 and 10 questions', min(lens) >= 5 and max(lens) <= 10, f'average {np.mean(lens):.1f}')
mast = {th: np.mean([t.mastery() for t in v]) for th, v in sims.items()}
check('stronger simulated students get higher mastery', mast[-2] < mast[0] < mast[2], {k: round(v) for k, v in mast.items()})
est = np.array([t.estimate()[0] for v in sims.values() for t in v]); tru = np.repeat(list(sims), 200)
check('ability estimate correlates with true ability', np.corrcoef(est, tru)[0, 1] > 0.7, f'r = {np.corrcoef(est, tru)[0, 1]:.2f}')
t = run(0, 1); check('no question is asked twice', len({a['id'] for a in t.answers}) == len(t.answers))
check('questions cover every subtopic', len({a['sub'] for a in t.answers}) == 3)
lvl = [q[1] for q in [(None, a['level']) for a in run(3, 2).answers]]
check('a very strong student gets pushed to hard questions', 3 in lvl, lvl)
t1 = assess.Test('python', seed=5); q = t1.next_question(); t1.answer(q['correct'], sure=True)
t2 = assess.Test('python', seed=5); q = t2.next_question(); t2.answer(q['correct'], sure=False)
check('a correct "not sure" answer counts less than a sure one', t2.mastery() < t1.mastery(), f'{t1.mastery():.1f} vs {t2.mastery():.1f}')
t3 = assess.Test('python', seed=5); q = t3.next_question(); t3.answer(q['wrong'][0])
check('a wrong answer lowers mastery below the starting point', t3.mastery() < assess.Test('python').mastery())
check('belief stays a probability distribution', abs(sum(t1.curves[-1]) - 1) < 1e-9)
check('forgetting: retention falls over time and starts at 1', abs(assess.retention(0) - 1) < 1e-9 and assess.retention(30) < assess.retention(1) < 1, f'1 day {assess.retention(1):.2f}, 30 days {assess.retention(30):.2f}')

# Hosted attempts use the current visitor's history only.
visitor_a, visitor_b = {}, {}
result = run(0, 9).report()
assess.save_result(result, now=100, history=visitor_a)
assess.save_result(result, now=200, history=visitor_a)
check('demo retakes stay in one visitor history', assess.mastery_today('docker', now=300, history=visitor_a)['tests'] == 2)
check('another visitor gets an empty history', assess.mastery_today('docker', now=300, history=visitor_b) is None)

# ---- coder models ----
if coding.CF:
    CF = coding.CF
    errs = [abs(coding.predict_rating(coding.features(c['solved'], c['n_tried'])) - c['pred']) for c in CF['check']]
    check('app recomputes the Kaggle model predictions exactly', max(errs) < 0.5, f'max difference {max(errs):.4f} over {len(errs)} coders')
    best = min(CF['models'], key=lambda m: m['mae'])
    check('rating model beats guessing the average', best['mae'] < 0.7 * CF['baseline_mae'], f'{best["model"]} MAE {best["mae"]:.0f} vs baseline {CF["baseline_mae"]:.0f}')
    check('percentile is monotonic', coding.percentile(1000) < coding.percentile(1600) < coding.percentile(2400))
    easy = [[r, ['implementation']] for r in [800] * 40]; hard = easy + [[r, ['dp', 'graphs']] for r in range(1600, 2400, 20)]
    pe, ph = (coding.predict_rating(coding.features(s, len(s))) for s in (easy, hard))
    check('solving harder problems raises the predicted rating', ph > pe + 200, f'{pe:.0f} -> {ph:.0f}')
else:
    print('SKIP - models/coders.json not downloaded yet')

if '--online' in sys.argv:
    for kind, user, test in (('codeforces', 'tourist', lambda r: r['rating'] > 3000 and len(r['solved']) > 500),
                             ('leetcode', 'lee215', lambda r: r['solved'].get('Hard', 0) > 50),
                             ('github', 'torvalds', lambda r: r['repos'] > 0 and 'C' in r['langs'])):
        try: r = getattr(coding, kind)(user); check(f'live fetch from {kind} ({user})', test(r))
        except Exception as e: check(f'live fetch from {kind} ({user})', False, e)
        time.sleep(2)
print(f'\n{sum(results)}/{len(results)} passed'); sys.exit(0 if all(results) else 1)
