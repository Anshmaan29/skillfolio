"""Skill test engine (no Streamlit here, so it can be unit-tested).
- Picks each next question adaptively with Item Response Theory (IRT): the difficulty that tells us most about you.
- Updates a belief about your ability after every answer with Bayes' theorem (prior x likelihood).
- Scores each subtopic with Bayesian Knowledge Tracing (BKT).
- Lets mastery fade over time with a forgetting curve.
All numbers below (guess, slip, discrimination, difficulty levels, BKT, forgetting) were fitted on real student answers (Riiid, Kaggle)."""
import json, random, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
BANK = json.loads((ROOT / 'questions' / 'bank.json').read_text())
SKILLS = sorted(BANK)
_kt = ROOT / 'models' / 'kt.json'
KT = json.loads(_kt.read_text()) if _kt.exists() else None
if KT and 'riiid' in KT:
    _i, _b = KT['riiid']['irt'], KT['riiid']['bkt']['median']
    GUESS, SLIP, A, B = _i['guess'], _i['slip'], _i['a_median'], _i['b_levels']
    BKT = (_b['L0'], _b['T'], _b['G'], _b['S'])
    FORGET = KT['curves']['forgetting_fit']
    stop = [s for s in KT['cat']['stopping'] if s['avg_questions'] <= 8.5]
    SD_STOP = max(stop, key=lambda s: -s['rmse'])['sd_stop'] if stop else 0.7
else:                                            # used only before the Kaggle results are downloaded
    GUESS, SLIP, A, B = 0.2, 0.05, 1.2, [-1.0, 0.0, 1.0]
    BKT, FORGET, SD_STOP = (0.4, 0.1, 0.2, 0.1), dict(p_inf=0.6, p0=0.8, tau_days=7.0), 0.7
MIN_Q, MAX_Q = 5, 10
TH = np.linspace(-4, 4, 81)
LEVELS = {1: 'easy', 2: 'medium', 3: 'hard'}
sig = lambda z: 1 / (1 + np.exp(-z))

def p_correct(theta, level):
    """Chance of answering a question of this level correctly, including lucky guesses and careless slips."""
    return GUESS + (1 - GUESS - SLIP) * sig(A * (theta - B[level - 1]))

def info(theta, level):
    """Fisher information: how much one answer at this level would tell us about someone at this ability."""
    P = p_correct(theta, level)
    return A ** 2 * (P - GUESS) ** 2 * (1 - SLIP - P) ** 2 / ((1 - GUESS - SLIP) ** 2 * P * (1 - P))

class Test:
    def __init__(self, skill, seed=None):
        self.skill, self.rng = skill, random.Random(seed)
        self.items = [dict(id=i, sub=q[0], level=q[1], text=q[2], correct=q[3], wrong=q[4], why=q[5]) for i, q in enumerate(BANK[skill]['q'])]
        self.sub_names = BANK[skill]['sub']
        self.lp = -0.5 * TH ** 2                   # log of the prior: ability ~ Normal(0, 1)
        self.answers, self.current, self.curves = [], None, [self.belief().tolist()]

    def belief(self):
        p = np.exp(self.lp - self.lp.max()); return p / p.sum()
    def estimate(self):
        p = self.belief(); m = float((p * TH).sum()); return m, float(np.sqrt((p * (TH - m) ** 2).sum()))
    def done(self):
        n = len(self.answers); return n >= MAX_Q or (n >= MIN_Q and self.estimate()[1] < SD_STOP)

    def next_question(self):
        """Choose the difficulty with the most information at the current estimate, then an unused question,
        preferring the subtopic asked least so far."""
        if self.done(): self.current = None; return None
        used = {a['id'] for a in self.answers}; free = [q for q in self.items if q['id'] not in used]
        theta = self.estimate()[0]; levels = {q['level'] for q in free}
        lvl = max(levels, key=lambda l: info(theta, l))
        asked = {s: sum(a['sub'] == s for a in self.answers) for s in self.sub_names}
        pool = [q for q in free if q['level'] == lvl]; least = min(asked[q['sub']] for q in pool)
        q = self.rng.choice([q for q in pool if asked[q['sub']] == least])
        opts = [q['correct']] + q['wrong']; self.rng.shuffle(opts)
        self.current = dict(q, options=opts, shown=time.time()); return self.current

    def answer(self, choice, sure=True):
        """Bayes' theorem: new belief = old belief x chance of this answer at each ability.
        A correct answer marked 'not sure' counts half, because it may be a guess."""
        q = self.current; ok = choice == q['correct']
        P = p_correct(TH, q['level']); w = 0.5 if (ok and not sure) else 1.0
        self.lp = self.lp + w * (np.log(P) if ok else np.log(1 - P))
        self.answers.append(dict(id=q['id'], sub=q['sub'], level=q['level'], text=q['text'], chosen=choice, correct=q['correct'],
                                 ok=ok, sure=sure, secs=time.time() - q['shown'], why=q['why']))
        self.curves.append(self.belief().tolist()); self.current = None
        return ok

    def mastery(self):
        """Chance you would solve a typical medium question without guessing, averaged over the belief."""
        return float((self.belief() * sig(A * (TH - B[1]))).sum() * 100)

    def subtopics(self):
        """Bayesian Knowledge Tracing per subtopic: P(learned) updated after each answer, in order."""
        L0, T, G, S = BKT; out = {}
        for code, name in self.sub_names.items():
            pL = L0; seen = 0
            for a in self.answers:
                if a['sub'] != code: continue
                seen += 1; pc = pL * (1 - S) + (1 - pL) * G
                post = pL * (1 - S) / pc if a['ok'] else pL * S / (1 - pc)
                pL = post + (1 - post) * T
            out[name] = dict(p=pL * 100, n=seen)
        return out

    def report(self):
        m, sd = self.estimate(); mast = self.mastery()
        level = 'Expert' if mast >= 85 else 'Strong' if mast >= 65 else 'Developing' if mast >= 40 else 'Beginner'
        return dict(skill=self.skill, mastery=mast, theta=m, sd=sd, level=level, n=len(self.answers),
                    correct=sum(a['ok'] for a in self.answers), subtopics=self.subtopics(),
                    overconfident=[a for a in self.answers if a['sure'] and not a['ok']],
                    lucky=[a for a in self.answers if a['ok'] and not a['sure']],
                    secs=float(np.mean([a['secs'] for a in self.answers])) if self.answers else 0.0,
                    wrong=[a for a in self.answers if not a['ok']])

# ---------------- memory over time ----------------
HIST = ROOT / 'data' / 'history.json'
def load_history():
    try: return json.loads(HIST.read_text())
    except Exception: return {}
def save_result(rep, now=None, history=None):
    h = load_history() if history is None else history; h.setdefault(rep['skill'], []).append(dict(ts=now or time.time(), mastery=rep['mastery'], theta=rep['theta'], sd=rep['sd'], n=rep['n']))
    if history is None:
        HIST.write_text(json.dumps(h, indent=1))
    return h

def retention(days):
    """Forgetting curve fitted on Riiid: share of what you knew that you still get right after this many days."""
    p_inf, p0, tau = FORGET['p_inf'], FORGET['p0'], FORGET['tau_days']
    return (p_inf + (p0 - p_inf) * np.exp(-days / tau)) / p0

def mastery_today(skill, now=None, history=None):
    h = (load_history() if history is None else history).get(skill)
    if not h: return None
    last = h[-1]; days = ((now or time.time()) - last['ts']) / 86400
    return dict(tested=last['mastery'], today=last['mastery'] * float(retention(days)), days=days, first=h[0]['mastery'], tests=len(h))
