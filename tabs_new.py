"""Skill test tab, Coding profile tab, and the extra Model lab sections."""
import html, time
import numpy as np, streamlit as st, plotly.graph_objects as go
import assess, coding
from ui import heading, score_readout, L, PAPER, AMBER, CORAL, TEAL, MUTED, LINE, WINE_WASH

note = lambda s: st.html(f'<div class="note">{s}</div>')
card = lambda s: st.html(f'<div class="card">{s}</div>')
esc = html.escape
SKILL_NAMES = {'aws': 'AWS', 'c++': 'C++', 'ci/cd': 'CI/CD', 'css': 'CSS', 'html': 'HTML',
               'javascript': 'JavaScript', 'mlops': 'MLOps', 'mongodb': 'MongoDB', 'nlp': 'NLP',
               'node.js': 'Node.js', 'oop': 'OOP', 'pytorch': 'PyTorch', 'rest api': 'REST API',
               'sql': 'SQL', 'typescript': 'TypeScript'}


def skill_name(skill):
    return SKILL_NAMES.get(skill, skill.capitalize())

# ================= SKILL TEST =================
def _start(skill):
    st.session_state.test = assess.Test(skill, seed=time.time_ns())
    st.session_state.feedback = None
    st.session_state.saved = False
    st.session_state.sure = True
def _answer(question_id, opt):
    S = st.session_state
    test = S.get('test')
    q = test.current if test else None
    if not q or q['id'] != question_id or S.get('feedback'):
        return
    ok = test.answer(opt, S.get('sure', True))
    S.feedback = dict(ok=ok, chosen=opt, correct=q['correct'], why=q['why'])
def _next(): st.session_state.feedback = None
def _quit(): st.session_state.test = None; st.session_state.feedback = None

def belief_chart(t):
    prior, now = np.array(t.curves[0]), np.array(t.curves[-1]); m, sd = t.estimate()
    fig = go.Figure([go.Scatter(x=assess.TH, y=prior, line=dict(color=MUTED, dash='dot'), name='before the test'),
                     go.Scatter(x=assess.TH, y=now, fill='tozeroy', fillcolor=WINE_WASH, line=dict(color=AMBER, width=2), name='current estimate')])
    for k, lab in enumerate(['easy', 'medium', 'hard']):
        fig.add_vline(x=assess.B[k], line=dict(color=LINE, dash='dash'), annotation_text=lab, annotation_font_color=MUTED)
    fig.update_xaxes(title='Ability (θ)', range=[-3.5, 3.5]); fig.update_yaxes(visible=False)
    L(fig, 265)
    st.caption(f'Estimated ability {m:+.2f}. Uncertainty (SD) {sd:.2f}.')

def skill_test_tab(have_names, add_skills, history=None):
    S = st.session_state
    t = S.get('test')
    if t is None:
        heading('Skill test', 'Check what you know with a short test. The difficulty adapts as you answer.')
        setup, brief = st.columns([3, 2], gap='medium')
        with setup.container(key='test-setup'):
            st.markdown('## Choose a skill')
            default = next((s for s in have_names if s in assess.SKILLS), 'docker')
            S.setdefault('test_skill', default)
            skill = st.selectbox('Skill to test', assess.SKILLS, format_func=skill_name, key='test_skill')
            st.button('Start test', type='primary', on_click=_start, args=(skill,))
            if history is not None:
                st.caption('Test history is kept for this session.')
            mt = assess.mastery_today(skill, history=history)
            if mt:
                st.caption(f'Last result {mt["tested"]:.0f}%. Estimated mastery today {mt["today"]:.0f}%.')
        with brief:
            st.html(f'<div class="folio-test-brief"><strong>{assess.MIN_Q}–{assess.MAX_Q} questions</strong>'
                    f'<small>{len(assess.SKILLS)} skills, {sum(len(skill["q"]) for skill in assess.BANK.values())} questions in the bank</small>'
                    '<p>The test picks the difficulty that best reduces uncertainty about your level. Afterward you can review your answers and see where to practice.</p></div>')
        hist = assess.load_history() if history is None else history
        if hist:
            st.markdown('### Your tested skills')
            rows = sorted(((s, assess.mastery_today(s, history=hist)) for s in hist), key=lambda x: -x[1]['today'])
            fig = go.Figure([go.Bar(y=[skill_name(s) for s, _ in rows], x=[m['tested'] for _, m in rows], orientation='h', marker_color=LINE, name='When tested'),
                             go.Bar(y=[skill_name(s) for s, _ in rows], x=[m['today'] for _, m in rows], orientation='h', marker_color=TEAL, name='Estimated today')])
            fig.update_xaxes(range=[0, 100], title='mastery %'); L(fig, max(220, 40 * len(rows)), barmode='overlay')
        return

    n = len(t.answers)
    feedback = S.get('feedback')
    if t.done() and not feedback:
        report_view(t, add_skills, history)
        st.button('Choose another skill', on_click=_quit)
        return

    heading(f'{skill_name(t.skill)} skill test', 'A correct answer counts less toward mastery when you mark it as a guess.')
    left, right = st.columns([3, 2], gap='medium')
    with left.container(key='question'):
        if feedback:
            col = TEAL if feedback['ok'] else CORAL
            st.markdown(f'## {"Correct" if feedback["ok"] else "Incorrect"}')
            card(
                ('' if feedback['ok'] else f'<p>You chose <b>{esc(feedback["chosen"])}</b><br>'
                 f'Correct answer <b style="color:{col}">{esc(feedback["correct"])}</b></p>')
                + esc(feedback['why'])
            )
            st.button('Next question' if not t.done() else 'See my report', type='primary', on_click=_next)
        else:
            q = t.current or t.next_question()
            st.html(f'<div class="folio-question-meta">Question {n + 1} / {assess.MAX_Q} &nbsp; | &nbsp; '
                    f'{esc(t.sub_names[q["sub"]])} &nbsp; | &nbsp; {esc(assess.LEVELS[q["level"]])}</div>')
            st.html(f'<h2>{esc(q["text"])}</h2>')
            st.toggle('I’m sure of my answer', key='sure')
            for k, option in enumerate(q['options']):
                st.button(f'{"ABCD"[k]}.  {option}', key=f'opt{n}_{k}', width='stretch',
                          on_click=_answer, args=(q["id"], option))
        st.button('Quit test', on_click=_quit)

    with right:
        st.markdown('## Ability estimate')
        belief_chart(t)
        st.caption(f'{n} answered. The test ends after {assess.MIN_Q}–{assess.MAX_Q} answers.')
        attempts = []
        for i, answer in enumerate(t.answers, 1):
            result = "correct" if answer["ok"] else "incorrect"
            label = f'Question {i}: {result}'
            attempts.append(f'<span role="img" aria-label="{label}" title="{label}" data-result="{result}">{i}</span>')
        if attempts:
            st.html(f'<div class="folio-attempts">{"".join(attempts)}</div>')


def report_view(t, add_skills, history):
    S = st.session_state; r = t.report()
    if not S.get('saved'):
        assess.save_result(r, history=history)
        S.saved = True
    hist = (assess.load_history() if history is None else history).get(t.skill, [])
    heading(f'{skill_name(t.skill)} test report', 'Mastery estimates how likely you are to solve a typical medium question without guessing.')
    c1, c2 = st.columns([1, 1])
    with c1: score_readout(r['mastery'], f'{r["level"]} mastery', before=hist[-2]['mastery'] if len(hist) > 1 else None, color=TEAL)
    with c2:
        card(f'<b>{r["correct"]} of {r["n"]} correct</b><br><small>ability θ = {r["theta"]:+.2f} ± {r["sd"]:.2f} · about {r["secs"]:.0f} s per question</small>')
        if len(hist) > 1: card(f'<small>Change since your first test</small><br><b style="color:{TEAL if hist[-1]["mastery"] >= hist[0]["mastery"] else CORAL}">{hist[-1]["mastery"] - hist[0]["mastery"]:+.0f} points</b>')
        if r['overconfident']: card(f'<small style="color:{CORAL}">Confident but incorrect ({len(r["overconfident"])})</small><br>' + '<br>'.join(esc(a['text']) for a in r['overconfident'][:3]))
    st.markdown('## Subtopic mastery')
    st.caption('Estimated with Bayesian Knowledge Tracing from the answers in this test.')
    sub = r['subtopics']; names = list(sub)
    fig = go.Figure(go.Bar(x=[sub[s]['p'] for s in names], y=names, orientation='h', marker_color=[TEAL if sub[s]['p'] >= 60 else CORAL for s in names],
                           text=[f'{sub[s]["p"]:.0f}% · {sub[s]["n"]} q' for s in names], textposition='outside'))
    fig.update_xaxes(range=[0, 110], title='Estimated mastery (%)'); L(fig, 220)
    if len(hist) > 1:
        fig = go.Figure(go.Scatter(y=[h['mastery'] for h in hist], x=list(range(1, len(hist) + 1)), mode='lines+markers', line_color=AMBER))
        fig.update_xaxes(title='test number', dtick=1); fig.update_yaxes(title='mastery %', range=[0, 100]); L(fig, 240, title='Mastery across attempts')
    if r['wrong']:
        st.markdown('## Review missed questions')
        for a in r['wrong']: card(f'<b>{esc(a["text"])}</b><br><small>Your answer</small> {esc(a["chosen"])}<br><small>Correct answer</small> <span style="color:{TEAL}">{esc(a["correct"])}</span><br><small>{esc(a["why"])}</small>')
    c = st.columns(2)
    c[0].button('Retake test', on_click=_start, args=(t.skill,), width='stretch')
    if r['mastery'] >= 40: c[1].button(f'Add {skill_name(t.skill)} to my profile', on_click=add_skills, args=([t.skill],), width='stretch')

# ================= CODING PROFILE =================
@st.cache_data(ttl=900, show_spinner=False)
def _fetch(kind, user):
    return getattr(coding, kind)(user)

def coding_tab(V, add_skills):
    S = st.session_state
    heading('Coding profile', 'See what your public coding activity says about your strengths and where to practice next.')
    with st.container(key='coding-input'):
        st.markdown('## Public profiles')
        c1, c2, c3 = st.columns(3)
        lc_u = c1.text_input('LeetCode username', key='lc_user', placeholder='e.g. lee215')
        cf_u = c2.text_input('Codeforces handle', key='cf_user', placeholder='e.g. tourist')
        gh_u = c3.text_input('GitHub username', key='gh_user', placeholder='e.g. torvalds')
        analyse, examples = st.columns([1, 3])
        go_clicked = analyse.button('Analyze profiles', type='primary')
        examples.button('Use public examples', on_click=lambda: S.update(lc_user='lee215', cf_user='tourist', gh_user='torvalds'))
    handles = tuple(value.strip() for value in (lc_u, cf_u, gh_u))
    if go_clicked and not any(handles):
        st.warning('Enter at least one public username to continue.')
    elif go_clicked:
        res, err = {}, {}
        with st.spinner('Fetching public profiles…'):
            for kind, handle in zip(('leetcode', 'codeforces', 'github'), handles):
                if not handle:
                    continue
                try:
                    res[kind] = _fetch(kind, handle)
                except Exception as e:
                    err[kind] = str(e)[:160]
        S.code = dict(handles=handles, res=res, err=err)
    if 'code' not in S:
        return
    if S.code.get('handles') != handles:
        st.caption('Usernames changed. Select Analyze profiles to load the new results.')
        return
    res, err = S.code['res'], S.code['err']
    platforms = {'leetcode': 'LeetCode', 'codeforces': 'Codeforces', 'github': 'GitHub'}
    for k, e in err.items():
        st.warning(f'{platforms[k]} could not load this profile. {e}')
    lc, cf, gh = res.get('leetcode'), res.get('codeforces'), res.get('github')

    scores = []
    if cf and cf['rating'] and coding.CF: scores.append(('Codeforces percentile', coding.percentile(cf['rating'])))
    if lc: scores.append(('LeetCode score', coding.leetcode_score(lc)))
    if scores:
        c1, c2 = st.columns([1, 2])
        best = max(scores, key=lambda s: s[1])
        with c1: score_readout(best[1], 'DSA readiness')
        with c2: card('<small>DSA readiness is the higher of:</small><br>' + '<br>'.join(f'<b>{n}</b> {v:.0f}' for n, v in scores) +
                      '<br><small>The LeetCode score uses a weighted count of solved problems, capped at 100.</small>')

    if cf:
        st.markdown(f'### Codeforces · {esc(cf["handle"])}')
        if not coding.CF: st.info('Coder models not found in models/coders.json.')
        else:
            x = coding.features(cf['solved'], cf['n_tried']); pred = coding.predict_rating(x)
            m = st.columns(4)
            m[0].metric('Contest rating', cf['rating'] or 'unrated'); m[1].metric('Predicted rating', f'{pred:.0f}')
            m[2].metric('Percentile', f'{coding.percentile(cf["rating"] or 0):.0f}% of active coders'); m[3].metric('Problems solved', len(cf['solved']))
            st.caption('Rating predictions use solved problems. Predictions tend toward the middle of the training range.')
            if cf['rating']:
                gap = pred - cf['rating']
                note(f'Your practice looks like a {pred:.0f} coder. ' + ('That is above your contest rating: you know more than contests show. Do more timed virtual contests.' if gap > 100
                     else 'That is below your contest rating: you do well in contests, so push into harder practice problems.' if gap < -100 else 'That matches your contest rating closely.'))
            sty = coding.style(cf['solved'])
            if sty: card(f'<small>your coding style (K-means on topic mix)</small><br><b>strong in {", ".join(sty["top"])}</b><br><small>light on {", ".join(sty["low"])} · {sty["size"]} coders in this group, median rating {sty["median_rating"]:.0f}</small>')
            band, prof, gaps, mine = coding.next_level(cf)
            c1, c2 = st.columns(2)
            with c1:
                T = coding.CF['tags']; top = np.argsort(-np.array(prof['tags']))[:10]
                fig = go.Figure([go.Bar(y=[T[i] for i in top], x=[mine[i] for i in top], orientation='h', marker_color=AMBER, name='you'),
                                 go.Bar(y=[T[i] for i in top], x=[prof['tags'][i] for i in top], orientation='h', marker_color=LINE, name=f'median coder at {band}')])
                L(fig, 380, barmode='group', title=f'Topics: you vs coders rated {band}-{band + 199}')
            with c2:
                E = coding.CF['edges']; r = [x_[0] for x_ in cf['solved'] if x_[0]]; hist_ = np.histogram(r, bins=E + [10000])[0]
                fig = go.Figure([go.Bar(x=[str(e) for e in E], y=hist_, marker_color=AMBER, name='you'), go.Bar(x=[str(e) for e in E], y=prof['diff'], marker_color=LINE, name=f'median at {band}')])
                fig.update_xaxes(title='problem rating'); L(fig, 380, barmode='group', title='Problem difficulty you solve')
            top_band = (cf['rating'] or 0) >= band
            if top_band: note(f'You are already above the highest band in our data ({band}+), so the comparison below is with that band.')
            if gaps:
                st.markdown(f'**{"Compared with coders at " + str(band) if top_band else "To reach " + str(band)}, close these gaps:** ' + ', '.join(f'{g[0]} ({g[3]:.0f} vs {g[2]:.0f})' for g in gaps[:4]))
                for p in coding.recommend(cf, [g[0] for g in gaps[:3]]):
                    cid = ''.join(ch for ch in p['id'] if ch.isdigit()); idx = p['id'][len(cid):]
                    card(f'<b><a href="https://codeforces.com/problemset/problem/{cid}/{idx}" target="_blank" style="color:{PAPER}">{esc(p["name"])}</a></b> '
                         f'<small>· {p["rating"]} · {", ".join(p["tags"][:3])} · solved by {p["solved"]:,}</small>')
    if lc:
        st.markdown(f'### LeetCode · {esc(lc["user"])}')
        s = lc['solved']; m = st.columns(4)
        for col, d in zip(m, ['Easy', 'Medium', 'Hard']):
            col.metric(d, s.get(d, 0), f'beats {lc["beats"][d]:.0f}%' if lc['beats'].get(d) else None)
        con = lc.get('contest')
        m[3].metric('contest rating', f'{con["rating"]:.0f}' if con else '—', f'top {con["topPercentage"]:.1f}%' if con and con.get('topPercentage') else None, delta_color='off')
        tg = sorted(lc['tags'].items(), key=lambda kv: -kv[1])[:10]
        if tg:
            fig = go.Figure(go.Barpolar(r=[v for _, v in tg], theta=[k for k, _ in tg], marker_color=AMBER, opacity=.8))
            fig.update_polars(bgcolor='rgba(0,0,0,0)', radialaxis=dict(gridcolor=LINE, showticklabels=False), angularaxis=dict(gridcolor=LINE))
            L(fig, 400, title='Most-solved topics')
            weak = [k for k in ('Dynamic Programming', 'Graph', 'Tree', 'Binary Search', 'Greedy', 'Backtracking', 'Heap (Priority Queue)', 'Sliding Window') if lc['tags'].get(k, 0) < 10]
            if weak: note('Interview topics with fewer than 10 solved: ' + ', '.join(weak) + '.')
    if gh:
        st.markdown(f'### GitHub · {esc(gh["user"])}')
        langs = list(gh['langs'].items())[:10]
        c1, c2 = st.columns([3, 2])
        with c1:
            if langs:
                tot = sum(v for _, v in langs)
                L(go.Figure(go.Bar(x=[v / tot * 100 for _, v in langs], y=[k for k, _ in langs], orientation='h', marker_color=TEAL)), 320, title='Languages in your recent repos (%)')
        with c2:
            card(f'<b>{gh["repos"]} repos · {gh["stars"]} stars</b><br>' + '<br>'.join(f'<small>{esc(r["name"])} · {esc(r["lang"] or "—")} · {esc(r["pushed"])}</small>' for r in gh['recent']))
            proven = coding.github_skills(gh, set(V))
            if proven: st.button(f'Add {len(proven)} languages and tools to my profile', on_click=add_skills, args=(proven,), width='stretch')

# ================= MODEL LAB EXTRAS =================
def model_lab_extra():
    KT, CF = assess.KT, coding.CF
    if KT and 'riiid' in KT:
        R = KT['riiid']
        st.markdown('## Skill test research')
        note(f'Task: predict whether a student gets the next question right from their earlier answers. Trained on {R["sampled_users"]:,} real students '
             f'sampled from Riiid ({R["total_answers"]:,} answers in the source dataset). Evaluation used {R["n_test_answers"]:,} answers from held-out students.')
        rows = [dict(dataset='Riiid', **r) for r in R['results']] + ([dict(dataset='ASSISTments 2009', **r) for r in KT['assistments']['results']] if 'assistments' in KT else [])
        st.dataframe([{'dataset': r['dataset'], 'model': r['model'], 'AUC': round(r['auc'], 3), 'accuracy %': round(r['acc'] * 100, 1), 'log loss': round(r['logloss'], 3)} for r in rows], width='stretch', hide_index=True)
        st.caption('Full item difficulty reached 0.757 AUC. The three-level model, closest to this app’s question bank, scored 0.726 against the 0.728 baseline. Bayesian Knowledge Tracing scored 0.646 and is used for the subtopic estimates. The ASSISTments run did not complete.')
        c1, c2 = st.columns(2)
        with c1:
            bs = R['by_step']; fig = go.Figure()
            for k, col in zip([m['model'] for m in R['results']], [MUTED, '#77745a', AMBER, TEAL, '#765373', CORAL]):
                fig.add_scatter(x=[b['window'] for b in bs], y=[b[k] for b in bs], name=k, line_color=col)
            fig.update_xaxes(title="student's answers seen so far"); fig.update_yaxes(title='AUC'); L(fig, 360, title='AUC by number of prior answers')
        with c2:
            cat = KT['cat']['curves']
            fig = go.Figure([go.Scatter(y=cat['app_random'], name='our bank, random order', line=dict(color=MUTED, dash='dot')),
                             go.Scatter(y=cat['app_adaptive'], name='our bank, adaptive (IRT)', line_color=AMBER),
                             go.Scatter(y=cat['real_adaptive'][:10], name='1,500 real items, adaptive', line_color=TEAL)])
            fig.update_xaxes(title='questions asked', tickvals=list(range(10)), ticktext=[str(i + 1) for i in range(10)]); fig.update_yaxes(title='error in ability estimate (RMSE)')
            L(fig, 360, title='Adaptive testing: big gain with many difficulty levels, small with our 3')
        c1, c2 = st.columns(2)
        with c1:
            fc, ff = KT['curves']['forgetting'], KT['curves']['forgetting_fit']
            L(go.Figure(go.Bar(x=[f['gap'] for f in fc], y=[f['acc'] * 100 for f in fc], marker_color=AMBER, text=[f'{f["n"]:,}' for f in fc], textposition='outside')), 340,
              title=f'Forgetting: right last time, right again? (fitted half-life τ ≈ {ff["tau_days"]:.1f} days)')
        with c2:
            d = R['dkt']['curve']
            fig = go.Figure([go.Scatter(x=[c['epoch'] for c in d], y=[c['test_auc'] for c in d], name='test AUC', line_color=TEAL),
                             go.Scatter(x=[c['epoch'] for c in d], y=[c['train_loss'] for c in d], name='train loss', line_color=AMBER, yaxis='y2')])
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False)); fig.update_xaxes(title='epoch')
            L(fig, 340, title=f'Deep Knowledge Tracing (LSTM, {R["dkt"]["params"] / 1e6:.1f}M weights) training')
        b = R['bkt']['median']; i = R['irt']
        note(f'Numbers the skill test uses, all fitted on Riiid: guess {i["guess"]:.2f}, slip {i["slip"]:.2f}, discrimination {i["a_median"]:.2f}, '
             f'difficulty levels {", ".join(f"{x:+.2f}" for x in i["b_levels"])}; BKT prior {b["L0"]:.2f}, learn {b["T"]:.2f}, guess {b["G"]:.2f}, slip {b["S"]:.2f}; stop when uncertainty is below {assess.SD_STOP}.')
    if CF:
        st.markdown('### Coder models (behind the coding profile)')
        note(f'{CF["n_users"]:,} real Codeforces coders, sampled evenly across rating bands from {CF["n_active"]:,} active ones through the official API. '
             f'Task: predict rating from solved problems. Guessing the average gives an error of {CF["baseline_mae"]:.0f} points. Deployed: {CF["deployed"]}.')
        st.dataframe([{'model': m['model'], 'MAE': round(m['mae']), 'RMSE': round(m['rmse']), 'R²': round(m['r2'], 3), 'train MAE': round(m['train_mae']), '5-fold CV MAE': round(m['cv_mae'])}
                      for m in sorted(CF['models'], key=lambda m: m['mae'])], width='stretch', hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            sc = CF['scatter']; fig = go.Figure([go.Scatter(x=sc['actual'], y=sc['pred'], mode='markers', marker=dict(color=AMBER, size=5, opacity=.6), name='coders'),
                                                 go.Scatter(x=[800, 3500], y=[800, 3500], line=dict(color=MUTED, dash='dot'), name='perfect')])
            fig.update_xaxes(title='real rating'); fig.update_yaxes(title='predicted'); L(fig, 360, title='Predicted vs real rating (unseen coders)')
        with c2:
            k = CF['kmeans']; L(go.Figure(go.Scatter(x=k['k'], y=k['silhouette'], line_color=AMBER, mode='lines+markers')), 360, title=f'K-means coding styles: k = {k["best_k"]} by silhouette')
