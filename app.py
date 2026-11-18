# Skillfolio career assessment workspace. Run: .venv/bin/streamlit run app.py
import io, html, os, time
import numpy as np, streamlit as st, plotly.graph_objects as go
from pathlib import Path
from core import D, V, ROLES, N, IX, T, norm, probs, flags, extract, effort, evolve, path, audit
import deep
from ui import NAME, css, masthead, heading, hero, role_rankings, score_text, score_readout, chips, L, galaxy3d, PAL, INK, PAPER, AMBER, CORAL, TEAL, MUTED, LINE, WINE_WASH
import assess
from tabs_new import skill_test_tab, coding_tab, model_lab_extra

st.set_page_config(page_title=NAME, page_icon=str(Path(__file__).with_name("skillfolio.png")), layout="wide")

SAMPLE_RESUME = """Aarav Mehta — B.Tech Computer Science, 3rd year
Projects
Built a movie recommendation system in Python using pandas and scikit-learn, and served it with a Flask REST API.
Created a sales dashboard in Tableau on top of a PostgreSQL database with SQL queries for weekly reporting.
Trained a CNN image classifier in PyTorch for plant disease detection and reached 91% validation accuracy.
Experience
Summer intern at a retail company: automated Excel reports with Python and cleaned data for the analytics team.
Led a team of four in a college hackathon and presented the project to a panel of judges.
Skills
Python, SQL, Git, HTML, CSS, JavaScript, machine learning, data analysis, communication, teamwork."""
SAMPLE_JD = """We are hiring an engineer to build and deploy machine learning models for our recommendation platform.
You will design data pipelines in Python and Spark, train deep learning models in PyTorch or TensorFlow, and ship them to production with Docker and Kubernetes on AWS.
You should be comfortable with SQL, model evaluation, A/B testing and MLOps tools such as MLflow.
Strong communication and the ability to work with product teams is required."""

def found_skills(ents):
    """NER spans -> (indices of skills on our map, phrases that are not on the map)."""
    on, new = [], []
    for _, _, s in ents:
        n = norm(s)
        if n in IX:
            if IX[n] not in on: on.append(IX[n])
        elif n not in new: new.append(n)
    return on, new

def highlight(text, ents):
    out, pos = [], 0
    for a, b, s in ents:
        out += [html.escape(text[pos:a]), f'<mark class="{"on" if norm(s) in IX else "new"}">{html.escape(text[a:b])}</mark>']; pos = b
    return '<div class="doc">' + ''.join(out + [html.escape(text[pos:])]).replace('\n', '<br>') + '</div>'

def extras(text, on):
    """Safety net: skills from our 500-skill list that appear word-for-word but the tagger missed."""
    return [IX[s] for s in extract(text) if IX[s] not in on]

def read_skills(text):
    return found_skills(deep.ner(text))[0] if deep.READY else [IX[s] for s in extract(text)]

css()
masthead()
PRESETS = {'Student': ['python', 'sql', 'html', 'css', 'javascript', 'git', 'communication', 'java'],
           'Data': ['python', 'sql', 'excel', 'tableau', 'statistics', 'data analysis', 'pandas'],
           'Web': ['javascript', 'react', 'html', 'css', 'node.js', 'git', 'mongodb']}
S = st.session_state
if 'skills' not in S:
    S.skills = [s for s in PRESETS['Student'] if s in IX]
S.setdefault('target_role', 'ML Engineer')
S.setdefault('weight_tests', True)
S.setdefault('new_skills', 3)
S.setdefault('learning_effort', .2)

# Keep inputs when their inactive tabs are not rendered.
for key in ('skills', 'target_role', 'weight_tests', 'resume', 'jd', 'explore', 'test_skill',
            'sure', 'lc_user', 'cf_user', 'gh_user', 'map_view', 'new_skills', 'learning_effort', 'filter_jobs'):
    if key in S:
        S[key] = S[key]


def preset(name):
    S.skills = [s for s in PRESETS[name] if s in IX]


def use(idx):
    S.skills = [V[i] for i in idx]


def add_skills(names):
    S.skills = list(dict.fromkeys(S.skills + [s for s in names if s in IX]))


t1, ts, tc, tx, te, t5 = st.tabs(
    ['Role fit', 'Skill test', 'Coding profile', 'Resume', 'Explore', 'Model lab'],
    key='workspace', on_change='rerun',
)
t2 = t3 = tj = t4 = None
if te.open:
    with te:
        t2, t3, tj, t4 = st.tabs(
            ['Skill map', 'Learning path', 'Job description', 'Similar jobs'],
            key='explore', on_change='rerun',
        )

if t1.open:
    with t1:
        hero_slot = st.empty()
        profile, rankings = st.columns([4, 6], gap='medium')
        with profile.container(key='profile'):
            st.markdown('## Your profile')
            st.selectbox('Target role', ROLES, key='target_role')
            st.multiselect('Current skills', V, key='skills', placeholder='Search the skill library')
            st.toggle('Use skill test scores', key='weight_tests',
                      help='A tested skill contributes its estimated mastery. A skill at 40% mastery counts as 0.4. Untested skills count fully.')
            st.caption('Load a sample profile')
            for col, name in zip(st.columns(3), PRESETS):
                col.button(name, on_click=preset, args=(name,), width='stretch')

skills = S.skills
target_name = S.get('target_role', 'ML Engineer')
# Shared deployments always use isolated history unless local persistence is explicitly enabled.
history = None if os.getenv('SKILLFOLIO_LOCAL_HISTORY') == '1' else S.setdefault('assessment_history', {})
have = [IX[s] for s in skills]
target = ROLES.index(target_name)
vec = flags(have)
tested = {}
test_signals = []
if S.get('weight_tests', True):
    for skill in skills:
        mastery = assess.mastery_today(skill, history=history)
        if mastery:
            vec[IX[skill]] = mastery['today'] / 100
            tested[skill] = mastery['today']
            test_signals.append((skill, mastery['tested'], mastery['tests']))
profile_key = (tuple(have), target, tuple(test_signals))
p_now = probs(vec[None])[0]
w = D['coef'][target]
prof = D['role_profile'][target]
missing = [i for i in np.argsort(-w * (prof > 0.02)) if i not in have][:12]


def note(content):
    st.html(f'<div class="note">{content}</div>')


if ts.open:
    with ts:
        skill_test_tab(skills, add_skills, history)
if tc.open:
    with tc:
        coding_tab(V, add_skills)

# ---------------- 0. resume x-ray (two transformers) ----------------
if tx.open:
    with tx:
        heading('Resume analysis', 'Upload a PDF or paste your resume to see the skills and roles the text models identify.')
        up = st.file_uploader('Upload resume PDF', type='pdf')
        if up and S.get('seen') != up.file_id:
            from pypdf import PdfReader
            try:
                resume = '\n'.join(page.extract_text() or '' for page in PdfReader(io.BytesIO(up.getvalue())).pages).strip()
                if not resume:
                    st.warning('This PDF has no extractable text. Paste your resume below.')
                else:
                    S.resume = resume
                    got = read_skills(resume)
                    if got:
                        use(got)
                    st.toast(f'Found {len(got)} skills in your resume')
            except Exception:
                st.error('This PDF could not be read. Try another PDF or paste your resume below.')
            S.seen = up.file_id
        if not deep.READY: st.info('Deep models not found in models/. The other tabs still work.')
        else:
            st.text_area('Resume text', key='resume', height=200)
            c = st.columns([1, 1, 3])
            c[0].button('Load sample resume', on_click=lambda: S.update(resume=SAMPLE_RESUME), width='stretch')
            text = S.get('resume', '')
            if c[1].button('Analyze resume', type='primary', width='stretch') and text.strip():
                t0 = time.time(); ents = deep.ner(text)
                S.xray = dict(text=text, ents=ents, p=deep.role_probs([text])[0], imp=deep.sentence_impact(text, target), target=target, ms=(time.time() - t0) * 1000)
            x = S.get('xray')
            if x and x['text'] == text:
                on, new = found_skills(x['ents'])
                st.markdown(highlight(text, x['ents']), unsafe_allow_html=True)
                st.markdown(f'<small style="color:{MUTED}"><span style="color:{TEAL}">■</span> {len(on)} skills on the map &nbsp; <span style="color:{AMBER}">■</span> {len(new)} skills the model found that are outside our 500 &nbsp;·&nbsp; analysis took {x["ms"]:.0f} ms</small>', unsafe_allow_html=True)
                ex = extras(text, on)
                if ex: st.markdown(f'<small style="color:{MUTED}">a plain word search also matched these (not added automatically, add them in the profile box if they are real skills of yours):</small> {chips(ex)}', unsafe_allow_html=True)
                if len(on) >= 3: st.button(f'Use these {len(on)} skills as my profile', on_click=use, args=(on,))
                st.markdown('### Compare the two models')
                note('DistilBERT reads your resume text. Logistic regression uses only the skills the text model found.')
                ps = probs(flags(on)[None])[0] if on else np.zeros(len(ROLES)); o = np.argsort(x['p'])
                fig = go.Figure([go.Bar(name='skill-list model', y=[ROLES[i] for i in o], x=ps[o] * 100, orientation='h', marker_color=LINE),
                                 go.Bar(name='text model', y=[ROLES[i] for i in o], x=x['p'][o] * 100, orientation='h', marker_color=AMBER)])
                L(fig, 430, barmode='group')
                if x['imp'] and x['target'] == target:
                    st.markdown(f'### Sentences that affect your fit for {target_name}')
                    note('Occlusion test: we delete one sentence at a time and re-run the model. If the score drops, that sentence was helping.')
                    for s, d in [i for i in x['imp'][:3] if i[1] > 0.002] + [i for i in x['imp'][-2:] if i[1] < -0.002]:
                        st.markdown(f'<div class="card"><small style="color:{TEAL if d > 0 else CORAL}">{"+" if d > 0 else ""}{d * 100:.1f} points</small><br>{html.escape(s)}</div>', unsafe_allow_html=True)
                elif x['target'] != target: st.caption('Target role changed. Choose Analyze resume again to refresh the sentence analysis.')



# ---------------- 1. role fit ----------------
if t1.open:
    enough = len(have) >= 3
    hero(hero_slot, target_name, p_now[target] * 100 if enough else None, ROLES[int(np.argmax(p_now))])
    with profile:
        if tested:
            st.caption('Test scores used: ' + ', '.join(f'{s} {m:.0f}%' for s, m in tested.items()))
    with rankings.container(key='rankings'):
        st.markdown('## Role rankings')
        if enough:
            st.caption('Model probabilities across the roles in the training data.')
            role_rankings(ROLES, p_now, target)
        else:
            st.info('Add at least three skills or load a sample profile.')
    if enough:
        with t1:
            st.markdown('## Skills to build for ' + target_name)
            st.html(chips(missing, "miss"))
            st.button('Test a skill', on_click=lambda: S.update(workspace='Skill test'))
            with st.expander('How the role score is calculated'):
                note('Logistic regression learns a weight for every skill and role. This chart shows which of your current skills push your target role score up or down.')
                c1, c2 = st.columns(2)
                with c1:
                    o = sorted(have, key=lambda i: w[i])
                    L(go.Figure(go.Bar(x=[w[i] for i in o], y=[V[i] for i in o], orientation='h',
                                       marker_color=[TEAL if w[i] > 0 else CORAL for i in o])),
                      max(260, 26 * len(o)), title='Skill weights for your target role')
                with c2:
                    fam = sorted(range(len(D['cluster_names'])), key=lambda c: -prof[D['clusters'] == c].sum())[:6]
                    cov = [prof[[i for i in have if D['clusters'][i] == c]].sum() / max(prof[D['clusters'] == c].sum(), 1e-9) * 100 for c in fam]
                    names = [D['cluster_names'][c] for c in fam]
                    fig = go.Figure([go.Scatterpolar(r=[100] * 7, theta=names + names[:1], line_color=MUTED, name='role requirements'),
                                     go.Scatterpolar(r=cov + cov[:1], theta=names + names[:1], fill='toself', line_color=AMBER, name='your coverage')])
                    fig.update_polars(bgcolor='rgba(0,0,0,0)', radialaxis=dict(range=[0, 100], gridcolor=LINE, showticklabels=False), angularaxis=dict(gridcolor=LINE))
                    L(fig, 370, title='Coverage by skill family')

# ---------------- 2. galaxy ----------------
cl = D['clusters']; DOT = dict(size=4 + 2.2 * np.log1p(D['freq'] / D['freq'].min()), color=[PAL[c % len(PAL)] for c in cl], opacity=.75)
HOVER = [f'{V[i]}<br>{D["cluster_names"][cl[i]]}' for i in range(N)]
if t2 and t2.open:
    with t2:
        heading('Skill map', 'Explore 500 skills grouped by how often they appear together in job postings.')
        if len(have) < 3:
            st.info('Add at least three skills in Role fit to use this tool.')
        else:
            note('Every dot is a skill. Skills that appear together in job postings sit close together (co-occurrence → SVD embedding → t-SNE). Colours are K-means clusters. Highlighted skills show what you have and what your target role could use next.')
            add = S.ga[1] if S.get('ga') and S.ga[0][:3] == profile_key else []
            if st.radio('Map view', ['3D galaxy', 'Flat map'], horizontal=True, key='map_view') == '3D galaxy':
                galaxy3d(have, missing, add)
                if add: st.caption('Stars marked with a plus sign came from Learning path.')
            else:
                xy = D['xy']
                fig = go.Figure(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode='markers', text=HOVER, hoverinfo='text', marker=DOT, name='all skills'))
                for idx, col, sym, nm in ((missing, CORAL, 'diamond', 'missing for ' + target_name), (have, AMBER, 'star', 'your skills')):
                    fig.add_scatter(x=xy[idx, 0], y=xy[idx, 1], mode='markers+text', text=[V[i] for i in idx], textposition='top center', textfont=dict(color=col, size=11),
                                    marker=dict(size=15, color=col, symbol=sym, line=dict(color=INK, width=1)), name=nm, hoverinfo='text')
                fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
                L(fig, 640)


# ---------------- 3. what-if lab ----------------
if t3 and t3.open:
    with t3:
        heading('Learning path', 'Simulate which new skills could raise your fit for the role you chose.')
        if len(have) < 3:
            st.info('Add at least three skills in Role fit to use this tool.')
        elif len(have) == N:
            st.info('Your profile already includes every skill in the library.')
        else:
            note('The search balances fit for your target role with how close each new skill is to what you already know.')
            c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment='bottom')
            limit = min(6, N - len(have))
            S.new_skills = min(S.new_skills, limit)
            if limit == 1:
                c1.caption('One skill is available to add.')
                k = 1
            else:
                k = c1.slider('New skills to learn', 1, limit, key='new_skills')
            lam = c2.slider('Preference for easier skills', 0.0, 0.6, step=0.05, key='learning_effort')
            key = (*profile_key, k, lam)
            if c3.button('Simulate learning path', type='primary', width='stretch'):
                S.ga = (key, *evolve(have, target, k, lam, profile=vec), vec)
            if S.get('ga', [None])[0] == key:
                _, bestset, hist, bests, baseline = S.ga
                order = path(have, bestset)
                steps = []
                simulated = baseline.copy()
                for i in order:
                    simulated[i] = 1
                    steps.append(probs(simulated[None])[0, target] * 100)
                c1, c2 = st.columns([2, 3])
                with c1:
                    score_readout(steps[-1], f'{target_name} after {k} skills',
                                  before=probs(baseline[None])[0, target] * 100, color=TEAL)
                with c2:
                    fig = go.Figure(go.Scatter(y=hist, mode='lines', line=dict(color=AMBER, width=2), fill='tozeroy', fillcolor=WINE_WASH))
                    fig.update_xaxes(title='generation'); fig.update_yaxes(title='best fitness')
                    L(fig, 250, title='Evolution: best fitness in each generation')
                st.markdown('### Your learning path')
                eff = effort(have)
                for n, (i, s) in enumerate(zip(order, steps), 1):
                    near = V[have[int(np.argmax(D['emb'][have] @ D['emb'][i]))]]
                    st.markdown(f'<div class="step" ><i>{n:02d}</i><div><b>{V[i]}</b><br><small style="color:{MUTED}">closest to your <b>{near}</b> · effort {eff[i]:.2f}</small></div><span>→ {s:.0f}%</span></div>', unsafe_allow_html=True)
                st.markdown('### Replay the search')
                note('Press play. The diamond markers show the best skill set in each generation.')
                xy = D['xy']; tr = lambda g: go.Scatter(x=xy[bests[g], 0], y=xy[bests[g], 1], mode='markers+text', text=[V[i] for i in bests[g]], textposition='top center',
                                                         textfont=dict(color=TEAL, size=12), marker=dict(size=18, color=TEAL, symbol='diamond', line=dict(color=INK, width=1)), name='best set')
                fig = go.Figure([go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers', marker=dict(DOT, opacity=.45), hoverinfo='skip', name='all skills'),
                                 go.Scatter(x=xy[have, 0], y=xy[have, 1], mode='markers', marker=dict(size=12, color=AMBER, symbol='star'), name='your skills'), tr(0)],
                                frames=[go.Frame(data=[tr(g)], traces=[2], name=str(g + 1)) for g in range(len(bests))])
                fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
                fig.update_layout(updatemenus=[dict(type='buttons', x=0, y=1.08, showactive=False, bgcolor=LINE, font=dict(color=PAPER),
                                                    buttons=[dict(label='▶ play', method='animate', args=[None, dict(frame=dict(duration=160, redraw=False), transition=dict(duration=120), fromcurrent=True)])])],
                                  sliders=[dict(currentvalue=dict(prefix='generation '), pad=dict(t=10), steps=[dict(method='animate', label=str(g + 1), args=[[str(g + 1)], dict(mode='immediate', frame=dict(duration=0, redraw=False))]) for g in range(len(bests))])])
                L(fig, 560)
            else:
                st.caption(f'Current fit for {target_name}: {score_text(p_now[target] * 100)}')


# ---------------- 4. JD match ----------------
if tj and tj.open:
    with tj:
        heading('Compare a job description', 'See which skills a specific posting asks for and how many your profile covers.')
        if not deep.READY: st.info('Deep models not found in models/.')
        else:
            note('Paste any job description. The skill model pulls out what it asks for, the text model says what kind of role it really is, and we compare both with your profile.')
            st.text_area('Job description', key='jd', height=170)
            c = st.columns([1, 1, 3])
            c[0].button('Load sample posting', on_click=lambda: S.update(jd=SAMPLE_JD), width='stretch')
            jd = S.get('jd', '')
            if c[1].button('Compare skills', type='primary', width='stretch') and jd.strip():
                S.jdres = dict(text=jd, ents=deep.ner(jd), p=deep.role_probs([jd])[0])
            r = S.get('jdres')
            if r and r['text'] == jd:
                on, new = found_skills(r['ents']); got = [i for i in on if i in have]; need = [i for i in on if i not in have]; jr = int(np.argmax(r['p']))
                c1, c2 = st.columns([2, 3])
                with c1: score_readout(100 * len(got) / max(len(on), 1), 'of this JD covered by you', color=CORAL if len(got) < len(on) / 2 else TEAL)
                with c2:
                    st.markdown(f'<div class="card"><small>the text model reads this posting as</small><br><b>{ROLES[jr]}</b> <small>· {r["p"][jr]*100:.0f}% sure</small></div>'
                                f'<div class="card"><small>this JD (micro view)</small><br><b>{len(got)} of {len(on)} skills</b><br><small>the {ROLES[jr]} market in general (macro view)</small><br><b>{p_now[jr]*100:.0f}% match</b></div>', unsafe_allow_html=True)
                st.markdown(highlight(jd, r['ents']), unsafe_allow_html=True)
                st.markdown(f'**You have**<br>{chips(got, "have") or "nothing yet"}<br><br>**You need**<br>{chips(need, "miss") or "nothing, apply"}' +
                            (f'<br><br>**Also asked for (not on our map)**<br>' + ''.join(f'<span class="chip new">{html.escape(s)}</span>' for s in new[:20]) if new else ''), unsafe_allow_html=True)


# ---------------- 5. job twins ----------------
if t4 and t4.open:
    with t4:
        heading('Similar jobs', 'Browse real postings with skill lists close to your own.')
        note('K-nearest neighbours with cosine similarity: the real postings whose skill lists look most like yours.')
        only = st.toggle(f'Only {target_name} postings', False, key='filter_jobs')
        if not have:
            st.info('Add skills in Role fit to find comparable postings.')
        else:
            u = flags(have)
            tw = D['twins']
            sim = T @ u / (np.linalg.norm(T, axis=1) * np.linalg.norm(u) + 1e-9)
            ranked = np.argsort(-sim)
            if only:
                ranked = ranked[np.asarray(tw['role'])[ranked] == target_name]
            if not len(ranked):
                st.info('There are no matching postings for this role in the dataset.')
            for r in ranked[:6]:
                need = [i for i in tw['ix'][r] if i not in have][:8]
                got = [i for i in tw['ix'][r] if i in have]
                st.html(f'<div class="card"><b>{html.escape(tw["title"][r])}</b> <small>· {html.escape(tw["company"][r])} · {tw["role"][r]} · similarity {sim[r]:.2f}</small><br>{chips(got, "have")}{chips(need, "miss")}</div>')



# ---------------- 6. model lab ----------------
if t5.open:
    with t5:
        heading('Model lab', 'Inspect the training results and rerun the role model on unseen job postings.')
        f, M = D['final'], sorted(D['models'], key=lambda m: -m['accuracy'])
        st.markdown('### Live model audit')
        note('Press a button and the app recomputes accuracy right now on held-out data the model never trained on. The control scrambles which column means which skill: a real model collapses to near guessing, a hard-coded one would not change.')
        c1, c2 = st.columns(2)
        if c1.button('Re-test the skill model on held-out postings', width='stretch'): S.aud = audit()
        if 'aud' in S:
            a_ = S.aud; m = c1.columns(3)
            m[0].metric(f'accuracy on {a_["n"]:,} unseen', f'{a_["accuracy"]*100:.1f}%'); m[1].metric('skills scrambled', f'{a_["scrambled"]*100:.1f}%'); m[2].metric('pure guessing', f'{a_["chance"]*100:.1f}%')
        if deep.READY:
            R_ = deep.META['role']
            if c2.button('Re-test the text model on 60 unseen job descriptions', width='stretch'):
                with st.spinner('Running DistilBERT…'):
                    t0 = time.time(); pr = deep.role_probs(R_['audit']['text'][:60]); S.aud2 = (float((pr.argmax(1) == np.array(R_['audit']['y'][:60])).mean()), (time.time() - t0) / 60 * 1000)
            if 'aud2' in S:
                m = c2.columns(2); m[0].metric('accuracy on 60 unseen', f'{S.aud2[0]*100:.0f}%'); m[1].metric('time per document', f'{S.aud2[1]:.0f} ms')

        st.markdown('### Skill-list model (classical ML)')
        c = st.columns(4)
        for col, (lab, val) in zip(c, [('test accuracy', f['accuracy']), ('top-3 accuracy', f['top3_accuracy']), ('ROC-AUC', f['roc_auc']), ('5-fold CV', f['cv_mean'])]): col.metric(lab, f'{val*100:.1f}%')
        note(f'{D["n_postings"]:,} real postings, {len(ROLES)} roles, 80/20 split, duplicates removed first. Role labels come from job titles, so some overlap between roles (e.g. Software Engineer vs Full Stack) is expected and shows up in the confusion matrix. Deployed model: logistic regression. Random Forest scores about 3 points higher but memorises its training data; logistic regression is small and every weight can be shown.')
        st.dataframe([{'model': m['model'], 'test acc %': round(m['accuracy'] * 100, 1), 'train acc %': round(m['train_accuracy'] * 100, 1), 'macro F1 %': round(m['f1'] * 100, 1),
                       'ROC-AUC': None if m['roc_auc'] is None else round(m['roc_auc'], 3), 'fit seconds': round(m['fit_seconds'], 1)} for m in M], width='stretch', hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure([go.Bar(name='train', x=[m['model'].split(' (')[0] for m in M], y=[m['train_accuracy'] * 100 for m in M], marker_color=LINE),
                             go.Bar(name='test', x=[m['model'].split(' (')[0] for m in M], y=[m['accuracy'] * 100 for m in M], marker_color=AMBER)])
            L(fig, 360, title='Overfitting check: train vs test accuracy', barmode='group')
            vc = D['validation_curve']
            fig = go.Figure([go.Scatter(x=[v['C'] for v in vc], y=[v['train'] * 100 for v in vc], name='train', line_color=MUTED),
                             go.Scatter(x=[v['C'] for v in vc], y=[v['val'] * 100 for v in vc], name='validation', line_color=AMBER)])
            fig.update_xaxes(type='log', title='C (less regularisation →)')
            L(fig, 320, title='Bias–variance: tuning regularisation')
        with c2:
            cm = np.array(f['confusion']); cm = cm / cm.sum(1, keepdims=True)
            L(go.Figure(go.Heatmap(z=cm, x=ROLES, y=ROLES, colorscale=[[0, INK], [1, AMBER]], showscale=False, text=np.round(cm * 100).astype(int), texttemplate='%{text}', textfont_size=9)), 360, title='Confusion matrix (% of each true role)')
            km = D['kmeans']
            fig = go.Figure([go.Scatter(x=km['k'], y=km['silhouette'], name='silhouette', line_color=AMBER),
                             go.Scatter(x=km['k'], y=np.array(km['inertia']) / max(km['inertia']), name='inertia (elbow, scaled)', line_color=MUTED, yaxis='y2')])
            fig.add_vline(x=km['best_k'], line_dash='dot', line_color=TEAL)
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False))
            L(fig, 320, title=f'K-means: choosing k = {km["best_k"]}')
        cal = D['calibration']
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash='dot'), name='perfect'),
                         go.Scatter(x=[c['confidence'] for c in cal], y=[c['accuracy'] for c in cal], mode='lines+markers', line_color=AMBER, name='our model')])
        fig.update_xaxes(title='model says'); fig.update_yaxes(title='actually right')
        L(fig, 320, title='Can the percentages be trusted? (calibration)')

        if deep.READY:
            K = deep.META['ner']; b_, g_, q_ = R_['baseline'], R_['distilbert'], R_['quantised']
            st.markdown('### Text models (deep learning)')
            note(f'DistilBERT (66 million weights) fine-tuned on Kaggle GPU with {R_["n_train"]:,} real job descriptions. The job title was deleted from every description first, so the model cannot cheat by reading it. Then both models were compressed to 8-bit numbers so they run on a laptop without a GPU.')
            st.dataframe([{'model': 'TF-IDF + logistic regression (baseline)', 'test acc %': round(b_['accuracy'] * 100, 1), 'macro F1 %': round(b_['f1'] * 100, 1), 'top-3 %': round(b_['top3'] * 100, 1), 'size MB': None},
                          {'model': 'DistilBERT fine-tuned (GPU, 32-bit)', 'test acc %': round(g_['accuracy'] * 100, 1), 'macro F1 %': round(g_['f1'] * 100, 1), 'top-3 %': round(g_['top3'] * 100, 1), 'size MB': round(q_['mb_before'])},
                          {'model': 'DistilBERT 8-bit (laptop), 800 test docs', 'test acc %': round(q_['accuracy'] * 100, 1), 'macro F1 %': None, 'top-3 %': None, 'size MB': round(q_['mb_after'])}], width='stretch', hide_index=True)
            c = st.columns(4); nd, nq = K['distilbert'], K['quantised']
            for col, (lab, val) in zip(c, [('skill tagger precision', nd['precision']), ('skill tagger recall', nd['recall']), ('skill tagger F1', nd['f1']), ('F1 after 8-bit', nq['f1'])]): col.metric(lab, f'{val*100:.1f}%')
            fig = go.Figure([go.Scatter(y=R_['loss_curve'], x=np.arange(1, len(R_['loss_curve']) + 1) * 50, name='role model', line_color=AMBER),
                             go.Scatter(y=K['loss_curve'], x=np.arange(1, len(K['loss_curve']) + 1) * 50, name='skill tagger', line_color=TEAL)])
            fig.update_xaxes(title='training step (mini-batch gradient descent)'); fig.update_yaxes(title='loss')
            L(fig, 320, title='Training loss going down during fine-tuning')
        model_lab_extra()
