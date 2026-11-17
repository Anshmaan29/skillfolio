"""Skillfolio theme, report components and charts."""
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import D, V, N

NAME = "Skillfolio"
INK, PAPER, AMBER, CORAL, TEAL, MUTED, LINE = (
    "#f6f5ef", "#1d2d28", "#285649", "#934537", "#39765f", "#53645b", "#d9dfd5"
)
FONTS = "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Hedvig+Letters+Serif:opsz@12..24&display=swap"
GSAP = "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"
MARK = '<span class="folio-mark" aria-hidden="true"><i></i><i></i><i></i></span>'


def css():
    stylesheet = Path(__file__).with_name('theme.css').read_text()
    st.html(f'<link href="{FONTS}" rel="stylesheet"><style>{stylesheet}</style>')


def masthead():
    st.html(f'<header class="folio-brand"><span class="folio-wordmark">{MARK}{NAME}</span>'
            '<small>Career assessment workspace</small></header>')


def heading(title, description):
    st.html(f'<div class="folio-heading"><h1>{html.escape(title)}</h1>'
            f'<p>{html.escape(description)}</p></div>')


def score_text(pct):
    return '<1%' if 0 < pct < 1 else f'{pct:.0f}%'


def score_bar(pct):
    pct = min(100, max(0, pct))
    return f'<span class="folio-track" aria-hidden="true"><span class="folio-fill" style="width:{pct:.2f}%"></span></span>'


def hero(slot, role, score, best_role):
    snapshot = '<span>Add at least three skills to see your role fit.</span>'
    if score is not None:
        snapshot = (f'<span>Target role</span><div class="folio-score"><strong>{html.escape(role)}</strong>'
                    f'<b>{html.escape(score_text(score))}</b></div>{score_bar(score)}'
                    f'<small>Closest match: {html.escape(best_role)}</small>')
    slot.html('<section class="folio-hero" aria-labelledby="role-fit-title"><div>'
              '<h1 id="role-fit-title">Where do your skills fit?</h1>'
              '<p>Compare your profile with tech roles and see which skills could improve your fit.</p>'
              f'<p class="folio-source">Based on <strong>{D["n_postings"]:,} job postings</strong> and <strong>{N} mapped skills</strong></p>'
              f'</div><div class="folio-snapshot">{snapshot}</div></section>')


def role_rankings(roles, probabilities, target):
    ranked = list(np.argsort(-probabilities))
    shown = set(ranked[:4]) | {target}

    def row(i):
        pct = probabilities[i] * 100
        selected = ' aria-current="true"' if i == target else ''
        label = '<small>Target role</small>' if i == target else ''
        return (f'<li{selected}><span>{html.escape(roles[i])}{label}</span>{score_bar(pct)}'
                f'<data value="{pct:.2f}">{html.escape(score_text(pct))}</data></li>')

    primary = ''.join(row(i) for i in ranked if i in shown)
    rest = ''.join(row(i) for i in ranked if i not in shown)
    st.html(f'<ol class="folio-ranks">{primary}</ol><details class="folio-rank-more">'
            f'<summary>Show other roles</summary><ol class="folio-ranks">{rest}</ol></details>')


def L(fig, h=380, **kw):
    fig.update_layout(height=h, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=12, r=18, t=66 if 'title' in kw else 40, b=18),
                      font=dict(family='DM Sans, system-ui, sans-serif', color=PAPER, size=12),
                      legend=dict(orientation='h', y=1.03, yanchor='bottom'),
                      xaxis=dict(gridcolor=LINE, zeroline=False, automargin=True),
                      yaxis=dict(gridcolor=LINE, zeroline=False, automargin=True), **kw)
    st.plotly_chart(fig, theme=None, width='stretch', config=dict(displayModeBar=False))


def score_readout(pct, label, before=None, color=AMBER):
    previous = f'<small>Previous score {before:.0f}%</small>' if before is not None else ''
    st.html(f'<div class="folio-measure" style="--score-accent:{color}"><b>{html.escape(score_text(pct))}</b>'
            f'<p>{html.escape(label)}</p>{score_bar(pct)}{previous}</div>')


def chips(idx, cls=''):
    return ''.join(f'<span class="chip {cls}">{html.escape(V[i])}</span>' for i in idx)


# ---------------- 3D skill galaxy (three.js) ----------------
PAL = ['#5b8def', '#57c7c0', '#a78bfa', '#6fcf97', '#e0b25c', '#e58a8a', '#6cb6d9', '#9aa0d6', '#4fb8a8', '#d1a86a', '#7fa6e6', '#b08fd8', '#74c9a0', '#d98fa6']
_GALAXY = r"""<link href="__FONTS__" rel="stylesheet">
<style>body{margin:0;background:transparent;overflow:hidden;font-family:'DM Sans',sans-serif}
#w{position:relative;height:__H__px;border:1px solid __LINE__;border-radius:8px;overflow:hidden;background:radial-gradient(ellipse at 50% 42%,#141c33 0%,__INK__ 72%);cursor:grab}
#w:active{cursor:grabbing} canvas{display:block}
.lb{position:absolute;left:0;top:0;font-size:11px;pointer-events:none;white-space:nowrap;will-change:transform;text-shadow:0 0 6px __INK__,0 0 3px __INK__,0 0 2px __INK__}
.fam{font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:__MUTED__;opacity:.5}
#tip{position:absolute;left:0;top:0;padding:6px 10px;background:#0f1524ee;border:1px solid __LINE__;border-radius:4px;font-size:11px;color:__PAPER__;pointer-events:none;opacity:0;transition:opacity .15s;z-index:5}
#tip small{color:__MUTED__;display:block;font-size:9.5px}
#lg{position:absolute;left:16px;bottom:14px;font-size:10.5px;color:__MUTED__;line-height:1.9} #lg i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}
#hint{position:absolute;right:16px;bottom:14px;font-size:10px;color:__MUTED__;letter-spacing:.08em;text-transform:uppercase;opacity:.7}
#err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:__MUTED__;font-size:12px}</style>
<div id="w"><div id="tip"></div>
<div id="lg"><div><i style="background:__AMBER__"></i>your skills</div><div><i style="background:__CORAL__"></i>the role wants these next</div><div id="lga" style="display:none"><i style="background:__TEAL__"></i>what evolution picked</div></div>
<div id="hint">drag to rotate · scroll to zoom · hover a star</div><div id="err">3D view needs an internet connection (three.js). Use the flat map.</div></div>
<script src="__GSAP__"></script>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
let THREE, OrbitControls;
try { THREE = await import('three'); ({OrbitControls} = await import('three/addons/controls/OrbitControls.js')); }
catch (e) { document.getElementById('err').style.display = 'flex'; throw e; }
const DT = __DATA__, w = document.getElementById('w'), tip = document.getElementById('tip');
const still = matchMedia('(prefers-reduced-motion:reduce)').matches || !window.gsap;
let W = w.clientWidth, H = w.clientHeight;
const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(50, W / H, 1, 2000);
let rnd;
try { rnd = new THREE.WebGLRenderer({antialias: true, alpha: true}); }
catch { const error = document.getElementById('err'); error.textContent = '3D is unavailable in this browser. Switch to the Flat map.'; error.style.display = 'flex'; throw new Error('WebGL unavailable'); }
rnd.setPixelRatio(Math.min(devicePixelRatio, 2)); rnd.setSize(W, H); w.prepend(rnd.domElement);
const ctl = new OrbitControls(cam, rnd.domElement); ctl.enableDamping = true; ctl.dampingFactor = .06; ctl.enablePan = false; ctl.minDistance = 35; ctl.maxDistance = 260;
ctl.autoRotate = !still; ctl.autoRotateSpeed = .45;
const V3 = i => new THREE.Vector3(...DT.P[i]);
function cloud(pos, col, size, opacity) {            // soft glowing round sprites, additive so dense regions shine
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3)); g.setAttribute('size', new THREE.Float32BufferAttribute(size, 1));
  const m = new THREE.ShaderMaterial({uniforms: {uScale: {value: 1}, uOp: {value: opacity}, uPx: {value: rnd.getPixelRatio()}}, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `attribute float size; uniform float uScale, uPx; varying vec3 vC; void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_PointSize = size * uScale * uPx * (150.0 / -mv.z); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform float uOp; varying vec3 vC; void main(){ float d = length(gl_PointCoord - .5); float a = pow(smoothstep(.5, .0, d), 2.2) * .85 + smoothstep(.13, .0, d); gl_FragColor = vec4(vC, a * uOp); }`});
  const p = new THREE.Points(g, m); scene.add(p); return p;
}
const C = h => { const c = new THREE.Color(h); return [c.r, c.g, c.b]; };
const base = cloud(DT.P.flat(), DT.c.flatMap(C), DT.s, .9);
const dust = []; for (let i = 0; i < 1400; i++) { const v = new THREE.Vector3().randomDirection().multiplyScalar(170 + Math.random() * 380); dust.push(v.x, v.y, v.z); }
cloud(dust, Array(1400).fill([.55, .6, .75]).flat(), Array.from({length: 1400}, () => 1.5 + Math.random() * 3), .5);
const mark = (idx, hex, sz) => idx.length ? cloud(idx.flatMap(i => DT.P[i]), idx.flatMap(() => C(hex)), idx.map(() => sz), 1) : null;
const mHave = mark(DT.have, '__AMBER__', 26), mMiss = mark(DT.miss, '__CORAL__', 20), mAdd = mark(DT.add, '__TEAL__', 30);
function lines(pairs, hex, op) {
  if (!pairs.length) return null;
  const g = new THREE.BufferGeometry().setFromPoints(pairs.flatMap(([a, b]) => [V3(a), V3(b)]));
  const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({color: hex, transparent: true, opacity: op, blending: THREE.AdditiveBlending, depthWrite: false})); scene.add(l); return l;
}
const dist = (a, b) => V3(a).distanceTo(V3(b));
const mst = []; { const inn = DT.have.slice(0, 1), out = DT.have.slice(1);          // constellation: minimum spanning tree over your skills
  while (out.length) { let best = [1e9]; for (const a of inn) for (const b of out) { const d = dist(a, b); if (d < best[0]) best = [d, a, b]; } mst.push([best[1], best[2]]); inn.push(best[2]); out.splice(out.indexOf(best[2]), 1); } }
const near = i => DT.have.reduce((m, h) => dist(i, h) < dist(i, m) ? h : m, DT.have[0]);
const lHave = lines(mst, '__AMBER__', .55), lMiss = lines(DT.miss.map(i => [i, near(i)]), '__CORAL__', .16), lAdd = lines(DT.add.map(i => [i, near(i)]), '__TEAL__', .7);
if (DT.add.length) document.getElementById('lga').style.display = 'block';
const labels = [];
const label = (pos, text, cls, color) => { const e = document.createElement('div'); e.className = 'lb ' + cls; e.textContent = text; if (color) e.style.color = color; w.appendChild(e); labels.push({e, pos}); };
DT.fn.forEach((n, c) => { const v = new THREE.Vector3(); let k = 0; DT.f.forEach((f, i) => { if (f === c) { v.add(V3(i)); k++; } }); if (k) label(v.divideScalar(k).multiplyScalar(1.28), n, 'fam'); });
DT.miss.slice(0, 6).forEach(i => label(V3(i), DT.n[i], '', '__CORAL__')); DT.have.forEach(i => label(V3(i), DT.n[i], '', '__AMBER__')); DT.add.forEach(i => label(V3(i), '+ ' + DT.n[i], '', '__TEAL__'));
const ray = new THREE.Raycaster(); ray.params.Points.threshold = 1.6; const mouse = new THREE.Vector2(); let hover = false;
w.addEventListener('pointermove', ev => { const r = w.getBoundingClientRect(); mouse.set((ev.clientX - r.left) / W * 2 - 1, -(ev.clientY - r.top) / H * 2 + 1); ray.setFromCamera(mouse, cam);
  const hit = ray.intersectObject(base)[0];
  if (hit) { const i = hit.index; tip.innerHTML = DT.n[i] + '<small>' + DT.fn[DT.f[i]] + '</small>'; tip.style.transform = `translate(${ev.clientX - r.left + 14}px,${ev.clientY - r.top + 12}px)`; tip.style.opacity = 1; hover = true; }
  else { tip.style.opacity = 0; hover = false; } });
w.addEventListener('pointerleave', () => { tip.style.opacity = 0; hover = false; });
const end = new THREE.Vector3(78, 38, 96);
if (still) cam.position.copy(end); else {
  cam.position.set(0, 60, 330); base.material.uniforms.uScale.value = 0;
  [lHave, lMiss, lAdd].forEach(l => l && gsap.from(l.material, {opacity: 0, duration: 1.4, delay: 1.5, ease: 'power2.out'}));
  [mHave, mMiss, mAdd].forEach((m, k) => m && gsap.from(m.material.uniforms.uScale, {value: 0, duration: 1.1, delay: 1.1 + k * .25, ease: 'back.out(2.2)'}));
  gsap.to(cam.position, {x: end.x, y: end.y, z: end.z, duration: 2.6, ease: 'power3.out'});
  gsap.to(base.material.uniforms.uScale, {value: 1, duration: 1.8, ease: 'power2.out'});
  gsap.from('.lb', {opacity: 0, duration: .8, delay: 1.9, stagger: .02});
}
const v = new THREE.Vector3();
(function tick() { requestAnimationFrame(tick); ctl.autoRotate = !still && !hover; ctl.update(); rnd.render(scene, cam);
  for (const {e, pos} of labels) { v.copy(pos).project(cam); const fam = e.classList.contains('fam');
    e.style.display = v.z < 1 ? 'block' : 'none'; e.style.transform = `translate(${(v.x * .5 + .5) * W}px,${(-v.y * .5 + .5) * H}px) translate(-50%,${fam ? '-50%' : '-170%'})`; } })();
addEventListener('resize', () => { W = w.clientWidth; H = w.clientHeight; cam.aspect = W / H; cam.updateProjectionMatrix(); rnd.setSize(W, H); });
</script>"""

def galaxy3d(have, missing, add=(), h=700):
    """Interactive WebGL star map of all skills. Positions come from the trained embedding (t-SNE to 3D)."""
    data = dict(P=np.round(D['xyz'] * 28, 2).tolist(), c=[PAL[c % len(PAL)] for c in D['clusters']],
                s=np.round(7 + 5.5 * np.log1p(D['freq'] / D['freq'].min()), 1).tolist(), n=V, f=D['clusters'].tolist(), fn=D['cluster_names'],
                have=[int(i) for i in have], miss=[int(i) for i in missing if i not in add], add=[int(i) for i in add])
    page = _GALAXY.replace('__DATA__', json.dumps(data)).replace('__H__', str(h))
    for k, val in dict(FONTS=FONTS, GSAP=GSAP, INK='#132722', PAPER='#f9f8ef', AMBER='#efc979', CORAL='#f1a092', TEAL='#8cd8ba', MUTED='#c3d3c9', LINE='#456359').items(): page = page.replace(f'__{k}__', val)
    st.iframe(page, height=h + 4, alt='Interactive 3D map of related job skills')
