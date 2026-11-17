# CareerTwin deep models (Kaggle GPU). Fine-tunes DistilBERT twice on real LinkedIn job descriptions:
#   1. role model : job-description text -> one of 13 roles          (sequence classification)
#   2. skill model: tags the words in any text that are skills        (token classification / NER)
# Then shrinks both with int8 quantisation (ONNX) so they run on a laptop CPU.
import pandas as pd, numpy as np, re, glob, json, time, os, random, subprocess, sys, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification,
                          get_linear_schedule_with_warmup)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime', 'onnxscript'])
RNG = 42; torch.manual_seed(RNG); np.random.seed(RNG); random.seed(RNG)
dev = 'cuda'; BASE = 'distilbert-base-uncased'
print(torch.__version__, torch.cuda.get_device_name(0))
find = lambda n: glob.glob(f'/kaggle/input/**/{n}', recursive=True)[0]

# ---------- data: same role labelling as the classical script ----------
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
t = posts.job_title.fillna('').str.lower(); posts['role'] = None
for name, rx in ROLES:
    posts.loc[posts.role.isna() & t.str.contains(rx, regex=True), 'role'] = name
posts = posts.dropna(subset=['role'])
df = posts.merge(pd.read_csv(find('job_skills.csv')), on='job_link').dropna(subset=['job_skills'])
df = df.drop_duplicates(['job_title', 'company', 'job_skills'])
df = pd.concat([g.sample(min(len(g), 6000), random_state=RNG) for _, g in df.groupby('role')])
keep = set(df.job_link)
summ = pd.concat(c[c.job_link.isin(keep)] for c in pd.read_csv(find('job_summary.csv'), chunksize=200000))
df = df.merge(summ, on='job_link').dropna(subset=['job_summary']).drop_duplicates('job_summary').reset_index(drop=True)
roles = sorted(df.role.unique()); df['y'] = df.role.map({r: i for i, r in enumerate(roles)})
print('postings with text:', len(df), df.role.value_counts().to_dict())

# Label leakage guard: delete the job title from the description, so the model must read the actual content.
TITLE = re.compile(r'\b(?:machine learning|ml|ai|data|devops|cloud|cyber ?security|security|full[- ]?stack|front[- ]?end|back[- ]?end|mobile|android|ios|qa|test|ux|ui|ui/ux|embedded|firmware|software|site reliability|platform|bi|business intelligence|product|web|java|python|\.net|automation)\s+(?:engineer|scientist|analyst|developer|architect|designer|tester)s?\b', re.I)
def mask(row):
    s = re.sub(re.escape(str(row.job_title)), ' ', row.job_summary, flags=re.I)
    return re.sub(r'\s+', ' ', TITLE.sub(' ', s)).strip()
df['text'] = df.apply(mask, axis=1)
tok = AutoTokenizer.from_pretrained(BASE)

def fit(model, ds, epochs, lr=3e-5, bs=32):
    model.to(dev).train(); dl = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl) * epochs), len(dl) * epochs)
    scaler = torch.amp.GradScaler('cuda'); curve, run = [], 0.0
    for ep in range(epochs):
        for i, (ids, att, lab) in enumerate(dl):
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            run += loss.item()
            if (i + 1) % 50 == 0: curve.append(run / 50); run = 0.0
        print('epoch', ep + 1, 'loss', curve[-1] if curve else None, flush=True)
    return curve

@torch.no_grad()
def logits(model, ids, att, bs=128):
    model.to(dev).eval(); out = []
    for i in range(0, len(ids), bs):
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(model(input_ids=ids[i:i+bs].to(dev), attention_mask=att[i:i+bs].to(dev)).logits.float().cpu())
    return torch.cat(out).numpy()

def log(*a):
    print(*a, flush=True); open('progress.txt', 'a').write(' '.join(map(str, a)) + '\n')

def export(model, name, ids, att):
    """PyTorch -> ONNX -> int8. Returns ([MB before, MB after], onnxruntime session) or (None, None)."""
    import onnxruntime as ort, contextlib, io
    from onnxruntime.quantization import quantize_dynamic, QuantType
    model.cpu().eval().float()
    class W(torch.nn.Module):
        def __init__(s): super().__init__(); s.m = model
        def forward(s, a, b): return s.m(input_ids=a, attention_mask=b).logits
    kw = dict(input_names=['input_ids', 'attention_mask'], output_names=['logits'], opset_version=17,
              dynamic_axes={'input_ids': {0: 'b', 1: 't'}, 'attention_mask': {0: 'b', 1: 't'}, 'logits': {0: 'b', 1: 't'} if name == 'ner' else {0: 'b'}})
    size = lambda f: sum(os.path.getsize(x) for x in glob.glob(f + '*')) / 1e6
    ok = False
    for dyn in (False, True):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                torch.onnx.export(W(), (ids[:2], att[:2]), f'{name}.fp32.onnx', dynamo=dyn, **kw)
            ok = True; log(name, 'onnx export ok, dynamo =', dyn); break
        except Exception as e:
            log(name, 'export failed, dynamo =', dyn, type(e).__name__, str(e)[:300].replace('\n', ' '))
    if not ok:
        model.half().save_pretrained(f'{name}_pt'); return None, None
    mb = [size(f'{name}.fp32.onnx')]
    try:
        quantize_dynamic(f'{name}.fp32.onnx', f'{name}.onnx', weight_type=QuantType.QInt8)
        for f in glob.glob(f'{name}.fp32.onnx*'): os.remove(f)
        mb.append(size(f'{name}.onnx'))
    except Exception as e:
        log(name, 'quantise failed', type(e).__name__, str(e)[:300].replace('\n', ' '))
        os.replace(f'{name}.fp32.onnx', f'{name}.onnx'); mb.append(mb[0])
    return mb, ort.InferenceSession(f'{name}.onnx', providers=['CPUExecutionProvider'])

def ort_logits(sess, ids, att, bs=32):
    return np.concatenate([sess.run(None, {'input_ids': ids[i:i+bs].numpy(), 'attention_mask': att[i:i+bs].numpy()})[0] for i in range(0, len(ids), bs)])

OUT = dict(roles=roles, base=BASE)

# ================= 1. ROLE MODEL =================
L1 = 320
tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=RNG)
enc = lambda s: tok(list(s), truncation=True, max_length=L1, padding='max_length', return_tensors='pt')
a, b = enc(tr.text), enc(te.text)
t0 = time.time()
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
base = LogisticRegression(C=5, max_iter=300).fit(tfidf.fit_transform(tr.text), tr.y)
bp = base.predict_proba(tfidf.transform(te.text))
baseline = dict(accuracy=accuracy_score(te.y, bp.argmax(1)), f1=f1_score(te.y, bp.argmax(1), average='macro'),
                top3=top_k_accuracy_score(te.y, bp, k=3), seconds=time.time() - t0)
print('TF-IDF + LR baseline', baseline, flush=True)

m1 = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(roles), attn_implementation='eager')
t0 = time.time(); curve1 = fit(m1, TensorDataset(a.input_ids, a.attention_mask, torch.tensor(tr.y.values)), epochs=3)
z = logits(m1, b.input_ids, b.attention_mask); yte = te.y.values
gpu = dict(accuracy=accuracy_score(yte, z.argmax(1)), f1=f1_score(yte, z.argmax(1), average='macro'),
           top3=top_k_accuracy_score(yte, z, k=3), seconds=time.time() - t0)
print('DistilBERT role model', gpu, flush=True)
log('role', baseline, gpu)
mb, sess = export(m1, 'role', b.input_ids, b.attention_mask)
quant = None
if sess:
    n = 800; zq = ort_logits(sess, b.input_ids[:n], b.attention_mask[:n])
    quant = dict(accuracy=accuracy_score(yte[:n], zq.argmax(1)), same_subset_fp32=accuracy_score(yte[:n], z[:n].argmax(1)), mb_before=mb[0], mb_after=mb[1])
log('quantised role model', quant)
audit = te.sample(300, random_state=RNG)
OUT['role'] = dict(max_len=L1, n_train=len(tr), n_test=len(te), baseline=baseline, distilbert=gpu, quantised=quant, loss_curve=curve1,
                   confusion=confusion_matrix(yte, z.argmax(1)).tolist(),
                   audit=dict(text=[x[:2500] for x in audit.text], y=audit.y.tolist()))
tok.save_pretrained('tok'); os.replace('tok/tokenizer.json', 'tokenizer.json')
json.dump(OUT, open('deep.json', 'w'), default=float)
del m1; torch.cuda.empty_cache()

# ================= 2. SKILL MODEL (NER) =================
# Distant supervision: the dataset lists each posting's skills; wherever one of them appears in the text, tag it B/I.
L2 = 256
def char_tags(text, skills):
    low, arr = text.lower(), np.zeros(len(text), dtype=np.int64); found = 0
    for s in sorted({x.strip().lower() for x in skills.split(',')}, key=len):
        if len(s) < 2: continue
        for mt in re.finditer(r'(?<![a-z0-9])' + re.escape(s) + r'(?![a-z0-9])', low):
            arr[mt.start():mt.end()] = 2; arr[mt.start()] = 1; found += 1
    return arr, found
nd = df.sample(min(len(df), 11000), random_state=RNG)
ptr, pte = train_test_split(nd, test_size=0.1, random_state=RNG)
def windows(part):
    I, A, Y, hit = [], [], [], 0
    for text, skills in zip(part.job_summary, part.job_skills):
        arr, f = char_tags(text, skills); hit += f > 0
        e = tok(text, truncation=True, max_length=L2, padding='max_length', return_overflowing_tokens=True, return_offsets_mapping=True)
        for w in range(min(len(e['input_ids']), 5)):
            lab = [-100 if s == t else int(arr[s]) for s, t in e['offset_mapping'][w]]
            if any(l > 0 for l in lab): I.append(e['input_ids'][w]); A.append(e['attention_mask'][w]); Y.append(lab)
    print('postings with at least one skill found in text:', hit, '/', len(part), '| windows:', len(I), flush=True)
    return torch.tensor(I), torch.tensor(A), torch.tensor(Y)
Itr, Atr, Ytr = windows(ptr); Ite, Ate, Yte = windows(pte)

def ents(seq):
    out, start = set(), None
    for i, l in enumerate(list(seq) + [0]):
        if start is not None and l != 2: out.add((start, i)); start = None
        if l == 1 or (l == 2 and start is None): start = i
    return out
def ner_scores(pred, gold):
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        ok = g != -100; ep, eg = ents(np.where(ok, p, 0)), ents(np.where(ok, g, 0))
        tp += len(ep & eg); fp += len(ep - eg); fn += len(eg - ep)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9), entities=tp + fn)

m2 = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=3, attn_implementation='eager')
t0 = time.time(); curve2 = fit(m2, TensorDataset(Itr, Atr, Ytr), epochs=2)
z2 = logits(m2, Ite, Ate); g2 = Yte.numpy()
ner = ner_scores(z2.argmax(-1), g2); ner['seconds'] = time.time() - t0
print('DistilBERT skill model', ner, flush=True)
log('ner', ner)
mb2, sess2 = export(m2, 'ner', Ite, Ate)
nq = None
if sess2:
    n = 400; nq = ner_scores(ort_logits(sess2, Ite[:n], Ate[:n]).argmax(-1), g2[:n])
    nq.update(same_subset_fp32=ner_scores(z2[:n].argmax(-1), g2[:n])['f1'], mb_before=mb2[0], mb_after=mb2[1])
log('quantised skill model', nq)
OUT['ner'] = dict(max_len=L2, n_train_windows=len(Itr), n_test_windows=len(Ite), distilbert=ner, quantised=nq, loss_curve=curve2)

import shutil; shutil.rmtree('tok', ignore_errors=True)
json.dump(OUT, open('deep.json', 'w'), default=float)
print('done', os.listdir('.'))
