"""Runs the two fine-tuned DistilBERT models (int8 ONNX) on the laptop CPU. No PyTorch needed."""
import json, re
from pathlib import Path
import numpy as np

M = Path(__file__).parent / 'models'
READY = all((M / f).exists() for f in ('role.onnx', 'ner.onnx', 'tokenizer.json', 'deep.json'))
if READY:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    META = json.loads((M / 'deep.json').read_text())
    _tok = Tokenizer.from_file(str(M / 'tokenizer.json')); _tok.no_padding(); _tok.no_truncation()
    _role = ort.InferenceSession(str(M / 'role.onnx'), providers=['CPUExecutionProvider'])
    _ner = ort.InferenceSession(str(M / 'ner.onnx'), providers=['CPUExecutionProvider'])
    CLS, SEP = _tok.token_to_id('[CLS]'), _tok.token_to_id('[SEP]')

def _run(sess, rows):
    L = max(len(r) for r in rows)
    ids = np.zeros((len(rows), L), dtype=np.int64); att = np.zeros_like(ids)
    for i, r in enumerate(rows): ids[i, :len(r)] = r; att[i, :len(r)] = 1
    return sess.run(None, {'input_ids': ids, 'attention_mask': att})[0]

def role_probs(texts, bs=16):
    """Text -> probability of each role (order = META['roles'])."""
    L = META['role']['max_len']; out = []
    rows = [[CLS] + _tok.encode(t, add_special_tokens=False).ids[:L - 2] + [SEP] for t in texts]
    for i in range(0, len(rows), bs):
        z = _run(_role, rows[i:i + bs]); z = z - z.max(1, keepdims=True); e = np.exp(z); out.append(e / e.sum(1, keepdims=True))
    return np.concatenate(out)

def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+|•', text) if len(s.strip()) > 25]

def sentence_impact(text, target, cap=24):
    """Occlusion: delete one sentence at a time and see how far the role probability moves."""
    S = sentences(text)[:cap]
    if len(S) < 2: return []
    full = ' '.join(S)
    P = role_probs([full] + [' '.join(S[:i] + S[i + 1:]) for i in range(len(S))])[:, target]
    return sorted(((s, float(P[0] - p)) for s, p in zip(S, P[1:])), key=lambda x: -x[1])

def ner(text):
    """Text -> [(start, end, phrase)] for every span the model tags as a skill."""
    e = _tok.encode(text, add_special_tokens=False)
    if not e.ids: return []
    W = META['ner']['max_len'] - 2
    rows = [[CLS] + e.ids[i:i + W] + [SEP] for i in range(0, len(e.ids), W)]
    lab = np.concatenate([_run(_ner, [r])[0, 1:-1].argmax(-1) for r in rows])
    out, start = [], None
    for i in range(len(lab) + 1):
        l = lab[i] if i < len(lab) else 0
        cont = i < len(lab) and e.tokens[i].startswith('##')            # never cut a word in half
        if start is not None and l != 2 and not cont:
            a, b = e.offsets[start][0], e.offsets[i - 1][1]; out.append((a, b, text[a:b])); start = None
        if start is None and (l == 1 or l == 2) and not cont: start = i
    return out
