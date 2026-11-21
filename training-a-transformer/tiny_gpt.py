# tiny_gpt.py · Copyright (c) 2026 Alireza Bayat Makou · SPDX-License-Identifier: MIT
# The model and the training loop are adapted from nanoGPT, Copyright (c) 2022 Andrej Karpathy (MIT License).
# License text: https://github.com/alirezabayatmk/language-modeling/blob/main/LICENSE-CODE · Simulation: https://alirezabayatmk.github.io/language-modeling/training-a-transformer/
"""tiny_gpt.py: the whole simulation in one file, from text to generated text.

Same data, same seeds, same operations as the simulation: running it reproduces
the numbers on the slides (losses, the trained model's predictions).
The model and the training loop follow Karpathy's nanoGPT (model.py, train.py).
Run: python tiny_gpt.py   (CPU only; about 20 s on a laptop)
"""
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================================== 0. settings
CORPUS = [
    "the capital of france is paris",
    "the capital of italy is rome",
    "the capital of spain is madrid",
    "the capital of japan is tokyo",
    "in france people speak french",
    "in italy people speak italian",
    "in spain people speak spanish",
    "in japan people speak japanese",
]
END = "_"                           # marks the end of a word
NUM_MERGES = 35                     # tokenizer: 22 symbols + 35 merges
SPECIALS = ["<pad>", "<s>", "</s>"]
D, H, LAYERS, D_FF = 8, 2, 2, 32    # width, heads, blocks, MLP width
BLOCK, BATCH = 16, 8                # tokens per window, windows per batch
STEPS, LR, MIN_LR, WARMUP, SEED = 600, 0.02, 0.002, 30, 0
WD, BETAS, CLIP, STD = 0.1, (0.9, 0.95), 1.0, 0.2

# ===================================================== 1. tokenizer (BPE)
word_counts = Counter(w for s in CORPUS for w in s.split())
words = list(dict.fromkeys(w for s in CORPUS for w in s.split()))


def merge(symbols, a, b):
    """Replace every neighboring pair (a, b) by one symbol a + b."""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(symbols[i]); i += 1
    return out


splits = {w: list(w) + [END] for w in words}
base = sorted(set("".join(words))) + [END]
merges = []
while len(merges) < NUM_MERGES:
    counts = {}
    for w in words:
        s = splits[w]
        for p in zip(s, s[1:]):
            counts[p] = counts.get(p, 0) + word_counts[w]
    best = max(counts, key=counts.get)
    merges.append(best)
    splits = {w: merge(splits[w], *best) for w in words}
vocab = SPECIALS + base + [a + b for a, b in merges]
stoi = {t: i for i, t in enumerate(vocab)}


def tokenize_word(w):
    s = list(w) + [END]
    for a, b in merges:
        s = merge(s, a, b)
    return s


def encode(sentence):
    toks = [t for w in sentence.split() for t in tokenize_word(w)]
    return [stoi["<s>"]] + [stoi[t] for t in toks]


# ===================================================== 2. data
seqs = [encode(s) + [stoi["</s>"]] for s in CORPUS]
stream = torch.tensor([t for s in seqs for t in s])   # 98 token IDs


def get_batch(data, gen, first_at_start=False):
    """BATCH random windows; the targets y are x shifted by one."""
    ix = torch.randint(len(data) - BLOCK, (BATCH,), generator=gen)
    if first_at_start:
        ix[0] = 0                   # step 1: row 0 holds our example
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
    return x, y


# ===================================================== 3. model (nanoGPT)
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c["n_embd"], 3 * c["n_embd"], bias=c["bias"])
        self.c_proj = nn.Linear(c["n_embd"], c["n_embd"], bias=c["bias"])
        self.n_head, self.n_embd = c["n_head"], c["n_embd"]

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        A = F.softmax(att.masked_fill(mask, float("-inf")), dim=-1)
        z = A @ v
        y = z.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c["n_embd"], 4 * c["n_embd"], bias=c["bias"])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * c["n_embd"], c["n_embd"], bias=c["bias"])

    def forward(self, x):
        pre = self.c_fc(x)
        hidden = self.gelu(pre)
        return self.c_proj(hidden)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1 = LayerNorm(c["n_embd"], bias=c["bias"])
        self.attn = CausalSelfAttention(c)
        self.ln_2 = LayerNorm(c["n_embd"], bias=c["bias"])
        self.mlp = MLP(c)

    def forward(self, x):
        z1 = self.ln_1(x)
        attn = self.attn(z1)
        x1 = x + attn
        z2 = self.ln_2(x1)
        mlp = self.mlp(z2)
        x2 = x1 + mlp
        return x2


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c["vocab_size"], c["n_embd"]),
            wpe=nn.Embedding(c["block_size"], c["n_embd"]),
            h=nn.ModuleList([Block(c) for _ in range(c["n_layer"])]),
            ln_f=LayerNorm(c["n_embd"], bias=c["bias"])))
        self.lm_head = nn.Linear(c["n_embd"], c["vocab_size"], bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                std = c["std"] / math.sqrt(2 * c["n_layer"])
                torch.nn.init.normal_(p, mean=0.0, std=std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.c["std"])
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.c["std"])

    def forward(self, idx, targets=None):
        T = idx.size(-1)
        e = self.transformer.wte(idx)
        p = self.transformer.wpe(torch.arange(T))
        x = e + p
        for b in self.transformer.h:
            x = b(x)
        final = self.transformer.ln_f(x)
        logits = self.lm_head(final)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss


CFG = dict(n_layer=LAYERS, n_head=H, n_embd=D, block_size=BLOCK,
           vocab_size=len(vocab), bias=False, std=STD)

# ===================================================== 4. training (nanoGPT)
def make_optimizer(m):
    """Weight decay on matrices and embeddings, not on LayerNorm gains."""
    decay = [p for n, p in m.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in m.named_parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": WD},
              {"params": nodecay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=LR, betas=BETAS)


def get_lr(it, steps=STEPS):
    """Linear warm-up, then cosine decay down to MIN_LR."""
    if it < WARMUP:
        return LR * (it + 1) / (WARMUP + 1)
    r = (it - WARMUP) / (steps - WARMUP)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * r)) * (LR - MIN_LR)


def train_step(m, opt, it, xb, yb):
    for g in opt.param_groups:
        g["lr"] = get_lr(it)
    _, loss = m(xb, yb)                     # forward: loss of this batch
    opt.zero_grad(set_to_none=True)
    loss.backward()                         # backward: every gradient
    torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
    opt.step()                              # AdamW update
    return loss.item()


def eval_loss(m, data, skip=0):
    """Mean loss over every window of the text (targets from skip on)."""
    ix = range(len(data) - BLOCK)
    xs = torch.stack([data[i:i + BLOCK] for i in ix])
    ys = torch.stack([data[i + 1:i + 1 + BLOCK] for i in ix])
    with torch.no_grad():
        logits, _ = m(xs)
    ce = F.cross_entropy(logits.transpose(1, 2), ys, reduction="none")
    return float(ce[:, skip:].mean())


def train(data, val_data=None, val_skip=0, first_at_start=False):
    torch.manual_seed(SEED)                 # the same random start
    model = GPT(CFG)
    opt = make_optimizer(model)
    gen = torch.Generator().manual_seed(1)  # the same batches
    losses, val_losses = [eval_loss(model, data)], []
    for it in range(STEPS):
        xb, yb = get_batch(data, gen, first_at_start and it == 0)
        train_step(model, opt, it, xb, yb)
        losses.append(eval_loss(model, data))
        if val_data is not None:
            val_losses.append(eval_loss(model, val_data, val_skip))
    return model, losses, val_losses


model, losses, _ = train(stream, first_at_start=True)
print(f"loss: start {losses[0]:.4f}, after {STEPS} steps {losses[-1]:.4f}")

# ===================================================== 5. when to stop
HOLD = 7                                    # hold out the last sentence
train_stream = torch.tensor([t for i, s in enumerate(seqs)
                             if i != HOLD for t in s])
CTX = BLOCK + 1 - len(seqs[HOLD])           # context: end of sentence 7
val_stream = torch.tensor(seqs[HOLD - 1][-CTX:] + seqs[HOLD])
_, tr, va = train(train_stream, val_stream, val_skip=CTX)
best = min(range(STEPS), key=lambda i: va[i])
print(f"held-out loss: lowest {va[best]:.4f} after step {best + 1}, "
      f"final {va[-1]:.4f} (training text: {tr[-1]:.4f})")

# ===================================================== 6. using the model
def next_token(logits, temperature=1.0, top_p=1.0, gen=None):
    """Scores / T, softmax, keep the top-p set, renormalize, draw one."""
    if temperature == 0:
        return int(torch.argmax(logits))   # greedy: the most likely
    p = torch.softmax(logits / temperature, -1)
    ps, order = p.sort(descending=True)
    keep = ps.cumsum(0) - ps < top_p        # smallest set reaching top_p
    p_keep = torch.zeros_like(p)
    p_keep[order[keep]] = ps[keep]
    p_keep = p_keep / p_keep.sum()
    return int(torch.multinomial(p_keep, 1, generator=gen))


def generate(prompt, temperature=0.0, top_p=1.0, seed=0, max_new=8):
    gen = torch.Generator().manual_seed(seed)
    ids = encode(prompt)
    for _ in range(max_new):
        with torch.no_grad():
            logits, _ = model(torch.tensor([ids[-BLOCK:]]))
        nxt = next_token(logits[0, -1], temperature, top_p, gen)
        ids.append(nxt)
        if vocab[nxt] == "</s>":
            break
    return " ".join(vocab[i] for i in ids)


for prompt in ["the capital of france is", "in japan people speak"]:
    print(generate(prompt))
print(generate("the capital of", temperature=1.0, top_p=0.9))

# dropout (not used in our run; nanoGPT's default is dropout = 0):
# while training, each number becomes 0 with probability p, the rest
# are divided by 1 - p; at use time it does nothing
print(F.dropout(torch.ones(8), p=0.1, training=True))
