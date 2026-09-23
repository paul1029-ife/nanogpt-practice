# nanoGPT practice — local setup

## Machine constraints that shaped this setup

- Intel i7 Mac, no CUDA and no MPS (MPS is Apple-silicon only) → **CPU training only**
- PyTorch dropped Intel-Mac (x86_64 macOS) wheels after **2.2.2**, so we are pinned there
- torch 2.2.2 was built against NumPy 1.x → **numpy must be `<2`** or torch fails to import
- torch 2.2.2 supports Python ≤ 3.11 → venv is **Python 3.11**

## Environment

```bash
uv venv --python 3.11
uv pip install "torch==2.2.2" "numpy<2" tiktoken tqdm
```

Run anything with `.venv/bin/python` (or `source .venv/bin/activate` first).

## Data

```bash
.venv/bin/python data/shakespeare_char/prepare.py
```

Produces `train.bin` (1,003,854 tokens), `val.bin` (111,540 tokens), `meta.pkl`.
Vocab is 65 characters — every distinct character in Tiny Shakespeare.

## Train

```bash
.venv/bin/python train.py config/train_shakespeare_char_cpu.py
```

Config lives in `config/train_shakespeare_char_cpu.py`. Two settings there exist
purely because of this machine:

- `device = 'cpu'`, `compile = False` — `torch.compile` costs minutes of warmup
  and buys nothing on CPU.
- `dtype = 'float32'` — **necessary**. `train.py:73` defaults dtype to `float16`
  when no CUDA bf16 device is present, and `train.py:196` then builds a
  `torch.cuda.amp.GradScaler(enabled=True)`, which cannot work without CUDA.

Model is ~0.8M params: 4 layers, 4 heads, 128 embedding dim, 64-char context.

## Sample

```bash
.venv/bin/python sample.py --out_dir=out-shakespeare-char-cpu --device=cpu --num_samples=3 --max_new_tokens=500
```

## Reading the loss

Char-level Tiny Shakespeare reference points:

| val loss | meaning |
|---|---|
| ~4.17 | untrained — uniform guess over 65 chars (`ln 65`) |
| ~2.5  | learned letter frequencies, no words |
| ~1.8  | real words, broken grammar |
| ~1.5  | this config's rough floor — Shakespeare-ish lines, speaker names |
| ~1.1  | full-size nanoGPT on a GPU |

Watch **train vs val together**. Train falling while val rises is overfitting,
and on a 1M-character dataset you can reach it.
