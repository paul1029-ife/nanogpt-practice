# Tiny char-level GPT sized for a CPU-only Intel Mac.
# Trains in roughly 10-15 minutes and produces recognisably Shakespeare-ish text.
# Run: python train.py config/train_shakespeare_char_cpu.py

out_dir = 'out-shakespeare-char-cpu'
eval_interval = 250
eval_iters = 50
log_interval = 10

always_save_checkpoint = False  # only save when val loss improves

wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 64  # context length: how many characters of history

# a baby GPT: ~0.8M params
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'cpu'
compile = False  # torch.compile has no speed benefit here and is slow to warm up

# train.py defaults dtype to float16 when no CUDA bf16 is present, which then
# switches on a CUDA GradScaler that cannot work on this machine. Force float32.
dtype = 'float32'
