"""Quick demo: Proton profiling a BERT forward pass to see a real call tree."""

import torch
import triton.profiler as proton
import triton.profiler.viewer as proton_viewer
from contextlib import contextmanager

DEVICE = "cuda"

@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)

# Load BERT
from transformers import BertModel, BertConfig

config = BertConfig(num_hidden_layers=4, max_position_embeddings=2048)  # smaller for speed
model = BertModel(config).half().to(DEVICE).eval()

# Fake input
batch_size, seq_len = 8, 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=DEVICE)
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=DEVICE)

# Warmup
for _ in range(3):
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()

# Profile
profile_name = "/tmp/proton_bert_demo"
proton.start(profile_name, hook="triton")
proton.deactivate(0)

with proton_context():
    for seq_len in [128, 512, 1024, 2048]:
        for _ in range(5):
            batch_size, seq_len = 8, seq_len
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=DEVICE)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=DEVICE)
            with proton.scope(f"bert_forward [seq_len={seq_len}]"):
                with torch.no_grad():
                    model(input_ids, attention_mask=attention_mask)

proton.finalize()

tree, metrics = proton_viewer.parse(["time/ms"], f"{profile_name}.hatchet")
proton_viewer.print_tree(tree, metrics)
