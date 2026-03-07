#!/usr/bin/env python3
"""Quick diagnostic: load a Track A checkpoint and test on key benchmarks."""
import sys, torch, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.models.loader import load_model
from src.reservoir.esn import ESN
from src.types import ReservoirConfig
from peft import PeftModel
from scripts.train_track_a_readonly import ReadOnlySidecarBundle, SidecarHookManager
from src.eval.benchmarks.memory import AssociativeRecall, VariableTracking

def test_checkpoint(ckpt_path, n=10, use_sidecar=True, device_str='cuda'):
    device = torch.device(device_str)
    dtype = torch.bfloat16

    wrapper = load_model('qwen3.5-0.8b', dtype=dtype, device=device_str)
    tok = wrapper.tokenizer
    hdim = wrapper.model.config.hidden_size
    sidecar_layers = list(range(3, wrapper.model.config.num_hidden_layers, 4))

    lora_path = f'{ckpt_path}/lora_adapter'
    if __import__('os').path.exists(f'{lora_path}/adapter_config.json'):
        model = PeftModel.from_pretrained(wrapper.model, lora_path)
        print(f"  Loaded LoRA adapter from {lora_path}")
    else:
        model = wrapper.model
        print(f"  No LoRA adapter found — using base model + sidecar only")
    model.eval()

    esn = ESN(ReservoirConfig(size=10000, spectral_radius=0.9, leak_rate=0.5,
              input_scaling=1.0, sparsity=0.01, seed=42), input_dim=hdim)
    hooks = None
    if use_sidecar:
        sidecar = ReadOnlySidecarBundle(sidecar_layers, 10000, hdim, 8, 0.0)
        sidecar.load_state_dict(torch.load(f'{ckpt_path}/sidecar_weights.pt', map_location=device))
        sidecar = sidecar.to(device).to(dtype).eval()
        hooks = SidecarHookManager(model, sidecar, sidecar_layers)
    else:
        print("  Sidecar DISABLED — testing LoRA only")
    embed = model.get_input_embeddings()

    def run_task(name, bench):
        hits = 0
        for i, ex in enumerate(bench):
            if i >= n: break
            ids = tok.encode(ex.input, padding=False, truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                embs = embed(ids)
            esn.reset()
            e = embs[0].detach().float().cpu().numpy()
            states = np.zeros((e.shape[0], esn.n), dtype=np.float32)
            for t in range(e.shape[0]):
                states[t] = esn.step(e[t])
            if hooks:
                hooks.set_reservoir_states(states[None])
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=16, pad_token_id=tok.eos_token_id, do_sample=False)
            if hooks:
                hooks.clear_reservoir_states()
            gen = tok.decode(out[0, ids.shape[-1]:]).strip()
            # Check if target appears at start or after common prefixes like "Answer: "
            gen_clean = gen.lstrip()
            hit = (gen_clean.startswith(ex.target)
                   or gen_clean == ex.target
                   or gen_clean.startswith(f"Answer: {ex.target}")
                   or gen_clean.startswith(f"answer: {ex.target}")
                   or ex.target in gen_clean.split('\n')[0][:20])
            hits += hit
            if i < 3:
                mark = "OK" if hit else "MISS"
                print(f'  {name}: expected=[{ex.target}] got=[{gen[:50]}] {mark}')
        return hits

    ar_hits = run_task("AR", AssociativeRecall(n=n, num_pairs=5))
    vt_hits = run_task("VT", VariableTracking(n=n, num_variables=3))
    print(f'  SCORE: AR={ar_hits}/{n}  VT={vt_hits}/{n}')

    if hooks:
        hooks.remove_hooks()
    del model, hooks, esn
    torch.cuda.empty_cache()
    return ar_hits, vt_hits

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/track_a_readonly/final"
    no_sidecar = "--no-sidecar" in sys.argv
    use_cpu = "--cpu" in sys.argv
    device_str = "cpu" if use_cpu else "cuda"
    print(f"Testing {ckpt}..." + (" (no sidecar)" if no_sidecar else "") + (f" ({device_str})" if use_cpu else ""))
    test_checkpoint(ckpt, use_sidecar=not no_sidecar, device_str=device_str)
