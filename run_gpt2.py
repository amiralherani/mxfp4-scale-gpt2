"""Does the MXFP4 block scale matter on a real model?  GPT-2 (124M), WikiText-2 test perplexity.

  python run_gpt2.py --stage weights    # quantisation error per weight matrix, no forward pass
  python run_gpt2.py --stage ppl        # perplexity: FP32, then SPEC and BEST in both scopes
  python run_gpt2.py --stage report     # paired bootstrap, writes results/summary.json

Weights only, direct rounding. No calibration data, no fine-tuning.
Everything is deterministic on CPU except the bootstrap, which is seeded.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / "hf_cache"))
RES = HERE / "results"
MODEL_ID = "openai-community/gpt2"
DATA_ID, DATA_CFG, DATA_SPLIT = "Salesforce/wikitext", "wikitext-2-raw-v1", "test"
CTX = 1024
SEED = 20261122
THREADS = 8

import mxfp4  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance():
    import torch, transformers
    return {"run_gpt2.py": sha(HERE / "run_gpt2.py"), "mxfp4.py": sha(HERE / "mxfp4.py"),
            "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "model": MODEL_ID,
            "dataset": f"{DATA_ID}/{DATA_CFG}/{DATA_SPLIT}", "seed": SEED}


def load_model():
    import torch
    from transformers import GPT2LMHeadModel
    torch.set_num_threads(THREADS)
    return GPT2LMHeadModel.from_pretrained(MODEL_ID).eval()


def tensor_plan(model):
    """Which matrices get quantised, and along which axis the blocks of 32 run.

    Blocks follow the dot-product (input) axis, because that is what an MX dot-product
    unit consumes. GPT-2's Conv1D stores W as (in, out) and computes x @ W, so those are
    transposed first. The token embedding is tied to the output head (logits = h @ wte.T),
    so its input axis is already the last one. The position embedding is a lookup table
    and is blocked along its 768-wide rows.
    """
    blocks, embeddings = {}, {}
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        if name.startswith("transformer.h."):
            blocks[name] = (p, True)
        elif name in ("transformer.wte.weight", "transformer.wpe.weight"):
            embeddings[name] = (p, False)
    return blocks, embeddings


def quantise_tensor(p, transpose):
    w = p.detach().cpu().numpy().astype(np.float64)
    w2 = w.T if transpose else w
    x = mxfp4.to_blocks(w2)
    out, errs = mxfp4.encode(x)
    st = mxfp4.block_stats(x, out, errs)
    st["best_outside_3_candidates"] = mxfp4.wider_search(x[:: max(1, len(x) // 20000)])
    recon = {}
    for m in mxfp4.MODES:
        r = out[m][0].reshape(w2.shape)
        r = r.T if transpose else r
        recon[m] = r.astype(np.float32)
        assert np.array_equal(recon[m].astype(np.float64), r), "quantised value is not exact in FP32"
    return recon, st


def quantise_all(model, verbose=True):
    blocks, embeddings = tensor_plan(model)
    stats, cache = {}, {}
    for scope, plan in (("blocks", blocks), ("embeddings", embeddings)):
        for name, (p, tr) in plan.items():
            recon, st = quantise_tensor(p, tr)
            st["scope"] = scope
            st["shape"] = list(p.shape)
            stats[name] = st
            for m in mxfp4.MODES:
                cache[f"{m}|{name}"] = recon[m]
            if verbose:
                print(f"{name:42s} {st['blocks']:8d} blocks   spec {st['SPEC']['rel_sse']:.6f}   "
                      f"best {st['BEST']['rel_sse']:.6f}   {st['best_offset_counts']}", flush=True)
    return stats, cache


def stage_weights():
    model = load_model()
    t0 = time.time()
    rep = {"provenance": provenance(), "selftest_gaussian": mxfp4.selftest()}
    rep["tensors"], _ = quantise_all(model)
    rep["aggregate"] = {}
    for label, scopes in (("blocks", ("blocks",)), ("blocks+embeddings", ("blocks", "embeddings"))):
        ts = [t for t in rep["tensors"].values() if t["scope"] in scopes]
        n2 = sum(t["norm2"] for t in ts)
        nb = sum(t["blocks"] - t["zero_blocks"] for t in ts)
        agg = {"tensors": len(ts), "blocks": sum(t["blocks"] for t in ts),
               "best_outside_3_candidates": sum(t["best_outside_3_candidates"] for t in ts),
               "best_offset_counts": {k: sum(t["best_offset_counts"][k] for t in ts) for k in ("-1", "0", "+1")}}
        for m in mxfp4.MODES:
            agg[m] = {"rel_sse": sum(t[m]["sse"] for t in ts) / n2,
                      "mean_block_rel_mse": sum(t[m]["mean_block_rel_mse"] * (t["blocks"] - t["zero_blocks"])
                                                for t in ts) / nb}
        rep["aggregate"][label] = agg
    rep["seconds"] = time.time() - t0
    RES.mkdir(exist_ok=True)
    (RES / "weights.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["aggregate"], indent=1))


def windows():
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(MODEL_ID)
    ds = load_dataset(DATA_ID, DATA_CFG, split=DATA_SPLIT)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return ids, (len(ids) - 1) // CTX          # full, non-overlapping windows only


def eval_nll(model, ids, n, limit=None):
    """Summed NLL over tokens 1..1023 of each non-overlapping 1024-token window."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(n if limit is None else min(n, limit)):
            w = ids[i * CTX:(i + 1) * CTX].unsqueeze(0)
            logits = model(w).logits[0, :-1].double()
            out.append(float(torch.nn.functional.cross_entropy(logits, w[0, 1:], reduction="sum")))
    return np.array(out)


def stage_ppl(limit=None):
    import torch
    model = load_model()
    blocks, embeddings = tensor_plan(model)
    _, qz = quantise_all(model, verbose=False)
    params = dict(model.named_parameters())
    orig = {n: params[n].detach().clone() for n in list(blocks) + list(embeddings)}
    ids, n = windows()
    RES.mkdir(exist_ok=True)
    path = RES / ("nll_windows.json" if limit is None else "nll_windows_smoke.json")
    res = json.loads(path.read_text()) if path.exists() else {}
    res["_meta"] = {"windows": n if limit is None else min(n, limit), "tokens_per_window": CTX - 1,
                    "total_tokens_in_split": int(len(ids)), "provenance": provenance()}

    def run(tag, names, mode):
        if tag in res:                          # lets an interrupted run pick up where it stopped
            return
        with torch.no_grad():
            for nm, p in orig.items():
                params[nm].copy_(p)
            for nm in names:
                params[nm].copy_(torch.from_numpy(qz[f"{mode}|{nm}"]))
        t0 = time.time()
        nll = eval_nll(model, ids, n, limit)
        res[tag] = nll.tolist()
        path.write_text(json.dumps(res))
        print(f"{tag:26s} ppl {np.exp(nll.sum() / (len(nll) * (CTX - 1))):.4f}   {time.time() - t0:.0f}s", flush=True)

    run("FP32", [], None)
    for scope, names in (("blocks", list(blocks)), ("blocks+embeddings", list(blocks) + list(embeddings))):
        for m in mxfp4.MODES:
            run(f"{scope}:{m}", names, m)


def stage_report():
    res = json.loads((RES / "nll_windows.json").read_text())
    meta = res.pop("_meta")
    T = meta["tokens_per_window"]
    arr = {k: np.array(v) for k, v in res.items()}
    nwin = len(arr["FP32"])
    boot = np.random.default_rng(SEED).integers(0, nwin, size=(10000, nwin))
    ppl = lambda a: float(np.exp(a.sum() / (len(a) * T)))
    summ = {"meta": meta, "ppl": {k: ppl(v) for k, v in arr.items()}, "best_vs_spec": {}}
    for scope in ("blocks", "blocks+embeddings"):
        d = (arr[f"{scope}:BEST"] - arr[f"{scope}:SPEC"]) / T       # nats per token, per window
        bs = d[boot].mean(1)
        p_spec, p_best, p_fp = summ["ppl"][f"{scope}:SPEC"], summ["ppl"][f"{scope}:BEST"], summ["ppl"]["FP32"]
        summ["best_vs_spec"][scope] = {
            "delta_nll_per_token": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "windows_best_is_better": int((d < 0).sum()), "windows": nwin,
            "ppl_damage_removed_pct": 100 * (p_spec - p_best) / (p_spec - p_fp)}
    (RES / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "meta"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["weights", "ppl", "report"])
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N windows, separate output file")
    a = ap.parse_args()
    {"weights": stage_weights, "ppl": lambda: stage_ppl(a.limit), "report": stage_report}[a.stage]()
