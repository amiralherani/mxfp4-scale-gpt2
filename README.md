# Does the MXFP4 block scale matter on a real model?

MXFP4 stores 32 values in 4 bits each, plus one power-of-two scale shared by the whole block. The OCP Microscaling spec (v1.1, section 6.3) gives an example rule for picking that scale, which is to put the largest element of the block in the top binade, and it says other rules are allowed.

I wanted to know how much the example rule leaves on the table, measured on a real network and not on Gaussian noise. So I quantised the weights of GPT-2 (124M) to MXFP4 twice:

- **SPEC**: the example rule from the spec.
- **BEST**: for every block, try the spec's scale, one step below it and one step above it, and keep whichever gives the lowest squared error.

Then I measured the weight error and the WikiText-2 test perplexity for both. Weights only, direct rounding, no calibration data and no fine-tuning, so the only thing that differs between the two runs is the scale.

## What came out

**Weight error** (squared error relative to the squared norm of the weights):

| Quantised | Blocks of 32 | SPEC | BEST | Change |
|---|---:|---:|---:|---:|
| The 48 matrices inside the transformer blocks | 2,654,208 | 0.013705 | 0.012869 | -6.1% |
| Those plus token and position embeddings | 3,884,952 | 0.013607 | 0.012823 | -5.8% |

**Which scale wins**, over all 3,884,952 blocks:

| Best scale | Blocks | Share |
|---|---:|---:|
| the spec's | 3,380,620 | 87.0% |
| one step above | 504,332 | 13.0% |
| one step below | 0 | 0% |

Blocks where one step below wins do exist (the self-test in `mxfp4.py` has one: a 4 followed by thirty-one values of 1/4), but GPT-2 does not contain a single one. I also tried every scale from four steps below to four steps above on a sample of about 20,000 blocks per matrix. The number of blocks whose best scale fell outside the three candidates: 0.

**Perplexity** on the full WikiText-2 test split (FP32 model: 29.95):

| Quantised | SPEC | BEST | NLL change per token (95% interval) | Windows where BEST is better |
|---|---:|---:|---|---:|
| Transformer blocks | 34.72 | 34.73 | +0.00012 [-0.00140, +0.00161] | 141 of 280 |
| Blocks and embeddings | 63.29 | 60.35 | -0.04765 [-0.05339, -0.04203] | 248 of 280 |

## How I read it

The weight error behaves the way synthetic data predicts. On unit-variance Gaussian blocks the same comparison gives 6.6% less error, and GPT-2 gives 6.1%.

The model does not care. With the transformer blocks quantised, perplexity is 34.72 with the spec rule and 34.73 with the best scale, and the interval sits on both sides of zero. I expected a small gain here and there isn't one. Six percent less weight error bought nothing.

The embeddings are a different story. Quantising them to 4 bits with no calibration does a lot of damage on its own (perplexity goes from the mid 30s to the 60s), and there the better scale removes 8.8% of the total damage and wins in 248 of 280 windows. I don't have an explanation I trust for why the embeddings respond and the blocks don't. The token embedding is tied to the output head in GPT-2, which may be part of it. I have not tested that.

So the summary is this: a better block scale reliably lowers weight error, and whether that reaches model quality depends on which tensors you are looking at. Weight MSE on its own is a poor target.

## What was fixed before the first run

I wrote these down before downloading the model, so that I could not pick the flattering comparison afterwards.

- Model `openai-community/gpt2`, unmodified. Data `Salesforce/wikitext`, `wikitext-2-raw-v1`, the whole test split.
- Non-overlapping 1024-token windows, every full window (280 of them), FP32 arithmetic.
- MXFP4 as in OCP MX v1.1: E2M1 elements, blocks of 32, round to nearest with ties to even.
- Blocks run along the dot-product (input) axis of each matrix, because that is what an MX dot-product unit consumes.
- Primary result: transformer blocks only. Secondary: blocks plus embeddings. Biases and LayerNorm stay in FP32.
- Paired comparison per window, 95% bootstrap interval over windows, 10,000 resamples, fixed seed.
- "No measurable benefit" means the interval for the primary result contains zero. That is the verdict here.

The first run compared more encoder variants than the two shown. The others are cheaper approximations of BEST and are not part of this repository. Every number on this page comes from a fresh run of exactly the code published here.

## Files

| File | What it is |
|---|---|
| `mxfp4.py` | The encoder, about 120 lines of numpy, with a self-test. |
| `run_gpt2.py` | Quantises the model, evaluates perplexity, runs the bootstrap. |
| `results/weights.json` | Error and best-scale counts for each of the 50 matrices. |
| `results/nll_windows.json` | Summed NLL for each of the 280 windows, for all five model variants. |
| `results/summary.json` | Perplexities, intervals, and the hashes of the scripts that produced them. |

## Run it yourself

```
pip install numpy torch transformers datasets
python mxfp4.py
python run_gpt2.py --stage weights
python run_gpt2.py --stage ppl
python run_gpt2.py --stage report
```

About 35 minutes on an 8-core desktop CPU, no GPU needed. On CPU the run is deterministic. My numbers were produced with Python 3.13.14, numpy 2.5.5, torch 2.15.0+cpu and transformers 5.24.0; other versions may differ in the last digits.

## Limits

One model and one dataset. Weights only: activations and the KV cache are not quantised. No calibration, clipping search or fine-tuning, all of which a production flow would use and any of which could change the picture. MXFP4 only; I have not run MXFP6 or MXFP8 through this.

## Elsewhere

The same files are on Hugging Face at https://huggingface.co/datasets/Amiralherani/mxfp4-scale-gpt2, and there is a Kaggle notebook that runs the whole thing in the browser: https://www.kaggle.com/code/amiralherani/does-the-mxfp4-block-scale-matter-gpt-2

## Contact

Amir Al-Herani, aalhorini@gmail.com. If you re-run this and get different numbers, I'd like to know.
