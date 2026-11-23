"""MXFP4 block encoder in plain numpy.

MXFP4 as in the OCP Microscaling (MX) spec v1.1: blocks of 32 values, each stored as a
4-bit E2M1 float, with one shared power-of-two scale per block (8-bit exponent).

Two ways of choosing the block scale are compared on the same blocks:

  SPEC   s = b, with b = floor(log2 max|x|) - 2.  This is the example rule in sec. 6.3
         of the spec: the largest element lands in the top binade of E2M1.
  BEST   whichever of b-1, b, b+1 gives the lowest squared error for the block
         (ties go to the smaller exponent).

Elements are rounded to the nearest E2M1 value, ties to the even 3-bit code, saturating
at 6. The representable magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6.
"""
import numpy as np

K = 32
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MIDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])            # rounding boundaries
TIE_UP = np.array([False, True, False, True, False, True, False])   # on a boundary: take the upper code?
S_MIN, S_MAX = -127, 127
MODES = ("SPEC", "BEST")


def round_e2m1(u):
    """Round to the E2M1 grid, ties to the even code, sign kept."""
    a = np.abs(u)
    idx = np.searchsorted(MIDS, a, side="left")
    j = np.minimum(idx, len(MIDS) - 1)
    tie = (idx < len(MIDS)) & (a == MIDS[j]) & TIE_UP[j]
    return np.copysign(GRID[idx + tie], u)


def quant_at(x, s):
    """x: (N, K) float64, s: (N,) integer exponents. Returns what a decoder would read back."""
    sc = np.ldexp(1.0, s.astype(np.int64))[:, None]
    return sc * round_e2m1(x / sc)


def spec_exponent(x):
    """b = floor(log2 max|x|) - 2 per block. All-zero blocks are flagged in nz."""
    m = np.abs(x).max(axis=1)
    nz = m > 0
    _, e = np.frexp(np.where(nz, m, 1.0))
    return ((e - 1) - 2).astype(np.int64), nz


def encode(x):
    """x: (N, K) float64 -> ({mode: (reconstruction, exponent)}, squared errors at b-1, b, b+1)."""
    assert x.ndim == 2 and x.shape[1] == K
    b, nz = spec_exponent(x)
    clip = lambda s: np.clip(s, S_MIN, S_MAX)
    s0, sp, sm = clip(b), clip(b + 1), clip(b - 1)
    q0, qp, qm = quant_at(x, s0), quant_at(x, sp), quant_at(x, sm)
    e0 = ((x - q0) ** 2).sum(1)
    ep = ((x - qp) ** 2).sum(1)
    em = ((x - qm) ** 2).sum(1)

    def pick(up, down):
        q = np.where(up[:, None], qp, q0)
        s = np.where(up, sp, s0)
        q = np.where(down[:, None], qm, q)
        s = np.where(down, sm, s)
        q[~nz] = 0.0
        return q, np.where(nz, s, S_MIN)

    none = np.zeros(len(x), bool)
    down = em <= np.minimum(e0, ep)
    out = {"SPEC": pick(none, none), "BEST": pick((ep < e0) & ~down, down)}
    return out, {"-1": em, "0": e0, "+1": ep}


def to_blocks(w2d):
    """(rows, cols) with cols divisible by 32 -> (rows*cols/32, 32). Blocks run along the last axis."""
    rows, cols = w2d.shape
    assert cols % K == 0, (rows, cols)
    return np.ascontiguousarray(w2d, dtype=np.float64).reshape(-1, K)


def block_stats(x, out, errs):
    n2 = (x ** 2).sum(1)
    ok = n2 > 0
    st = {"blocks": int(len(x)), "zero_blocks": int((~ok).sum()), "norm2": float(n2.sum())}
    for m in MODES:
        e = ((x - out[m][0]) ** 2).sum(1)
        st[m] = {"sse": float(e.sum()), "rel_sse": float(e.sum() / n2.sum()),
                 "mean_block_rel_mse": float((e[ok] / n2[ok]).mean())}
    em, e0, ep = errs["-1"], errs["0"], errs["+1"]
    down = (em <= np.minimum(e0, ep)) & (em < np.minimum(e0, ep)) & ok
    up = (ep < e0) & ~(em < np.minimum(e0, ep)) & ok
    st["best_offset_counts"] = {"-1": int(down.sum()), "+1": int(up.sum())}
    st["best_offset_counts"]["0"] = int(ok.sum()) - int(down.sum()) - int(up.sum())
    return st


def wider_search(x, lo=-4, hi=4):
    """Try every exponent from b-4 to b+4. Returns how many blocks have their best scale outside b-1..b+1."""
    b, nz = spec_exponent(x)
    es = np.stack([((x - quant_at(x, np.clip(b + d, S_MIN, S_MAX))) ** 2).sum(1) for d in range(lo, hi + 1)])
    am = es.argmin(0) + lo
    return int(((am < -1) | (am > 1))[nz].sum())


def selftest():
    def blk(first, rest):
        return np.array([[float(first)] + [float(rest)] * 31])
    # rounding, including every tie
    u = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.9, -2.5, -3.5])
    assert round_e2m1(u).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -2.0, -4.0]
    # one large value and 31 small ones: stepping the scale DOWN is better, even though
    # the large value then saturates. Errors at b-1, b, b+1 are exactly 1, 31/16, 31/16.
    x = blk(4, 0.25); out, errs = encode(x)
    assert (errs["-1"][0], errs["0"][0], errs["+1"][0]) == (1.0, 31 / 16, 31 / 16)
    assert out["SPEC"][1][0] == 0 and out["BEST"][1][0] == -1
    # a block where stepping up loses: 7 followed by 31 halves
    x = blk(7, 0.5); out, errs = encode(x)
    assert errs["+1"][0] / errs["0"][0] == 35 / 4 and out["BEST"][1][0] == 0
    # scale ties go to the smaller exponent; an all-zero block stays zero
    x = blk(4, 0); out, _ = encode(x); assert out["BEST"][1][0] == 0
    x = np.zeros((1, 32)); out, _ = encode(x); assert (out["BEST"][0] == 0).all()
    # unit-variance Gaussian blocks
    rng = np.random.default_rng(12345)
    x = rng.standard_normal((200000, 32)); out, errs = encode(x); st = block_stats(x, out, errs)
    g = {m: st[m]["mean_block_rel_mse"] for m in MODES}
    assert abs(g["SPEC"] - 0.013368) < 5e-5 and abs(g["BEST"] - 0.012489) < 5e-5, g
    assert wider_search(x[:50000]) == 0
    return g


if __name__ == "__main__":
    print("selftest OK", selftest())
