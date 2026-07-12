# N-BUMP on the decisive saturation cells (declared before seeing new data; criterion
# unchanged from saturation_predictions.json). e13/e12 at n=32 gave: Haiku gain +0.344
# [.19,.50] (BINDS, sig) vs Sonnet gain +0.156 [.03,.28] (attenuated but nonzero); the
# BETWEEN-MODEL gain difference (+0.19) is borderline at n=32. Double n on the four
# decisive cells only -- k=4 at B in {15,100} x {Sonnet, Haiku} -- and bootstrap the
# PAIRED-over-facts gain difference (same probe facts in both models).
import os
os.environ["PROV_PROBES"] = "16"          # 16/type = 64 probes (first 32 already cached)
import numpy as np
from run import _build_backend
from e12_provisioning import probe_set
from e_ci_backfill import cell_perfact, diff_ci

RNG = np.random.default_rng(0)


def main():
    pset = probe_set()
    bs = _build_backend("anthropic", "claude-sonnet-4-6", "data/cache_prov_claude-sonnet-4-6.json")
    bh = _build_backend("anthropic", "claude-haiku-4-5", "data/cache_sat_claude-haiku-4-5.json")
    s15 = cell_perfact(bs, pset, 15, 4); s100 = cell_perfact(bs, pset, 100, 4)
    h15 = cell_perfact(bh, pset, 15, 4); h100 = cell_perfact(bh, pset, 100, 4)

    m, lo, hi = diff_ci(s100, s15)
    print(f"[Sonnet n={len(pset)}] k=4 gain B15->B100: +{m:.3f} [{lo:+.3f}, {hi:+.3f}]")
    m, lo, hi = diff_ci(h100, h15)
    print(f"[Haiku  n={len(pset)}] k=4 gain B15->B100: +{m:.3f} [{lo:+.3f}, {hi:+.3f}]")
    # paired-over-facts between-model gain difference
    d = (np.array(h100) - np.array(h15)) - (np.array(s100) - np.array(s15))
    boot = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"[Haiku gain - Sonnet gain] +{d.mean():.3f} "
          f"[{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"
          f"{'  CI excl 0 -> attenuation contrast SIGNIFICANT' if np.percentile(boot, 2.5) > 0 else ''}")
    print(f"absolute S(4,100): Sonnet {np.mean(s100):.3f} vs Haiku {np.mean(h100):.3f}")


if __name__ == "__main__":
    main()
