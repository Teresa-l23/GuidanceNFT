# 2D tree toy

The toy figures of the paper, built on the Autoguidance 2D tree example (EDM2).

* `toy_nft_guidance.py`, `toy_nft_guided_vs_separate_comparison.py` and helpers: NFT on the
  tree with a branch-distance reward, comparing a separately trained model, joint training
  on the guided field, and joint training of paired branches (Figure 1).
* `toy_cfg4_iterated_push.py`: oracle iterated NFT updates under CFG 4 for each loss
  center (ideal, NFT, conditional, and their bias-only variants), recording the reference
  error fields (method figure and appendix).
* `plot_toy_cfg4_bias_only_fields.py` and `plot_figure1_samplers.py`: draw the figure panels.

## Run

```bash
export GUIDANCENFT_ASSETS=/path/to/assets
./setup_upstream.sh                     # EDM2 @ 4bf8162 + patch, NVIDIA toy models
scripts/reproduce_paper_figures.sh      # about 15 minutes on one GPU
```

Outputs go to `$GUIDANCENFT_ASSETS/runs/toy/<timestamp>/figures/`:
`toy_reference_fields.pdf` is the appendix figure (its statistics are in
`toy_reference_fields.json`), `toy_reference_fields_bottom.pdf` the method figure, and
`figure1_samplers_zoom.pdf` Figure 1 (drawn by `plot_figure1_samplers.py`). GPU nondeterminism changes
the reported statistics by at most about 0.6% relative to the paper; round-0 quantities
agree to 1e-6.

The toy models are the released Autoguidance toy networks
(`clsA-layers04-dim64/iter4096` and `clsAB-layers04-dim32/iter0512`), downloaded from
NVIDIA's CDN by `setup_upstream.sh`. `toy_example.py` and the models are licensed under
CC BY-NC-SA 4.0, and so is `patches/toy_example.patch`.
