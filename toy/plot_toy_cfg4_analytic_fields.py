"""Initial-state CFG4 tree toy: constructed fields versus re-noised rollout fields.

No training. The CFG4 rollout endpoint law q0 has no closed form, so its
re-noised score is evaluated as the exact Gaussian-mixture score of a large
endpoint population (binned to centroids; error O((bin/sigma)^2)). The tilted
law is q+ ∝ r q0 with r the NFT optimality weight. Arrows are denoising
velocities v = sigma * score = (D - x) / sigma = -dx/dsigma.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_guided_score_nft import (
    ASSETS, Field, ToyNFTConfig, _load_pickle_model, configured_branch_distance_reward, rng, sample,
    select_branch_segments, tree_segments, gt,
)

RUN = ASSETS / "runs/autoguidance_nft/toy_cfg4_score_nft_one_step/slurm-56410342"
INK, MUTED, TARGET = "#1a1a1a", "#8a8a8a", "#9063ae"
CENTERS = (("cond", r"$v_c$ (A64 branch)", "#0072b2"),
           ("cfg", r"$v_{\mathrm{cfg4}}=4v_c-3v_u$", "#d55e00"),
           ("fit", r"$v_\phi$ (fitted center)", "#009e73"))


def reward_config(guidance):
    return ToyNFTConfig(guidance_scale=guidance, branch_path="L", branch_levels=3,
                        secondary_branch_class="B", secondary_branch_path="R", secondary_branch_levels=3,
                        reward="exponential_branch_distance", reward_distance_scale=.55, sampling_classes="AB")


def reward_weights(y, config, clip):
    """NFT optimality weight r in [0,1]; a smaller advantage clip gives a stronger tilt."""
    rewards, _ = configured_branch_distance_reward(y, config)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    return rewards, (advantages.clamp(-clip, clip) / clip / 2 + .5).clamp(0, 1)


def rollout_population(guided, count, seed, cache, chunk=32768):
    """Sample or reload CFG rollout endpoints; starts are GT(AB) at sigma_max=5."""
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["count"] == count and saved["seed"] == seed:
            return saved["endpoints"]
    parts = []
    for index in range(0, count, chunk):
        size = min(chunk, count - index)
        start = gt("AB", torch.device("cpu")).sample(size, 5, generator=rng("cpu", seed + index))
        parts.append(sample(guided, start)[-1])
        print(json.dumps({"stage": "rollout", "done": index + size, "of": count}), flush=True)
    endpoints = torch.cat(parts)
    torch.save({"endpoints": endpoints, "count": count, "seed": seed}, cache)
    return endpoints


def bin_population(y, r, width):
    """Compress endpoints to per-bin count, centroid and mean r, mean r*y (float64)."""
    y, r = y.double(), r.double()
    cell = torch.floor(y / width).long()
    _, inverse, counts = torch.unique(cell, dim=0, return_inverse=True, return_counts=True)
    n = counts.double()
    centroid = torch.zeros(len(n), 2, dtype=torch.float64).index_add_(0, inverse, y) / n[:, None]
    r_mean = torch.zeros(len(n), dtype=torch.float64).index_add_(0, inverse, r) / n
    ry_mean = torch.zeros(len(n), 2, dtype=torch.float64).index_add_(0, inverse, r[:, None] * y) / n[:, None]
    return {"n": n, "y": centroid, "r": r_mean, "ry": ry_mean, "total": float(len(y))}


@torch.no_grad()
def mixture_fields(x, bins, sigma, chunk=512):
    """Re-noised mixture quantities at x [N,2]: velocities of q, q+, q-, alpha, logp, ESS.

    v_q = (E[Y|x]-x)/sigma, v_+ = (E[rY|x]/E[r|x]-x)/sigma, and likewise v_-
    with 1-r; alpha = E[r|x], so v_q = alpha v_+ + (1-alpha) v_-.
    """
    out = {k: [] for k in ("v_q", "v_pos", "v_neg", "alpha", "logp", "ess")}
    log_n = bins["n"].log()
    for xx in x.double().split(chunk):
        logits = log_n[None] - (xx[:, None] - bins["y"][None]).square().sum(-1) / (2 * sigma**2)
        a = logits.softmax(-1)
        mean, alpha, ry = a @ bins["y"], a @ bins["r"], a @ bins["ry"]
        pos = ry / alpha[:, None].clamp_min(1e-12)
        neg = (mean - ry) / (1 - alpha[:, None]).clamp_min(1e-12)
        out["v_q"].append((mean - xx) / sigma)
        out["v_pos"].append((pos - xx) / sigma)
        out["v_neg"].append((neg - xx) / sigma)
        out["alpha"].append(alpha)
        out["logp"].append(torch.logsumexp(logits, -1) - np.log(bins["total"] * 2 * np.pi * sigma**2))
        out["ess"].append(1 / (a.square() / bins["n"][None]).sum(-1))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def model_fields(fields, x, sigma, chunk=4096):
    """Constructed model velocities sigma*score at x, float64 numpy."""
    x = x.float()
    return {name: np.concatenate([(sigma * field.score(c, sigma).detach()).double().numpy() for c in x.split(chunk)])
            for name, field in fields.items()}


def relative_rms(gap, reference):
    return float(np.sqrt(np.square(gap).sum(-1).mean() / np.square(reference).sum(-1).mean()))


def mean_cosine(a, b):
    return float(((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)).mean())


def branch_view(cls, path):
    segments = select_branch_segments(cls, path, 3)
    vertices = np.array([p for s in segments for p in (s.start, s.end)])
    center = (vertices.min(0) + vertices.max(0)) / 2
    return center, float(np.ptp(vertices, axis=0).max() / 2 + .09)


def draw_tree(ax, config, center, radius):
    for segment in tree_segments("AB"):
        ax.plot(*np.array([segment.start, segment.end]).T, color=MUTED, lw=.5, alpha=.7, zorder=2)
    for cls, path in ((config.branch_class, config.branch_path), (config.secondary_branch_class, config.secondary_branch_path)):
        for segment in select_branch_segments(cls, path, 3):
            ax.plot(*np.array([segment.start, segment.end]).T, color=TARGET, lw=1.5, zorder=2)
    ax.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius), aspect="equal")
    ax.set_xticks([]), ax.set_yticks([])


def quiver(ax, grid, vectors, keep, unit, spacing, label):
    """True-length arrows: a vector of norm `unit` spans one arrow-grid spacing."""
    q = ax.quiver(grid[keep, 0], grid[keep, 1], vectors[keep, 0], vectors[keep, 1], angles="xy",
                  scale_units="xy", scale=unit / spacing, color=INK, width=.0032, headwidth=3.6,
                  headlength=4, headaxislength=3.5, minlength=0, zorder=4)
    ax.quiverkey(q, .80, -.045, unit, f"{unit:.2g} {label}", labelpos="E", coordinates="axes",
                 fontproperties={"size": 7})


def field_figure(output, name, config, center, radius, sigma, fields, bins, metrics, resolution=96, stride=4):
    """Row 1: absolute velocities. Row 2: center minus analytic rollout field. Row 3: v_+ minus each center."""
    lin_x = torch.linspace(center[0] - radius, center[0] + radius, resolution, dtype=torch.float64)
    lin_y = torch.linspace(center[1] - radius, center[1] + radius, resolution, dtype=torch.float64)
    xx, yy = torch.meshgrid(lin_x, lin_y, indexing="xy")
    grid_t = torch.stack((xx.flatten(), yy.flatten()), -1)
    grid = grid_t.numpy()
    mix = mixture_fields(grid_t, bins, sigma)
    model = model_fields(fields, grid_t, sigma)
    support = (mix["logp"] >= mix["logp"].max() - 6) & (mix["ess"] >= 50) & (mix["alpha"] > 1e-3)
    index = np.arange(len(grid))
    sparse = support & ((index % resolution) % stride == stride // 2) & ((index // resolution) % stride == stride // 2)
    spacing = 2 * radius / resolution * stride
    shape = (resolution, resolution)
    density = np.where(support, mix["logp"], np.nan).reshape(shape)

    absolute = (("cond", model["cond"]), ("cfg", model["cfg"]), ("q", mix["v_q"]), ("pos", mix["v_pos"]))
    titles = {"cond": r"$v_c$: conditional branch (A64)", "cfg": r"$v_{\mathrm{cfg4}}=4v_c-3v_u$: constructed CFG4",
              "q": r"$\bar v_q$: analytic field of the CFG4 rollout law", "pos": r"$v_+$: analytic field of the tilted law $q_+\propto r\,q$"}
    gaps = (("cond", model["cond"] - mix["v_q"]), ("cfg", model["cfg"] - mix["v_q"]),
            ("fit", model["fit"] - mix["v_q"]), ("pos", mix["v_pos"] - mix["v_q"]))
    gap_titles = {"cond": r"$v_c-\bar v_q$", "cfg": r"$v_{\mathrm{cfg4}}-\bar v_q$",
                  "fit": r"$v_\phi-\bar v_q$ (fitted center)", "pos": r"$v_+-\bar v_q$ (reward signal)"}

    signal = mix["v_pos"] - mix["v_q"]
    targets = (("cond", mix["v_pos"] - model["cond"]), ("cfg", mix["v_pos"] - model["cfg"]),
               ("fit", mix["v_pos"] - model["fit"]), ("q", signal))
    target_titles = {"cond": r"$v_+-v_c$", "cfg": r"$v_+-v_{\mathrm{cfg4}}$", "fit": r"$v_+-v_\phi$",
                     "q": r"$v_+-\bar v_q$: true reward direction"}

    def heat_arrows(ax, value, unit_value, cmap):
        norm = np.where(support, np.linalg.norm(value, axis=1), np.nan).reshape(shape)
        image = ax.pcolormesh(xx, yy, norm, cmap=cmap, vmin=0, vmax=unit_value, alpha=.8, shading="auto", zorder=1, rasterized=True)
        draw_tree(ax, config, center, radius)
        quiver(ax, grid, value, sparse, unit_value, spacing, "velocity")
        return image

    def note(ax, text):
        ax.text(.03, .97, text, transform=ax.transAxes, va="top", fontsize=9.5, color=INK,
                bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=2.5), zorder=6)

    def colorbar(ax, image, label):
        box = ax.get_position()
        cax = fig.add_axes([box.x0 + .02 * box.width, box.y0 - .019, .55 * box.width, .006])
        bar = fig.colorbar(image, cax=cax, orientation="horizontal")
        bar.ax.tick_params(labelsize=7)
        bar.set_label(label, fontsize=7.5)

    fig, axes = plt.subplots(3, 4, figsize=(16.5, 14.6))
    fig.subplots_adjust(left=.012, right=.988, top=.92, bottom=.085, wspace=.04, hspace=.33)
    unit = float(np.quantile(np.linalg.norm(mix["v_q"][support], axis=1), .95))
    for ax, (key, value) in zip(axes[0], absolute):
        ax.pcolormesh(xx, yy, density, cmap="Greys", alpha=.35, shading="auto", zorder=1, rasterized=True)
        draw_tree(ax, config, center, radius)
        quiver(ax, grid, value, sparse, unit, spacing, "velocity")
        ax.set_title(titles[key], fontsize=10.5)
    center_norms = np.concatenate([np.linalg.norm(v[support], axis=1) for k, v in gaps if k != "pos"])
    center_unit = float(np.quantile(center_norms, .95))
    tilt_unit = float(np.quantile(np.linalg.norm(signal[support], axis=1), .95))
    for ax, (key, value) in zip(axes[1, :3], gaps[:3]):
        image = heat_arrows(ax, value, center_unit, "Oranges")
        ax.set_title(gap_titles[key], fontsize=10.5)
        note(ax, f"relative RMS = {metrics[key]:.2f}")
    colorbar(axes[1, 2], image, "center-gap norm (shared by the three panels)")
    ax = axes[1, 3]
    alpha = np.where(support, mix["alpha"], np.nan).reshape(shape)
    image = ax.pcolormesh(xx, yy, alpha, cmap="Greens", alpha=.85, shading="auto", zorder=1, rasterized=True)
    draw_tree(ax, config, center, radius)
    ax.set_title(r"$\alpha(x)=E[r\mid x]$: posterior optimality weight", fontsize=10.5)
    colorbar(ax, image, "α (reward signal needs α to vary within ~σ)")
    for ax, (key, value) in zip(axes[2], targets):
        if key == "q":
            image = heat_arrows(ax, value, tilt_unit, "Blues")
            note(ax, f"relative RMS = {metrics['pos']:.2f}")
            colorbar(ax, image, "norm (own scale)")
        else:
            image = heat_arrows(ax, value, center_unit, "Oranges")
            note(ax, f"cosine with true reward direction = {metrics['cos_' + key]:+.2f}")
        ax.set_title(target_titles[key], fontsize=10.5)
    colorbar(axes[2, 2], axes[2, 2].collections[0], "norm (same scale as the center gaps)")
    fig.suptitle(f"CFG4 tree toy at the initial state, σ = {sigma:g}   ({name} view; {metrics['tilt_label']})", fontsize=14, y=.985)
    fig.text(.5, .952, "Arrows are denoising velocities v = σ·score at true relative length (one scale per group; see keys). "
             "Gray shade: log density of the re-noised CFG4 rollout law; purple: reward branches. "
             "Arrows shown only where the rollout law has support.", ha="center", fontsize=9, color="#444444")
    fig.text(.012, .008, "Row 2: center minus the analytic rollout field. Row 3: where the positive branch v+ points from each center "
             f"(cosine averaged over x ~ re-noised rollout law). Shared center scale {center_unit:.2g} per grid step; true reward direction uses {tilt_unit:.2g}.",
             fontsize=8.5, color="#444444")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fields_{name}_sigma{sigma:g}.{suffix}", dpi=170)
    plt.close(fig)
    return {"absolute_unit": unit, "center_gap_unit": center_unit, "tilt_unit": tilt_unit, "supported": int(support.sum())}


def sweep_figure(output, rows):
    """Relative RMS of each center gap, and of the reward signal, across noise levels."""
    sigma = np.array([row["sigma"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for key, label, color in CENTERS:
        value = [row["relative_rms"][key] for row in rows]
        ax.plot(sigma, value, color=color, lw=2, marker="o", ms=4, label=label)
        ax.annotate(label, (sigma[-1], value[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    value = [row["relative_rms"]["pos"] for row in rows]
    ax.plot(sigma, value, color=INK, lw=1.5, ls="--", marker="s", ms=3.5, label=r"$v_+-\bar v_q$ (reward signal)")
    ax.annotate(r"$v_+$ (reward signal)", (sigma[-1], value[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(sigma.min() * .85, sigma.max() * 6)
    ax.set_xlabel("noise level σ"), ax.set_ylabel(r"RMS$\,\|\,\cdot-\bar v_q\|$ / RMS$\,\|\bar v_q\|$")
    ax.set_title("Gap to the analytic CFG4 rollout field, initial state", fontsize=11)
    ax.grid(True, which="major", color="#e6e6e6", lw=.6), ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"gap_vs_sigma.{suffix}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guidance", type=float, default=4)
    parser.add_argument("--population", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--fitted", type=Path, default=RUN / "seed-20260925/fitted_score.pt")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[.03, .1, .3])
    parser.add_argument("--sweep", type=float, nargs="+", default=[.02, .03, .05, .1, .2, .3, .5, 1, 2, 5])
    parser.add_argument("--probes", type=int, default=4096)
    parser.add_argument("--bin-width", type=float, default=.003)
    parser.add_argument("--advantage-clip", type=float, default=5, help="r = clip(adv, ±c)/(2c) + 0.5; the saved NFT runs use 5")
    parser.add_argument("--endpoints", type=Path, help="reuse a cached rollout_endpoints.pt from another output")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ASSETS.resolve()):
        parser.error("persistent output must be under group assets")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    device = torch.device("cpu")
    good = _load_pickle_model(str(ASSETS / "models/autoguidance-toy/clsA-layers04-dim64-iter4096.pkl"), device).requires_grad_(False)
    bad = _load_pickle_model(str(ASSETS / "models/autoguidance-toy/clsAB-layers04-dim32-iter0512.pkl"), device).requires_grad_(False)
    center = copy.deepcopy(good)
    center.load_state_dict(torch.load(args.fitted, map_location="cpu", weights_only=True)["center"])
    fields = {"cond": Field(good), "uncond": Field(bad), "cfg": Field(good, bad, args.guidance),
              "fit": Field(center.eval().requires_grad_(False))}
    config = reward_config(args.guidance)
    endpoints = rollout_population(fields["cfg"], args.population, args.seed, args.endpoints or args.output / "rollout_endpoints.pt")
    rewards, weights = reward_weights(endpoints, config, args.advantage_clip)
    bins = bin_population(endpoints, weights, args.bin_width)
    half = bin_population(endpoints[::2], weights[::2], args.bin_width)
    print(json.dumps({"stage": "population", "endpoints": len(endpoints), "bins": len(bins["n"]),
                      "reward_mean": float(rewards.mean()), "weight_mean": float(weights.mean()),
                      "weight_min": float(weights.min()), "weight_max": float(weights.max())}), flush=True)

    # Probes x = y + sigma*eps follow the re-noised rollout law, i.e. the NFT training states.
    generator = torch.Generator().manual_seed(args.seed + 1)
    probe_y = endpoints[torch.randperm(len(endpoints), generator=generator)[:args.probes]].double()
    noise = torch.randn(probe_y.shape, generator=generator, dtype=torch.float64)
    rows = []
    for sigma in sorted(set(args.sweep) | set(args.sigmas)):
        x = probe_y + sigma * noise
        mix, mix_half, model = mixture_fields(x, bins, sigma), mixture_fields(x, half, sigma), model_fields(fields, x, sigma)
        # v_+ is undefined where the posterior weight vanishes; drop those probes.
        keep = (mix["alpha"] > 1e-3) & (mix_half["alpha"] > 1e-3)
        mix, mix_half, model = ({k: v[keep] for k, v in d.items()} for d in (mix, mix_half, model))
        rows.append({"sigma": sigma, "relative_rms": {
            "cond": relative_rms(model["cond"] - mix["v_q"], mix["v_q"]),
            "cfg": relative_rms(model["cfg"] - mix["v_q"], mix["v_q"]),
            "fit": relative_rms(model["fit"] - mix["v_q"], mix["v_q"]),
            "pos": relative_rms(mix["v_pos"] - mix["v_q"], mix["v_q"]),
            **{f"cos_{k}": mean_cosine(mix["v_pos"] - model[k], mix["v_pos"] - mix["v_q"]) for k in ("cond", "cfg", "fit")},
            "half_population_v_q": relative_rms(mix_half["v_q"] - mix["v_q"], mix["v_q"]),
            "half_population_v_pos_gap": relative_rms(mix_half["v_pos"] - mix_half["v_q"] - mix["v_pos"] + mix["v_q"], mix["v_pos"] - mix["v_q"])},
            "identity_max_abs_error": float(np.abs(mix["alpha"][:, None] * mix["v_pos"] + (1 - mix["alpha"][:, None]) * mix["v_neg"] - mix["v_q"])[mix["alpha"] < 1 - 1e-3].max()),
            "alpha_mean": float(mix["alpha"].mean()), "probes_kept": int(keep.sum()), "ess_median": float(np.median(mix["ess"]))})
        print(json.dumps({"stage": "sweep", **rows[-1]}), flush=True)
    sweep_figure(args.output, [row for row in rows if row["sigma"] in set(args.sweep)])

    views = {"full": (np.array([.56, .72]), .82), "branchAL": branch_view("A", "L")}
    pages = []
    for sigma in args.sigmas:
        metrics = {**next(row for row in rows if row["sigma"] == sigma)["relative_rms"],
                   "tilt_label": f"advantage clip {args.advantage_clip:g}, r ∈ [{float(weights.min()):.2f}, {float(weights.max()):.2f}]"}
        for name, (view_center, radius) in views.items():
            detail = field_figure(args.output, name, config, view_center, radius, sigma, fields, bins, metrics)
            pages.append({"view": name, "sigma": sigma, **detail})
            print(json.dumps({"stage": "figure", **pages[-1]}), flush=True)
    summary = {"status": "exploratory_visualization_no_training", "guidance": args.guidance, "population": len(endpoints),
               "seed": args.seed, "bin_width": args.bin_width, "bins": len(bins["n"]), "probes": args.probes,
               "fitted_center": str(args.fitted), "velocity_definition": "v = sigma * score = (D - x) / sigma",
               "reference": "exact Gaussian-mixture score of the binned finite CFG rollout population; not a closed form",
               "advantage_clip": args.advantage_clip,
               "tilt": "q+ ∝ r q0, r = clip((R-mean)/(std+1e-4), ±c)/(2c) + 0.5 from population statistics",
               "sweep": rows, "pages": pages}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
