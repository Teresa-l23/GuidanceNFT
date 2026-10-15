"""Iterate the oracle NFT step on the CFG4 tree toy and split it into reward and center terms.

Each round rolls out the current sampler field, rebuilds the re-noised rollout
mixture (v_q, v_+, alpha = E[r | x]) and adds, in score units,

    R = 2 alpha (s_+ - s_q)             reward-driven term, the same for every center
    B = (2 alpha - 1) (s_q - C)         center-induced term

to the sampler field, tabulated on a per-noise-level grid. Arms differ only in
the center C and in which term they keep. No network is trained.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_guided_score_nft import (
    configured_branch_distance_reward, endpoint_metrics, gt, rng, sample, select_branch_segments, tree_segments,
)
from plot_toy_cfg4_analytic_fields import (
    ASSETS, INK, TARGET, Field, _load_pickle_model, bin_population, draw_tree, reward_config, reward_weights,
)

# arm -> (kept terms, center, label, color)
ARMS = {
    "ideal": ("R", None, r"ideal center $\bar v_q$ (reward term only)", "#1a1a1a"),
    "std": ("RB", "cur", r"center = sampler field (standard NFT)", "#d55e00"),
    "cond": ("RB", "cond", r"center $v_c$ (A64 branch)", "#0072b2"),
    "std_bias": ("B", "cur", r"center term only, center = sampler field", "#e69f00"),
    "cond_bias": ("B", "cond", r"center term only, center $v_c$", "#56b4e9"),
    # Policy = conditional branch: the push moves v_c and the CFG sampler w v_c - (w-1) v_u moves by w times the push.
    "vc_ideal": ("R", None, r"ideal center $\bar v_q$", "#1a1a1a"),
    "vc_std": ("RB", "vc", r"center = current $v_c$", "#d55e00"),
    "vc_bias": ("B", "vc", r"center term only (no reward)", "#0072b2"),
}
ORIGIN = np.array([.6, .7])
DIAG_LEVELS = (15, 20, 24)


class TableField:
    """score = base score + bilinear lookup of the accumulated push at this noise level."""
    def __init__(self, base, table, sigmas, half, origin):
        self.base, self.table, self.sigmas, self.half, self.origin = base, table, sigmas, half, origin

    def score(self, x, sigma, graph=False):
        score = self.base.score(x, sigma).detach()
        level = int((self.sigmas - float(sigma)).abs().argmin())
        u = ((x - self.origin) / self.half[level])[None, :, None, :]
        add = torch.nn.functional.grid_sample(self.table[level].permute(2, 0, 1)[None], u, mode="bilinear",
                                              padding_mode="zeros", align_corners=True)
        return score + add[0, :, :, 0].T


@torch.no_grad()
def mixture(x, bins, sigma):
    """E[Y|x], E[r|x], E[rY|x] and log-density (up to a constant) of the re-noised mixture; x, Y origin-centered."""
    chunk = max(256, int(2e8 // len(bins["r"])))
    parts = []
    for xx in x.split(chunk):
        d2 = (xx[:, :1] - bins["y"][None, :, 0]).square() + (xx[:, 1:] - bins["y"][None, :, 1]).square()
        logits = bins["log_n"][None] - d2 / (2 * sigma**2)
        lse = torch.logsumexp(logits, -1)
        a = (logits - lse[:, None]).exp()
        parts.append((a @ bins["y"], (a @ bins["r"])[:, None], a @ bins["ry"], lse))
    return [torch.cat(p) for p in zip(*parts)]


def network_scores(field, x, sigma, chunk=65536):
    return torch.cat([field.score(c, sigma).detach() for c in x.split(chunk)])


def weighted_rms(value, weight):
    return float((value.square().sum(-1) * weight).sum().sqrt())


def effective_cells(y, width=.02):
    cell = torch.floor(y.cpu().double() / width).long()
    p = torch.unique(cell, dim=0, return_counts=True)[1].double()
    p = p / p.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def run(args):
    kept, center_kind, _, _ = ARMS[args.arm]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"arm": args.arm, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}), flush=True)
    good = _load_pickle_model(str(ASSETS / "models/autoguidance-toy/clsA-layers04-dim64-iter4096.pkl"), device).requires_grad_(False)
    bad = _load_pickle_model(str(ASSETS / "models/autoguidance-toy/clsAB-layers04-dim32-iter0512.pkl"), device).requires_grad_(False)
    guided, cond, config = Field(good, bad, args.guidance), Field(good), reward_config(args.guidance)
    amplify = args.guidance if args.arm.startswith("vc_") else 1.  # sampler change per unit policy change

    sigmas = (5 ** (1 / 7) + torch.arange(32) / 31 * (.002 ** (1 / 7) - 5 ** (1 / 7))) ** 7
    half = (1. + 4 * sigmas).to(device)
    origin = torch.tensor(ORIGIN, dtype=torch.float32, device=device)
    n = args.resolution
    lin = torch.linspace(-1, 1, n, device=device)
    xx, yy = torch.meshgrid(lin, lin, indexing="xy")
    unit_grid = torch.stack((xx.flatten(), yy.flatten()), -1)
    active = [level for level in range(32) if float(sigmas[level]) >= args.sigma_floor]
    grids = {level: unit_grid * half[level] for level in active}  # origin-centered
    base = {level: network_scores(guided, grids[level] + origin, float(sigmas[level])) for level in active}
    cond_grid = {level: network_scores(cond, grids[level] + origin, float(sigmas[level])) for level in active} if center_kind in ("cond", "vc") or args.arm == "ideal" else {}
    table = torch.zeros(32, n, n, 2, device=device)
    field = TableField(guided, table, sigmas, half, origin)

    initial = torch.cat([gt("AB", device).sample(min(32768, args.population - i), 5, generator=rng(device, args.seed + i))
                         for i in range(0, args.population, 32768)])
    history, shown, fields0, snaps = [], [], {}, {}
    for round_index in range(args.rounds + 1):
        y = torch.cat([sample(field, c)[-1] for c in initial.split(32768)])
        row = {"arm": args.arm, "round": round_index, **endpoint_metrics(y, config), "effective_cells": effective_cells(y)}
        shown.append(y[:args.keep].cpu().numpy())
        snap = round_index in args.snapshot_rounds
        if round_index < args.rounds or snap:
            _, weights = reward_weights(y.cpu(), config, args.advantage_clip)
            binned = bin_population(y.cpu(), weights, args.bin_width)
            origin64 = origin.cpu().double()
            bins = {"log_n": binned["n"].log().float().to(device), "y": (binned["y"] - origin64).float().to(device),
                    "r": binned["r"].float().to(device),
                    "ry": (binned["ry"] - binned["r"][:, None] * origin64).float().to(device)}
            row["bins"] = len(binned["n"])
            for level in (active if round_index < args.rounds else DIAG_LEVELS):
                sigma, x = float(sigmas[level]), grids[level]
                mean, alpha, ry, lse = mixture(x, bins, sigma)
                s_q = (mean - x) / sigma**2
                reward = 2 * (ry - alpha * mean) / sigma**2
                current = base[level] + table[level].reshape(-1, 2)
                if center_kind == "vc":  # current conditional branch = v_c + accumulated policy push
                    center = cond_grid[level] + table[level].reshape(-1, 2) / amplify
                else:
                    center = current if center_kind == "cur" else cond_grid[level] if center_kind == "cond" else s_q
                bias = (2 * alpha - 1) * (s_q - center)
                if level in DIAG_LEVELS:
                    weight = torch.softmax(lse, 0)
                    would_be = (2 * alpha - 1) * (s_q - current)  # standard-NFT center term at the current field
                    row[f"gap_{level}"] = weighted_rms(s_q - current, weight) / weighted_rms(s_q, weight)
                    row[f"bias_over_reward_{level}"] = weighted_rms(would_be, weight) / weighted_rms(reward, weight)
                    if round_index == 0 and args.arm == "ideal":  # round 0 is shared by every arm
                        fields0[level] = {"sigma": sigma, "half": float(half[level]), "logp": lse.cpu().numpy(),
                                          "reward": (sigma * reward).cpu().numpy(),
                                          "bias_std": (sigma * (2 * alpha - 1) * (s_q - current)).cpu().numpy(),
                                          "bias_cond": (sigma * (2 * alpha - 1) * (s_q - cond_grid[level])).cpu().numpy()}
                if snap and level in DIAG_LEVELS:  # velocities v = sigma * score on the level grid
                    snaps.update({f"s{round_index}_{level}_{k}": v.cpu().numpy() for k, v in (
                        ("logp", lse - torch.logsumexp(bins["log_n"], 0)), ("alpha", alpha[:, 0]), ("vq", sigma * s_q),
                        ("cur", sigma * current), ("center", sigma * center), ("bias", sigma * bias), ("reward", sigma * reward))})
                    snaps.update({f"s{round_index}_{level}_sigma": sigma, f"s{round_index}_{level}_half": float(half[level])})
                if round_index < args.rounds:
                    push = (reward if "R" in kept else 0) + (bias if "B" in kept else 0)
                    table[level] += args.eta * amplify * push.reshape(n, n, 2)
            if not torch.isfinite(table).all():
                raise FloatingPointError("nonfinite push table")
        history.append(row)
        print(json.dumps(row), flush=True)

    np.savez_compressed(args.output / f"arm_{args.arm}.npz", endpoints=np.stack(shown),
                        **{f"f{level}_{k}": v for level, d in fields0.items() for k, v in d.items()}, **snaps)
    (args.output / f"arm_{args.arm}.json").write_text(json.dumps({"config": {k: str(v) for k, v in vars(args).items()}, "history": history}, indent=2, allow_nan=False) + "\n")
    print("COMPLETE", args.arm, flush=True)


def plot(args):
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    config, view, radius = reward_config(args.guidance), np.array([.56, .72]), .82
    arms = [a for a in (args.arms or ARMS) if (args.output / f"arm_{a}.json").is_file()]
    history = {a: json.loads((args.output / f"arm_{a}.json").read_text())["history"] for a in arms}
    data = {a: np.load(args.output / f"arm_{a}.npz") for a in arms}
    rounds = len(history[arms[0]]) - 1
    columns = sorted({0, 1, rounds // 4, rounds // 2, rounds})

    # One figure: endpoints per arm and round (two tones only), then three curves.
    fig = plt.figure(figsize=(2.7 * len(columns), 2.75 * len(arms) + 3.3))
    outer = fig.add_gridspec(2, 1, height_ratios=(2.75 * len(arms), 2.7), left=.035, right=.985, top=.965, bottom=.07, hspace=.11)
    top = outer[0].subgridspec(len(arms), len(columns), wspace=.03, hspace=.12)
    tree = [np.array([segment.start, segment.end]) for segment in tree_segments("AB")]
    target = [np.array([segment.start, segment.end]) for segment in select_branch_segments(config.branch_class, config.branch_path, 3)]
    for i, arm in enumerate(arms):
        for j, k in enumerate(columns):
            ax, y, m = fig.add_subplot(top[i, j]), data[arm]["endpoints"][k], history[arm][k]
            for line in target:
                ax.plot(*line.T, color=TARGET, lw=6, alpha=.28, solid_capstyle="round", zorder=1)
            for line in tree:
                ax.plot(*line.T, color="#b5b5b5", lw=.45, zorder=2)
            hit = (configured_branch_distance_reward(torch.from_numpy(y), config)[1] <= .04).numpy()
            ax.scatter(*y[~hit].T, s=1.6, c="#a3a3a3", alpha=.55, linewidths=0, zorder=3, rasterized=True)
            ax.scatter(*y[hit].T, s=1.6, c=INK, alpha=.55, linewidths=0, zorder=4, rasterized=True)
            ax.set(xlim=(view[0] - radius, view[0] + radius), ylim=(view[1] - radius, view[1] + radius), aspect="equal")
            ax.set_xticks([]), ax.set_yticks([])
            for side in ax.spines.values():
                side.set_edgecolor("#cfcfcf"), side.set_linewidth(.8)
            if i == 0:
                ax.set_title(f"round {k}", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(ARMS[arm][2], fontsize=9.5, color=ARMS[arm][3], labelpad=5)
            ax.text(.035, .035, f"on target {m['hit']:.2f}\ndiversity {m['effective_cells']:.0f}", transform=ax.transAxes,
                    va="bottom", fontsize=8.5, color="#333333", zorder=6)
    bottom = outer[1].subgridspec(1, 3, wspace=.22)
    panels = (("hit", "fraction on target"), ("effective_cells", "diversity (effective occupied cells)"), ("on_tree", "fraction on the tree"))
    for position, (key, title) in enumerate(panels):
        ax = fig.add_subplot(bottom[0, position])
        for arm in arms:
            values = [row[key] for row in history[arm]]
            ax.plot(range(len(values)), values, color=ARMS[arm][3], lw=2, marker="o", ms=3.5, label=ARMS[arm][2])
        ax.set_title(title, fontsize=10, color=INK), ax.set_xlabel("round")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", color="#ececec", lw=.6), ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if position == 1:
            ax.legend(frameon=False, fontsize=8.5)
    fig.text(.035, .012, f"Black: endpoints within 0.04 of the target branches (purple band); gray: other endpoints. Same initial noise in every panel. "
             f"CFG scale {args.guidance:g}, step size {args.eta:g}, {rounds} oracle rounds.", fontsize=8.5, color="#555555")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"iterated_summary.{suffix}", dpi=150)
    plt.close(fig)

    if "ideal" in data and f"f{DIAG_LEVELS[0]}_reward" in data["ideal"]:
        d = data["ideal"]
        n = int(np.sqrt(len(d[f"f{DIAG_LEVELS[0]}_logp"])))
        index = np.arange(n * n)
        unit = np.stack(np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="xy"), -1).reshape(-1, 2)
        fig, axes = plt.subplots(2, len(DIAG_LEVELS), figsize=(5.4 * len(DIAG_LEVELS), 11))
        fig.subplots_adjust(left=.01, right=.99, top=.92, bottom=.03, wspace=.04, hspace=.1)
        for j, level in enumerate(DIAG_LEVELS):
            grid, logp = ORIGIN + unit * float(d[f"f{level}_half"]), d[f"f{level}_logp"]
            stride = max(1, round(2 * radius / 26 / (2 * float(d[f"f{level}_half"]) / n)))  # about 26 arrows across the view
            inside = (np.abs(grid - view) < radius).all(-1)
            keep = inside & (logp >= logp[inside].max() - 4) & ((index % n) % stride == 0) & ((index // n) % stride == 0)
            reward = d[f"f{level}_reward"]
            scale = np.quantile(np.linalg.norm(reward[keep], axis=1), .95) / (2 * float(d[f"f{level}_half"]) / n * stride) / 1.5
            for i, (key, name, color) in enumerate((("bias_std", "center = sampler field (standard NFT)", ARMS["std"][3]),
                                                    ("bias_cond", r"center $v_c$", ARMS["cond"][3]))):
                ax, bias = axes[i, j], d[f"f{level}_{key}"]
                draw_tree(ax, config, view, radius)
                for vectors, c in ((reward, INK), (bias, color)):
                    ax.quiver(grid[keep, 0], grid[keep, 1], vectors[keep, 0], vectors[keep, 1], angles="xy", scale_units="xy",
                              scale=scale, color=c, width=.003, headwidth=3.6, headlength=4, headaxislength=3.5, minlength=0, zorder=4)
                ratio = np.sqrt(np.square(bias[keep]).sum() / np.square(reward[keep]).sum())
                ax.set_title(f"{name}\n$\\sigma$ = {float(d[f'f{level}_sigma']):.3f}: center term = {ratio:.2f} × reward term", fontsize=10, loc="left")
        fig.suptitle("Round 0, same CFG4 field: reward term R (black) and center term B (color), same arrow scale", fontsize=12.5)
        for suffix in ("png", "pdf"):
            fig.savefig(args.output / f"round0_terms.{suffix}", dpi=140)
        plt.close(fig)
    print("COMPLETE plot", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), help="arms to show with --plot (default: all present)")
    parser.add_argument("--guidance", type=float, default=4., help="CFG scale of the initial sampler")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--eta", type=float, default=1.)
    parser.add_argument("--advantage-clip", type=float, default=2)
    parser.add_argument("--population", type=int, default=65536)
    parser.add_argument("--keep", type=int, default=4096, help="endpoints stored per round for plotting")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--sigma-floor", type=float, default=.02)
    parser.add_argument("--bin-width", type=float, default=.003)
    parser.add_argument("--snapshot-rounds", type=int, nargs="*", default=[], help="rounds whose fields are stored at DIAG_LEVELS")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ASSETS.resolve()):
        parser.error("persistent output must be under group assets")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    plot(args) if args.plot else run(args)


if __name__ == "__main__":
    main()
