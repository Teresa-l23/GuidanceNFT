"""Reference-error arms of the iterated oracle NFT step, drawn with the Autoguidance toy palette.

Reads arm_<name>.npz written by toy_cfg4_iterated_push.py with
--snapshot-rounds. No sampling and no training. Velocities are v = sigma * score.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from toy_guided_score_nft import ASSETS, Field, _load_pickle_model, gt, select_branch_segments
from plot_toy_cfg4_analytic_fields import reward_config
from toy_cfg4_iterated_push import ORIGIN

# arm, applied terms (R = Delta, B = (1 - 2 alpha) e), row label
ROWS = (("ideal", "R", r"$v_{\rm ref}=\bar v_q$:  $\Delta$"),
        ("std_bias", "B", r"$v_{\rm ref}=v_w$:  $(1-2\alpha)e$"),
        ("std", "RB", r"$v_{\rm ref}=v_w$:  $\Delta+(1-2\alpha)e$"),
        ("cond_bias", "B", r"$v_{\rm ref}=v_c$:  $(1-2\alpha)e$"),
        ("cond", "RB", r"$v_{\rm ref}=v_c$:  $\Delta+(1-2\alpha)e$"))
VIEW, RADIUS = np.array([.56, .72]), .82
TOP_VIEW, TOP_RADIUS = np.array([.40, .46]), 1.03  # the base model of Figure 1 covers the whole class-A tree
CONTEXT, CONTEXT_EDGE, TREE, TREE_EDGE, TARGET_BLUE, POINT = "#f1f1f1", "#d2d2d2", "#D9EBF4", "#8DB8CD", "#4095C5", "#262626"  # paper Figure 1, target #0072B2 tinted 25% towards white


def contours(ax, extent, values, levels, cmap, top, **kwargs):
    """Autoguidance `contours`: filled bands plus faint black isolines; only the light part [0, top] of the colormap."""
    pale = matplotlib.colors.LinearSegmentedColormap.from_list(f"{cmap}_pale", plt.get_cmap(cmap)(np.linspace(0, top, 256)))
    ax.contourf(*extent, values, levels=levels, cmap=pale, extend="max", antialiased=True, **kwargs)
    ax.contour(*extent, values, levels=levels, colors="black", alpha=.14, linewidths=.4, linestyles="solid", **kwargs)


def arrows(ax, grid, vectors, keep, color, step, unit):
    """Arrows of length |v| / unit grid steps, clipped at 1.6 steps."""
    v = vectors[keep] / unit
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    v = v * np.minimum(1, 1.6 / np.maximum(norm, 1e-12)) * step
    ax.quiver(grid[keep, 0], grid[keep, 1], v[:, 0], v[:, 1], angles="xy", scale_units="xy", scale=1, color=color,
              width=5e-3, headwidth=4, headlength=3, headaxislength=2.5, minlength=0, zorder=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="figure stem (default: <run>/bias_only_fields)")
    parser.add_argument("--level", type=int, default=20, help="noise level index (15, 20, 24 = sigma 0.317, 0.093, 0.029)")
    parser.add_argument("--rounds", type=int, nargs=2, default=(0, 10), help="before / after rounds")
    parser.add_argument("--figure1-data", type=Path, help="evaluation dir of the Figure 1 toy; adds <stem>_combined, <stem>_top and <stem>_bottom")
    parser.add_argument("--top-arrows", action="store_true", help="draw sampler velocities in the Figure 1 row")
    parser.add_argument("--top-filled", action="store_true", help="Figure 1 row as the filled tree with endpoints over a paler density")
    parser.add_argument("--top-points", type=int, default=256, help="stored endpoints drawn per panel in the Figure 1 row (the first ones; 0 = none)")
    parser.add_argument("--arms", nargs="+", choices=[row[0] for row in ROWS], help="rows to draw (default: all)")
    args = parser.parse_args()
    stem = args.output or args.run / "bias_only_fields"
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 14, "font.family": "DejaVu Sans"})
    rows = [row for row in ROWS if not args.arms or row[0] in args.arms]
    config, (r0, r1), level = reward_config(4.), args.rounds, args.level
    data = {arm: np.load(args.run / f"arm_{arm}.npz") for arm, *_ in rows}
    history = {arm: json.loads((args.run / f"arm_{arm}.json").read_text())["history"] for arm, *_ in rows}
    get = lambda arm, r, key: data[arm][f"s{r}_{level}_{key}"]
    sigma, half = float(get(rows[0][0], r0, "sigma")), float(get(rows[0][0], r0, "half"))
    n = int(np.sqrt(len(get(rows[0][0], r0, "logp"))))
    lin = np.linspace(-1, 1, n) * half
    extent = (ORIGIN[0] + lin, ORIGIN[1] + lin)
    grid = np.stack(np.meshgrid(*extent, indexing="xy"), -1).reshape(-1, 2)
    index = np.arange(n * n)
    stride = max(1, round(2 * RADIUS / 22 / (2 * half / (n - 1))))
    step = stride * 2 * half / (n - 1)
    lattice = ((index % n) % stride == 0) & ((index // n) % stride == 0) & (np.abs(grid - VIEW) < RADIUS).all(-1)

    # Ground-truth tree regions as in the paper's Figure 1 (Autoguidance: -sqrt(max logp - logp) >= -2.12).
    # The reward is the distance to the union of two subtrees; the class-B one lies mostly outside the view.
    target = (select_branch_segments(config.branch_class, config.branch_path, config.branch_levels)
              + select_branch_segments(config.secondary_branch_class, config.secondary_branch_path, config.secondary_branch_levels))

    def scene(view, radius, m=400):
        tx, ty = view[0] + np.linspace(-radius, radius, m), view[1] + np.linspace(-radius, radius, m)
        txy = torch.tensor(np.stack(np.meshgrid(tx, ty, indexing="xy"), -1), dtype=torch.float32).reshape(-1, 2)
        tree = {}
        with torch.no_grad():
            for classes in ("AB", "A"):
                logp = torch.cat([gt(classes, torch.device("cpu")).logp(chunk) for chunk in txy.split(1024)]).reshape(m, m)
                tree[classes] = -(logp.max() - logp).clamp_min(0).sqrt().numpy()
        return {"view": view, "radius": radius, "tx": tx, "ty": ty, "txy": txy, "tree": tree}

    main_scene = scene(VIEW, RADIUS)

    def backdrop(ax, filled, where=None):
        where = where or main_scene
        tx, ty, tree, view, radius = (where[k] for k in ("tx", "ty", "tree", "view", "radius"))
        if filled:
            for classes, fill, edge in (("AB", CONTEXT, CONTEXT_EDGE), ("A", TREE, TREE_EDGE)):
                ax.contourf(tx, ty, tree[classes], levels=[-2.12, 0], colors=[fill], extend="max", antialiased=True, zorder=1)
                ax.contour(tx, ty, tree[classes], levels=[-2.12], colors=[edge], linewidths=.45, linestyles="solid", zorder=2)
            for segment in target:
                ax.plot(*np.array([segment.start, segment.end]).T, color=TARGET_BLUE, lw=3.4 * .8 ** segment.depth,
                        solid_capstyle="round", zorder=3)
        else:
            ax.contour(tx, ty, tree["AB"], levels=[-2.12], colors="black", alpha=.35, linewidths=.5, linestyles="solid", zorder=4)
            for segment in target:  # rewarded branches, with a white halo so they read on every shading
                line, width = np.array([segment.start, segment.end]).T, 3.4 * .8 ** segment.depth
                ax.plot(*line, color="white", lw=width + 2.2, solid_capstyle="round", zorder=4.2)
                ax.plot(*line, color=TARGET_BLUE, lw=width, solid_capstyle="round", zorder=4.4)
        ax.set(xlim=(view[0] - radius, view[0] + radius), ylim=(view[1] - radius, view[1] + radius), aspect="equal")
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_linewidth(.6)

    inside = (np.abs(grid - VIEW) < RADIUS).all(-1)
    support = lambda logp: logp >= logp[inside].max() - 4

    def unit(vectors, logp):
        """Rollout-density-weighted RMS norm over the view: arrows of this norm are drawn 0.8 grid steps long."""
        weight = np.exp(logp[inside] - logp[inside].max())
        return float(np.sqrt((np.square(vectors[inside]).sum(-1) * weight).sum() / weight.sum())) / .8

    gap0 = {arm: get(arm, r0, "center") - get(arm, r0, "vq") for arm, *_ in rows}
    keep0 = {arm: support(get(arm, r0, "logp")) for arm, *_ in rows}
    applied = {arm: sum(get(arm, r0, key) for key, term in (("reward", "R"), ("bias", "B")) if term in kept) for arm, kept, _ in rows}
    gap_top = max(np.quantile(np.linalg.norm(gap0[arm][keep0[arm] & inside], axis=1), .95) for arm, *_ in rows)
    units = {"center": max(unit(get(arm, r0, "center"), get(arm, r0, "logp")) for arm, *_ in rows),
             "applied": max(unit(applied[arm], get(arm, r0, "logp")) for arm, *_ in rows)}

    titles = (r"(a) Reference $v_{\rm ref}$", r"(b) Error $\|e\|$ and first update",
              rf"(c) Field after {r1} rounds", rf"(d) Samples after {r1} rounds")
    fig, axes = plt.subplots(len(rows), 4, figsize=(13.6, 3.36 * len(rows) + .45), squeeze=False)
    fig.subplots_adjust(left=.035, right=.995, top=1 - .42 / (3.36 * len(rows) + .45), bottom=.004, wspace=.03, hspace=.03)
    summary = {"sigma": sigma, "level": level, "rounds": [r0, r1], "arrow_unit_per_grid_step": units, "gap_color_max": float(gap_top), "rows": {}}
    density = lambda logp, keep: np.where(keep, logp - logp[keep & inside].max(), np.nan).reshape(n, n)
    for i, (arm, kept, label) in enumerate(rows):
        logp0, logp1 = get(arm, r0, "logp"), get(arm, r1, "logp")
        keep1 = support(logp1)
        ax = axes[i, 0]  # the reference field at the first round, on the re-noised rollout density
        contours(ax, extent, density(logp0, keep0[arm]), np.linspace(-4, 0, 16)[:-1], "Reds", .45)
        arrows(ax, grid, get(arm, r0, "center"), keep0[arm] & lattice, "C3", step, units["center"])
        backdrop(ax, False)
        ax.set_ylabel(label, fontsize=15)

        ax = axes[i, 1]  # reference error at the first round
        norm = np.where(keep0[arm], np.linalg.norm(gap0[arm], axis=1), np.nan).reshape(n, n)
        contours(ax, extent, norm, np.linspace(0, gap_top, 16)[1:], "Blues", .8)
        # A row whose update is much smaller than the shared unit is magnified by an integer factor, stated in the panel.
        gain = max(1, int(units["applied"] / unit(applied[arm], logp0) / 1.5))
        arrows(ax, grid, applied[arm] * gain, keep0[arm] & lattice, "C0", step, units["applied"])
        if gain > 1:
            ax.text(.97, .03, rf"arrows $\times{gain}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=13,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=2))
        backdrop(ax, False)

        ax = axes[i, 2]  # sampler field after the bias-only rounds, on its own re-noised rollout density
        contours(ax, extent, density(logp1, keep1), np.linspace(-4, 0, 16)[:-1], "Greens", .5)
        # The pushed field can be orders of magnitude larger off the rollout support, so this panel uses its own median unit.
        units[f"cur_{arm}"] = float(np.median(np.linalg.norm(get(arm, r1, "cur")[keep1 & lattice], axis=1))) / .8
        arrows(ax, grid, get(arm, r1, "cur"), keep1 & lattice, "C2", step, units[f"cur_{arm}"])
        backdrop(ax, False)

        ax = axes[i, 3]  # endpoints
        backdrop(ax, True)
        ax.scatter(*data[arm]["endpoints"][r1].T, s=2.6, c=POINT, alpha=.7, linewidths=0, zorder=6, rasterized=True)
        h0, h1 = history[arm][r0], history[arm][r1]
        ax.text(.03, .03, f"reward {h1['reward']:.2f}\nhit {h1['hit']:.2f}\non-tree {h1['on_tree']:.2f}\ncells {h1['effective_cells']:.0f}",
                transform=ax.transAxes, va="bottom", fontsize=13, zorder=8, bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=2))
        weight = np.exp(logp0 - logp0.max())
        rms = lambda v: float(np.sqrt((np.square(v).sum(-1) * weight).sum() / weight.sum()))
        summary["rows"][arm] = {"relative_rms_gap_round0": rms(gap0[arm]) / rms(get(arm, r0, "vq")),
                                "rms_bias_round0": rms(get(arm, r0, "bias")), "rms_delta_round0": rms(get(arm, r0, "reward")),
                                "arrow_gain_b": gain,
                                **{f"{k}_round{r}": history[arm][r][k] for r in (r0, r1) for k in ("reward", "hit", "on_tree", "effective_cells")}}
        if i == 0:
            for ax, title in zip(axes[0], titles):
                ax.set_title(title, fontsize=16, pad=7)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{stem}.{suffix}", dpi=200)
    plt.close(fig)

    # Main-text strip: the two fields that keep Delta (green), then the error of v_c and the field it alone produces (blue).
    if {"ideal", "cond_bias", "cond"} <= set(data):
        def field_panel(ax, arm, title, cmap, color, top=.5, size=17):
            logp1 = get(arm, r1, "logp")
            keep1 = support(logp1)
            contours(ax, extent, density(logp1, keep1), np.linspace(-4, 0, 16)[:-1], cmap, top)
            arrows(ax, grid, get(arm, r1, "cur"), keep1 & lattice, color, step, units[f"cur_{arm}"])
            backdrop(ax, False)
            ax.set_title(title, fontsize=size, pad=7)
            h = history[arm][r1]
            ax.text(.03, .03, f"hit {h['hit']:.2f}\ncells {h['effective_cells']:.0f}", transform=ax.transAxes, va="bottom", fontsize=14,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=2))

        def strip(axes, letters, size=17):
            """Sampler fields after the rounds (green) and first-round displacements (blue); five axes add Delta itself."""
            axes, letters = list(axes), list(letters)
            field_panel(axes.pop(0), "ideal", rf"({letters.pop(0)}) Matched reference", "Greens", "C2", size=size)
            if len(axes) == 4:  # the intended update in the first round over the whole view, on its own arrow scale
                ax, delta = axes.pop(0), applied["ideal"]
                norm = np.linalg.norm(delta, axis=1)
                # magnitudes above the 97% quantile share the last, darkest shade
                contours(ax, extent, norm.reshape(n, n), np.linspace(0, np.quantile(norm[inside], .97), 25)[1:], "Blues", .8)
                arrows(ax, grid, delta, lattice, "C0", step, unit(delta, get("ideal", r0, "logp")))
                backdrop(ax, False)
                ax.set_title(rf"({letters.pop(0)}) Intended update $\Delta$", fontsize=size, pad=7)
            field_panel(axes[0], "cond", rf"({letters[0]}) $v_{{\rm ref}}=v_c$: $\Delta+(1-2\alpha)e$", "Greens", "C2", size=size)
            ax = axes[1]
            norm = np.where(keep0["cond_bias"], np.linalg.norm(gap0["cond_bias"], axis=1), np.nan).reshape(n, n)
            contours(ax, extent, norm, np.linspace(0, gap_top, 16)[1:], "Blues", .8)
            arrows(ax, grid, applied["cond_bias"], keep0["cond_bias"] & lattice, "C0", step, units["applied"])
            backdrop(ax, False)
            ax.set_title(rf"({letters[1]}) Error $\|e\|$ and $(1-2\alpha)e$", fontsize=size, pad=7)
            field_panel(axes[2], "cond_bias", rf"({letters[2]}) $(1-2\alpha)e$ only", "Greens", "C2", size=size)

        paper = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 15}  # manuscript typography
        with plt.rc_context(paper):
            fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.82))
            fig.subplots_adjust(left=.004, right=.996, top=.89, bottom=.008, wspace=.03)
            strip(axes, "abcd")
            for suffix in ("png", "pdf"):
                fig.savefig(f"{stem}_main.{suffix}", dpi=200)
            plt.close(fig)

        # Teaser: the trained-network comparison of Figure 1 (stored endpoints, unchanged) above the strip.
        if args.figure1_data:
            with np.load(args.figure1_data / "guided_vs_separate_data.npz", allow_pickle=False) as saved:
                groups, methods = saved["groups"].tolist(), saved["methods"].tolist()
                steps = saved["outer_steps"].tolist()
                pick = lambda final, group, method: saved["endpoints"][steps.index(max(steps) if final else 0), groups.index(group), methods.index(method)]
                top = [("(a) Base", pick(False, "separate_nft", "a64")), ("(b) CFG", pick(False, "separate_nft", "a64_init_ab32_w3")),
                       ("(c) NFT", pick(True, "separate_nft", "a64")), ("(d) NFT + CFG", pick(True, "separate_nft", "a64_init_ab32_w3")),
                       ("(e) Update guided field", pick(True, "joint_guided_field", "a64_trained_ab32_w3"))]
            # The same five samplers as fields: pretrained pair, and the evaluated (EMA) weights of the final checkpoints.
            cpu, training = torch.device("cpu"), args.figure1_data.parent / "training"
            good = _load_pickle_model(str(ASSETS / "models/autoguidance-toy/clsA-layers04-dim64-iter4096.pkl"), cpu).requires_grad_(False)
            bad = _load_pickle_model(str(ASSETS / "models/autoguidance-toy/clsAB-layers04-dim32-iter0512.pkl"), cpu).requires_grad_(False)

            def trained(template, run, key):
                model = copy.deepcopy(template)
                model.load_state_dict(torch.load(training / run / "checkpoint.pt", map_location=cpu, weights_only=False)["models"][key], strict=True)
                return model.eval().requires_grad_(False)

            nft, w = trained(good, "a64_nft", "good_ema"), 3.
            fields = [Field(good), Field(good, bad, w), Field(nft), Field(nft, bad, w),
                      Field(trained(good, "joint_guided_field", "good_ema"), trained(bad, "joint_guided_field", "bad_ema"), w)]
            where = scene(TOP_VIEW, TOP_RADIUS)
            side = (where["tx"], where["ty"])
            m = len(side[0])
            skip = max(1, round(m / 22))
            cells = np.arange(m * m)
            coarse = ((cells % m) % skip == skip // 2) & ((cells // m) % skip == skip // 2)
            spacing = skip * 2 * TOP_RADIUS / (m - 1)
            drawn = []
            for field, (title, points) in zip(fields, top):  # endpoints re-noised to the same sigma as the bottom row
                d2 = torch.cdist(where["txy"], torch.from_numpy(points).float()).square()
                logp = torch.logsumexp(-d2 / (2 * sigma**2), 1).numpy()
                keep = (logp >= logp.max() - 4) & coarse
                velocity = np.zeros((m * m, 2), dtype=np.float32)
                velocity[keep] = sigma * field.score(where["txy"][torch.from_numpy(keep)], sigma).detach().numpy()
                drawn.append((title, points, logp, keep, velocity))
            top_unit = float(np.median(np.concatenate([np.linalg.norm(v[k], axis=1) for *_, k, v in drawn]))) / .8

            def top_row(axes, size=15):
                for ax, (title, points, logp, keep, velocity) in zip(axes, drawn):
                    shown = np.where(logp >= logp.max() - 4, logp - logp.max(), np.nan).reshape(m, m)
                    if args.top_filled:  # a paler density, behind the filled tree
                        contours(ax, side, shown, np.linspace(-4, 0, 16)[:-1], "Blues", .58, zorder=.5)
                    else:
                        contours(ax, side, shown, np.linspace(-4, 0, 16)[:-1], "Greens", .55)
                    if args.top_arrows:
                        arrows(ax, where["txy"].numpy(), velocity, keep, "C2", spacing, top_unit)
                    if args.top_points:
                        ax.scatter(*points[:args.top_points].T, c=POINT, linewidths=0, rasterized=True,
                                   **(dict(s=4, alpha=.75, zorder=6) if args.top_filled else dict(s=2.6, alpha=.55, zorder=4.1)))
                    backdrop(ax, args.top_filled, where)
                    ax.set_title(title, fontsize=size, pad=6)

            with plt.rc_context(paper):
                fig, axes = plt.subplots(2, 5, figsize=(13.6, 6.12))
                fig.subplots_adjust(left=.004, right=.996, top=.945, bottom=.004, wspace=.03, hspace=.125)
                top_row(axes[0])
                strip(axes[1], "fghij", size=15)
                for suffix in ("png", "pdf"):
                    fig.savefig(f"{stem}_combined.{suffix}", dpi=200)
                plt.close(fig)

                # The two rows as separate figures, each with its own legend below the panels.
                blues, greens = plt.get_cmap("Blues"), plt.get_cmap("Greens")
                line = lambda color, label, **kw: Line2D([], [], color=color, label=label, **kw)
                arrow = lambda color, label: line(color, label, lw=0, marker=r"$\rightarrow$", markersize=20)
                target_key = line(TARGET_BLUE, "High-reward branches", lw=3.2, solid_capstyle="round")
                top_keys = ([Patch(facecolor=TREE, edgecolor=TREE_EDGE, label="Data distribution"), target_key,
                             Patch(facecolor=blues(.4), edgecolor="none", label="Density of re-noised samples"),
                             line(POINT, "Samples", lw=0, marker="o", markersize=5)] if args.top_filled else
                            [line("#9a9a9a", "Data distribution", lw=1), target_key,
                             Patch(facecolor=greens(.4), edgecolor="none", label="Density of re-noised samples"),
                             line(POINT, "Samples", lw=0, marker="o", markersize=4)])
                bottom_keys = [line("#9a9a9a", "Data distribution", lw=1), target_key,
                               Patch(facecolor=greens(.4), edgecolor="none", label="Rollout density"),
                               arrow("C2", "Velocity"),
                               Patch(facecolor=blues(.5), edgecolor="none", label="Magnitude"), arrow("C0", "Displacement")]
                for name, keys, draw in (("top", top_keys, top_row), ("bottom", bottom_keys, lambda axes, size: strip(axes, "abcde", size=size))):
                    fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.5))
                    fig.subplots_adjust(left=.004, right=.996, top=.9, bottom=.125, wspace=.03)
                    draw(axes, size=17)
                    fig.legend(handles=keys, loc="lower center", ncol=len(keys), frameon=False, fontsize=16, handlelength=1.6,
                               handletextpad=.5, columnspacing=1.5, borderaxespad=.1)
                    for suffix in ("png", "pdf"):
                        fig.savefig(f"{stem}_{name}.{suffix}", dpi=200)
                    plt.close(fig)
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
