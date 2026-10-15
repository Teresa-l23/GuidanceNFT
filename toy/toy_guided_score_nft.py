"""One-update CFG transfer, score-policy NFT, and score-anchor diagnostics.

Outputs are exploratory diagnostics, not a claim that any update must improve.
The legacy detached-MAE NFT loss and a plain-MSE theory control are both run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

TOY_ROOT = Path(__file__).resolve().parent
# toy_example.py, dnnlib and torch_utils come from the patched EDM2 checkout (setup_upstream.sh).
sys.path.insert(0, str(Path(os.environ.get("EDM2_ROOT", TOY_ROOT / "third_party" / "edm2"))))
sys.path.insert(0, str(TOY_ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_example import do_sample, gt, tree_segments
from toy_nft_guidance import (
    ToyNFTConfig, _load_pickle_model, _toy_branch_loss,
    configured_branch_distance_reward, distance_to_segments, select_branch_segments,
)
from nft.math import advantage_to_reward_weight, nft_policy_loss_per_example

ASSETS = Path(os.environ.get("GUIDANCENFT_ASSETS", TOY_ROOT.parent / "assets"))


def frozen(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


class Field:
    """A scalar-potential field; only main can be optimized, guide stays frozen."""
    def __init__(self, main, guide=None, weight=1.0):
        self.main, self.guide, self.weight = main, guide, weight

    def score(self, x, sigma, graph=False):
        main = self.main.score(x, sigma, graph=graph)
        if self.guide is None:
            return main
        return self.weight * main + (1 - self.weight) * self.guide.score(x, sigma).detach()

    def logp(self, x, sigma):
        value = self.main.logp(x, sigma)
        if self.guide is not None:
            value = self.weight * value + (1 - self.weight) * self.guide.logp(x, sigma)
        return value


def rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def sample(field, initial, steps=32, sigma_max=5.0):
    """Keep full trajectories, with shared initial states supplied by the caller."""
    trajectory = do_sample(field, initial, num_steps=steps, sigma_max=sigma_max).detach()
    if not torch.isfinite(trajectory).all():
        raise FloatingPointError("nonfinite sampling trajectory")
    return trajectory


def reward_weights(y, config):
    rewards, distances = configured_branch_distance_reward(y, config)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    _, weights = advantage_to_reward_weight(advantages, 5.0)
    return rewards.detach(), distances.detach(), weights.detach()


@torch.no_grad()
def empirical(x, y, sigma, weights=None, chunk=128):
    """Exact Gaussian-mixture log density, score, ESS, and MSE NFT direction.

    The mixture is the finite empirical endpoint population, not an oracle for
    the unknown population. direction = 2 Cov(r,Y|x)/sigma**2 (beta=1).
    Stable float64 posterior calculations; query chunking limits memory.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 2 or y.shape[1] != 2:
        raise ValueError("x and y must have shape [N,2]")
    if sigma <= 0 or len(y) == 0 or chunk <= 0:
        raise ValueError("positive sigma, chunk and nonempty endpoints required")
    if weights is not None and weights.shape != (len(y),):
        raise ValueError("weights must match endpoints")
    yy = y.double()
    outputs = {k: [] for k in ("logp", "score", "ess", "direction")}
    for xx in x.double().split(chunk):
        logits = -(xx[:, None] - yy[None]).square().sum(-1) / (2 * sigma**2)
        a = logits.softmax(-1)
        mean = a @ yy
        outputs["logp"].append(torch.logsumexp(logits, -1) - np.log(len(y) * 2 * np.pi * sigma**2))
        outputs["score"].append((mean - xx) / sigma**2)
        outputs["ess"].append(a.square().sum(-1).reciprocal())
        direction = torch.zeros_like(mean)
        if weights is not None:
            aw = a * weights.double()[None]
            direction = 2 * (aw @ yy - aw.sum(-1, keepdim=True) * mean) / sigma**2
        outputs["direction"].append(direction)
    return {k: torch.cat(v).to(x.dtype) for k, v in outputs.items()}


def fit_score(base, endpoints, args, seed, output):
    """Fit a fresh A64 clone by DSM on frozen CFG endpoints, without rewards."""
    center = copy.deepcopy(base).eval().requires_grad_(True)
    optimizer = torch.optim.Adam(center.parameters(), lr=args.fit_lr, betas=(0.9, 0.99))
    generator = rng(endpoints.device, seed)
    history = []
    for step in range(args.fit_steps):
        index = torch.randint(len(endpoints), (args.batch,), device=endpoints.device, generator=generator)
        y = endpoints[index]
        sigma = (torch.randn(args.batch, device=y.device, generator=generator) * 1.5 - 2.3).exp().clamp(.002, 5)
        noise = torch.randn(y.shape, device=y.device, generator=generator)
        score = center.score(y + sigma[:, None] * noise, sigma, graph=True)
        loss = .5 * (sigma[:, None] * score + noise).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(center.parameters(), 5)
        if not torch.isfinite(norm) or not torch.isfinite(loss):
            raise FloatingPointError("nonfinite score fit")
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.fit_steps:
            row = {"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(norm)}
            history.append(row)
            print(json.dumps({"stage": "fit", **row}), flush=True)
    (output / "fit_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return center.requires_grad_(False)


def nft_records(y, seed, levels=8):
    """Re-noise a sealed buffer on an EDM rho=7 grid; no resampling during NFT."""
    generator = rng(y.device, seed)
    sigma_grid = torch.linspace(5**(1/7), .002**(1/7), levels + 1, device=y.device)[:-1] ** 7
    targets = y.repeat_interleave(levels, dim=0)
    sigma = sigma_grid.repeat(len(y))
    noise = torch.randn(targets.shape, device=y.device, generator=generator)
    return targets + sigma[:, None] * noise, sigma, targets


def branch_objective(current, old, center, y, weights, loss_kind):
    """beta=1; center is detached; use existing reward scaling and legacy loss."""
    positive = center.detach() + current - old.detach()
    negative = center.detach() - current + old.detach()
    if loss_kind == "legacy_mae":
        pos, neg = _toy_branch_loss(positive, y), _toy_branch_loss(negative, y)
    elif loss_kind == "mse":
        pos, neg = (positive - y).square().mean(-1), (negative - y).square().mean(-1)
    else:
        raise ValueError(loss_kind)
    return nft_policy_loss_per_example(pos, neg, weights, beta=1, advantage_clip=5)


def one_update(base_field, y, weights, args, seed, loss_kind, center=None):
    """Exactly one accumulated Adam update; optimize main only, never guide."""
    main = copy.deepcopy(base_field.main).eval().requires_grad_(True)
    current = Field(main, base_field.guide, base_field.weight)
    old = Field(frozen(base_field.main), base_field.guide, base_field.weight)
    anchor = old if center is None else center
    x, sigma, target = nft_records(y, seed)
    rw = weights.repeat_interleave(8)
    optimizer = torch.optim.Adam(main.parameters(), lr=args.nft_lr, betas=(.9, .99))
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for start in range(0, len(x), args.batch):
        xx, ss, yy = x[start:start+args.batch], sigma[start:start+args.batch], target[start:start+args.batch]
        ss2 = ss[:, None].square()
        prediction = xx + ss2 * current.score(xx, ss, graph=True)
        before = (xx + ss2 * old.score(xx, ss)).detach()
        middle = before if center is None else (xx + ss2 * anchor.score(xx, ss)).detach()
        loss = branch_objective(prediction, before, middle, yy, rw[start:start+args.batch], loss_kind)
        (loss.sum() / len(x)).backward()
        loss_sum += float(loss.detach().sum()) / len(x)
    norm = torch.nn.utils.clip_grad_norm_(main.parameters(), 5)
    if not torch.isfinite(norm) or not np.isfinite(loss_sum):
        raise FloatingPointError("nonfinite NFT update")
    optimizer.step()
    delta = sum(float((a.detach() - b).square().sum()) for a, b in zip(main.parameters(), old.main.parameters()))
    if base_field.guide is not None and any(p.grad is not None for p in base_field.guide.parameters()):
        raise RuntimeError("frozen guide received parameter gradients")
    return current, {"optimizer_steps": 1, "records": len(x), "loss": loss_sum,
                     "grad_norm": float(norm), "parameter_delta_l2": delta**.5}


def endpoint_metrics(y, config):
    rewards, distances, _ = reward_weights(y, config)
    segments = tree_segments("AB")
    starts = torch.tensor([s.start for s in segments], device=y.device)
    ends = torch.tensor([s.end for s in segments], device=y.device)
    tree_dist = torch.cat([distance_to_segments(c, starts, ends) for c in y.split(256)])
    return {"reward": float(rewards.mean()), "hit": float((distances <= .04).float().mean()),
            "on_tree": float((tree_dist <= .02).float().mean()), "samples": len(y)}


def alignment(a, b, mask):
    valid = mask & (a.norm(dim=-1) > 1e-9) & (b.norm(dim=-1) > 1e-9)
    if not valid.any():
        return {"count": 0, "cosine": None, "opposed_fraction": None, "rms": None}
    cosine = torch.nn.functional.cosine_similarity(a[valid], b[valid], dim=-1)
    return {"count": int(valid.sum()), "cosine": float(cosine.mean()),
            "opposed_fraction": float((cosine < 0).float().mean()),
            "rms": float(a[valid].square().mean().sqrt())}


def draw_tree(ax, config):
    for segment in tree_segments("AB"):
        ax.plot([segment.start[0], segment.end[0]], [segment.start[1], segment.end[1]],
                color="#d8b993" if segment.cls == "A" else "#d5d5d5", lw=.65, zorder=1)
    specs = ((config.branch_class, config.branch_path), (config.secondary_branch_class, config.secondary_branch_path))
    for cls, path in specs:
        if cls is None:
            continue
        for segment in select_branch_segments(cls, path, 3):
            ax.plot([segment.start[0], segment.end[0]], [segment.start[1], segment.end[1]], color="#0072b2", lw=1.4)
    ax.set(xlim=(-.9, 1.5), ylim=(-.9, 1.5), aspect="equal")
    ax.tick_params(labelsize=6)


def save_figure(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize(output, config, grid, arrays, methods, sigmas, metrics):
    """Use fixed queries/scales; each difference uses that branch's own baseline."""
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, methods):
        draw_tree(ax, config)
        points = arrays[f"{name}_endpoints"]
        ax.scatter(points[:, 0], points[:, 1], s=.7, c="#262626", alpha=.35, zorder=3)
        ax.set_title(f"{name}\nR={metrics[name]['reward']:.4f}; tree={metrics[name]['on_tree']:.3f}", fontsize=8)
    for ax in list(axes.flat)[len(methods):]:
        ax.axis("off")
    save_figure(fig, output, "endpoints")
    selected = ["strong_then_cfg", "strong_guided_buffer", "guided_naive", "guided_anchored", "score_guided_buffer", "score_native"]
    nx = len(np.unique(grid[:, 0]))
    for index, sigma in enumerate(sigmas):
        fig, axes = plt.subplots(3, len(selected), figsize=(19, 9), constrained_layout=True)
        density = arrays["cfg_empirical_logp"][index]
        mask = arrays["density_mask"][index]
        all_arrows = np.concatenate([arrays[f"{n}_delta_score"][index][mask] for n in selected])
        scale = max(float(np.quantile(np.linalg.norm(all_arrows, axis=-1), .95)) / .08, 1e-5) if len(all_arrows) else 1
        # Row 3 has its own shared scale: empirical endpoint-score change.
        empirical_arrows = np.concatenate([arrays[f"{n}_delta_endpoint_score"][index][mask] for n in selected])
        endpoint_scale = max(float(np.quantile(np.linalg.norm(empirical_arrows, axis=-1), .95)) / .08, 1e-5) if len(empirical_arrows) else 1
        sparse = mask.copy()
        sparse &= (np.arange(len(grid)) % 3 == 0)
        for col, name in enumerate(selected):
            for row, key in enumerate((f"{name}_score", f"{name}_delta_score", f"{name}_delta_endpoint_score")):
                ax = axes[row, col]
                ax.contourf(grid[:, 0].reshape(nx, nx), grid[:, 1].reshape(nx, nx), density.reshape(nx, nx), levels=np.linspace(density.max()-9, density.max(), 10), cmap="Blues", alpha=.45)
                draw_tree(ax, config)
                arrow = arrays[key][index]
                # Absolute score uses a common scale across all six methods.
                absolute = max(float(np.quantile(np.linalg.norm(arrays["cfg_score"][index][mask], axis=-1), .95)) / .08, 1e-5) if mask.any() else 1
                chosen_scale = absolute if row == 0 else scale if row == 1 else endpoint_scale
                ax.quiver(grid[sparse, 0], grid[sparse, 1], arrow[sparse, 0], arrow[sparse, 1], angles="xy", scale_units="xy", scale=chosen_scale, width=.004)
                ax.set_title(name if row == 0 else "", fontsize=8)
            axes[0, 0].set_ylabel("Updated sampler score")
            axes[1, 0].set_ylabel("Change in sampler score")
            axes[2, 0].set_ylabel("Change in re-noised endpoint score")
        fig.suptitle(f"sigma={sigma:g}; arrows are scores, not dx/dsigma; common scale within each row", fontsize=11)
        save_figure(fig, output, f"score_changes_sigma{sigma:g}")
        # Show the geometric reference explicitly, keeping its magnitude separate
        # from one finite Adam step. Direction-normalized arrows aid comparison.
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
        for ax, key, title in zip(axes,
                ("mse_ideal_direction", "strong_then_cfg_delta_score", "guided_anchored_delta_score", "score_guided_buffer_delta_score"),
                ("Empirical MSE ideal displacement", "NFT then CFG", "Score-anchored guided NFT", "NFT of fitted score")):
            draw_tree(ax, config)
            arrow = arrays[key][index].copy()
            norms = np.linalg.norm(arrow, axis=-1)
            valid = sparse & (norms > 1e-9)
            arrow /= np.maximum(norms[:, None], 1e-9)
            ax.quiver(grid[valid, 0], grid[valid, 1], arrow[valid, 0], arrow[valid, 1], angles="xy", scale_units="xy", scale=18, width=.004)
            ax.set_title(title, fontsize=9)
        fig.suptitle(f"Directions only (unit arrows), sigma={sigma:g}; empirical MSE reference is not the legacy-loss optimum", fontsize=10)
        save_figure(fig, output, f"improvement_directions_sigma{sigma:g}")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    index = 1 if len(sigmas) > 1 else 0
    for ax, key in zip(axes, ("cfg_score", "cfg_empirical_score", "anchor_gap")):
        draw_tree(ax, config)
        mask = arrays["density_mask"][index] & (np.arange(len(grid)) % 3 == 0)
        arrow = arrays[key][index]
        # Shared scale prevents exaggerating a smaller gap.
        scale = max(float(np.quantile(np.linalg.norm(arrays["cfg_score"][index], axis=-1), .95)) / .08, 1e-5)
        ax.quiver(grid[mask, 0], grid[mask, 1], arrow[mask, 0], arrow[mask, 1], angles="xy", scale_units="xy", scale=scale)
        ax.set_title(key, fontsize=9)
    fig.suptitle(f"Sampling score versus re-noised rollout score, sigma={sigmas[index]:g}")
    save_figure(fig, output, "anchor_gap")
    fig, axes = plt.subplots(1, len(selected), figsize=(18, 3), constrained_layout=True)
    for ax, name in zip(axes, selected):
        draw_tree(ax, config)
        for suffix, color in (("trajectory_before", "#999999"), ("trajectory", "#0072b2")):
            for path in arrays[f"{name}_{suffix}"][:, :32].transpose(1, 0, 2):
                ax.plot(path[:, 0], path[:, 1], color=color, alpha=.3, lw=.6)
        ax.set_title(name, fontsize=8)
    fig.suptitle("Paired full sampling trajectories; gray before, blue after; first 32 fixed sample IDs")
    save_figure(fig, output, "trajectories")


def run_seed(args, seed, root):
    output = root / f"seed-{seed}"
    output.mkdir()
    device = torch.device(args.device)
    torch.manual_seed(seed)
    config = ToyNFTConfig(guidance_scale=args.guidance, branch_path="L", branch_levels=3,
                          secondary_branch_class="B", secondary_branch_path="R", secondary_branch_levels=3,
                          reward="exponential_branch_distance", reward_distance_scale=.55, sampling_classes="AB")
    good = _load_pickle_model(str(ASSETS / "models/autoguidance-toy/clsA-layers04-dim64-iter4096.pkl"), device).requires_grad_(False)
    bad = _load_pickle_model(str(ASSETS / "models/autoguidance-toy/clsAB-layers04-dim32-iter0512.pkl"), device).requires_grad_(False)
    guide_state = {k: v.detach().clone() for k, v in bad.state_dict().items()}
    base, guided = Field(good), Field(good, bad, args.guidance)
    initial = lambda n, offset: gt("AB", device).sample(n, 5, generator=rng(device, seed+offset))
    fit_endpoints = sample(guided, initial(args.fit_samples, 10))[-1]
    center = fit_score(good, fit_endpoints, args, seed+20, output)
    score_field = Field(center)
    torch.save({"center": center.state_dict(), "fit_steps": args.fit_steps, "seed": seed}, output / "fitted_score.pt")
    # All buffers are independent of fit data; paired starts across populations.
    train_init = initial(args.nft_samples, 30)
    populations = {"base": sample(base, train_init)[-1], "cfg": sample(guided, train_init)[-1], "score": sample(score_field, train_init)[-1]}
    weights = {key: reward_weights(y, config)[2] for key, y in populations.items()}
    evaluation_init = initial(args.eval_samples, 40)
    initial_trajectories = {"base": sample(base, evaluation_init), "cfg": sample(guided, evaluation_init), "score": sample(score_field, evaluation_init)}
    # Independent mixture reference for convergence diagnostics and oracle arrows.
    reference = sample(guided, initial(args.reference_samples, 50))[-1]
    ref_weights = reward_weights(reference, config)[2]
    lin = torch.linspace(-.9, 1.5, args.grid, device=device)
    xx, yy = torch.meshgrid(lin, lin, indexing="xy")
    grid = torch.stack((xx.flatten(), yy.flatten()), -1)
    sigmas = [.03, .1, .3]
    common = {"grid": grid.cpu().numpy(), "sigmas": np.array(sigmas)}
    diagnostics = []
    shared = {k: [] for k in ("cfg_score", "cfg_empirical_score", "cfg_empirical_logp", "anchor_gap", "density_mask", "mse_ideal_direction")}
    for sigma in sigmas:
        ref = empirical(grid, reference, sigma, ref_weights)
        half = empirical(grid, reference[:len(reference)//2], sigma, ref_weights[:len(reference)//2])
        mask = (ref["logp"] >= ref["logp"].max()-7) & (ref["ess"] >= 8)
        current = guided.score(grid, sigma).detach()
        estimate = center.score(grid, sigma).detach()
        gap = current - ref["score"]
        for key, value in (("cfg_score", current), ("cfg_empirical_score", ref["score"]), ("cfg_empirical_logp", ref["logp"]), ("anchor_gap", gap), ("density_mask", mask), ("mse_ideal_direction", ref["direction"])):
            shared[key].append(value.cpu().numpy())
        denominator = ref["score"][mask].square().mean().clamp_min(1e-12) if mask.any() else torch.tensor(float("nan"))
        diagnostics.append({"sigma": sigma, "supported_grid_points": int(mask.sum()),
                            "fit_relative_rmse": float(((estimate[mask]-ref["score"][mask]).square().mean()/denominator).sqrt()) if mask.any() else None,
                            "initial_A64_relative_rmse": float(((base.score(grid, sigma).detach()[mask]-ref["score"][mask]).square().mean()/denominator).sqrt()) if mask.any() else None,
                            "guided_sampler_relative_rmse": float((gap[mask].square().mean()/denominator).sqrt()) if mask.any() else None,
                            "mixture_half_full_relative_rmse": float(((half["score"][mask]-ref["score"][mask]).square().mean()/denominator).sqrt()) if mask.any() else None})
    common.update({k: np.stack(v) for k, v in shared.items()})
    torch.save({"fit_endpoints": fit_endpoints.cpu(), "reference_endpoints": reference.cpu(),
                "train_populations": {k:v.cpu() for k,v in populations.items()},
                "reward_weights": {k:v.cpu() for k,v in weights.items()}, "evaluation_initial": evaluation_init.cpu()}, output / "buffers.pt")
    summaries = {}
    for loss_kind in args.losses:
        dest = output / loss_kind
        dest.mkdir()
        fields = {"base": base, "cfg": guided, "score": score_field}
        before = {key: key for key in fields}
        contracts = (
            ("strong", base, "base", None, "base"),
            ("strong_guided_buffer", base, "cfg", None, "cfg"),
            ("guided_naive", guided, "cfg", None, "cfg"),
            ("guided_anchored", guided, "cfg", score_field, "cfg"),
            ("score_guided_buffer", score_field, "cfg", None, "score"),
            ("score_native", score_field, "score", None, "score"),
        )
        updates = {}
        for name, field, population, anchor, baseline in contracts:
            trained, update = one_update(field, populations[population], weights[population], args, seed+60, loss_kind, anchor)
            fields[name] = Field(trained.main, bad, args.guidance) if name == "strong_guided_buffer" else trained
            before[name] = baseline
            updates[name] = {**update, "population": population, "anchor": "fitted_score" if anchor else "old_field"}
            print(json.dumps({"stage":"nft", "seed":seed, "loss_kind":loss_kind, "method":name, **update}), flush=True)
        fields["strong_then_cfg"] = Field(fields["strong"].main, bad, args.guidance)
        before["strong_then_cfg"] = "cfg"
        metrics, arrays, direction_metrics = {}, dict(common), {}
        base_empirical = {key: [empirical(grid, initial_trajectories[key][-1], s)["score"] for s in sigmas] for key in ("base", "cfg", "score")}
        for name, field in fields.items():
            trajectory = initial_trajectories[name] if name in initial_trajectories else sample(field, evaluation_init)
            metrics[name] = endpoint_metrics(trajectory[-1], config)
            arrays[f"{name}_endpoints"] = trajectory[-1].cpu().numpy()
            arrays[f"{name}_trajectory"] = trajectory[:, :64].cpu().numpy()
            arrays[f"{name}_trajectory_before"] = initial_trajectories[before[name]][:, :64].cpu().numpy()
            values = {k: [] for k in ("score", "potential", "delta_score", "endpoint_score", "delta_endpoint_score")}
            direction_metrics[name] = []
            baseline_field = fields[before[name]]
            for index, sigma in enumerate(sigmas):
                score = field.score(grid, sigma).detach()
                delta = score - baseline_field.score(grid, sigma).detach()
                endpoint_score = empirical(grid, trajectory[-1], sigma)["score"]
                endpoint_delta = endpoint_score - base_empirical[before[name]][index]
                for k, v in (("score", score), ("potential", field.logp(grid, sigma).detach()), ("delta_score", delta), ("endpoint_score", endpoint_score), ("delta_endpoint_score", endpoint_delta)):
                    values[k].append(v.cpu().numpy())
                oracle = torch.as_tensor(common["mse_ideal_direction"][index], device=device)
                mask = torch.as_tensor(common["density_mask"][index], device=device)
                direction_metrics[name].append({"sigma":sigma, **alignment(delta, oracle, mask)})
            arrays.update({f"{name}_{k}":np.stack(v) for k,v in values.items()})
        # Check CFG transfer amplification numerically at all fixed query points.
        transfer_error = float(np.max(np.abs(arrays["strong_then_cfg_delta_score"] - args.guidance * arrays["strong_delta_score"])))
        transfer_scale = float(np.max(np.abs(arrays["strong_then_cfg_delta_score"])))
        if transfer_error > 2e-4 * max(1, transfer_scale):
            raise AssertionError(f"CFG transfer identity failed: {transfer_error}")
        if any(not torch.equal(v, guide_state[k]) for k,v in bad.state_dict().items()):
            raise AssertionError("AB32 guide mutated")
        pairwise = {}
        for other in ("guided_naive", "guided_anchored", "score_guided_buffer", "strong_guided_buffer"):
            pairwise[other] = []
            for index, sigma in enumerate(sigmas):
                transfer = torch.as_tensor(arrays["strong_then_cfg_delta_score"][index], device=device)
                direct = torch.as_tensor(arrays[f"{other}_delta_score"][index], device=device)
                mask = torch.as_tensor(common["density_mask"][index], device=device)
                pairwise[other].append({"sigma":sigma, **alignment(transfer, direct, mask),
                                       "difference_rms":float((transfer[mask]-direct[mask]).square().mean().sqrt()) if mask.any() else None})
        np.savez_compressed(dest / "fields_and_samples.npz", **arrays)
        torch.save({name: field.main.state_dict() for name,field in fields.items()}, dest / "policies.pt")
        summary = {"seed":seed, "loss":loss_kind, "guidance":args.guidance, "updates":updates,
                   "metrics":metrics, "fit_diagnostics":diagnostics, "direction_metrics":direction_metrics,
                   "transfer_vs_direct":pairwise,
                   "transfer_identity_max_abs_error":transfer_error, "guide_unchanged":True,
                   "oracle_interpretation":"Finite empirical-population MSE displacement, beta=1; not an exact optimum of legacy MAE normalization or one Adam step."}
        (dest / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        visualize(dest, config, common["grid"], arrays, list(fields), sigmas, metrics)
        summaries[loss_kind] = summary
    (output / ".complete").write_text("all losses complete\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260925, 20260926, 20260927])
    parser.add_argument("--guidance", type=float, default=4)
    parser.add_argument("--fit-steps", type=int, default=2000)
    parser.add_argument("--fit-samples", type=int, default=8192)
    parser.add_argument("--reference-samples", type=int, default=8192)
    parser.add_argument("--nft-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--fit-lr", type=float, default=.001)
    parser.add_argument("--nft-lr", type=float, default=.0002)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--losses", nargs="+", choices=("legacy_mae", "mse"), default=["legacy_mae", "mse"])
    args = parser.parse_args()
    if any(getattr(args,k) <= 0 for k in ("guidance","fit_steps","fit_samples","reference_samples","nft_samples","eval_samples","fit_lr","nft_lr","batch","grid")):
        parser.error("all numeric settings must be positive")
    if args.reference_samples < 2 or args.grid < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("reference/grid >=2 and unique seeds required")
    if not args.output.resolve().is_relative_to(ASSETS.resolve()):
        parser.error("persistent experiment outputs must be under group assets")
    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["toy_source_root"] = str(TOY_ROOT)
    config["status"] = "exploratory_one_optimizer_update"
    config["source_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), TOY_ROOT/"toy_example.py", TOY_ROOT/"toy_nft_guidance.py", TOY_ROOT/"nft/math.py")}
    (args.output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    results = {str(seed):run_seed(args, seed, args.output) for seed in args.seeds}
    (args.output/"summary.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    (args.output/".complete").write_text("all seeds complete\n")
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
