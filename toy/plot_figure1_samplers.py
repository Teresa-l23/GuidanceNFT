"""Figure 1 sampler panels. Reads the stored endpoints only; no sampling, no training.

Usage: plot_figure1_samplers.py <evaluation dir> <output dir> thin
(`thin` gives the paper version, figure1_samplers_zoom.pdf.)
"""
import sys
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_guided_score_nft import gt, select_branch_segments, tree_segments
from plot_toy_cfg4_analytic_fields import reward_config

THIN = len(sys.argv) > 3; LEVEL = 0. if THIN else -2.12
D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SIGMA = 0.09283874183893204
GREY, GREY_EDGE, TREE, TREE_EDGE = "#f1f1f1", "#d2d2d2", "#EFEFEF", "#C8C8C8"
TARGET, TARGET_EDGE, ON, OFF, POINT = "#D9EBF4", "#8DB8CD", "#262626", "#D55E00", "#262626"
FULL = (np.array([.40, .46]), 1.03)
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})

with np.load(D / "guided_vs_separate_data.npz", allow_pickle=False) as z:
    g, m, steps = z["groups"].tolist(), z["methods"].tolist(), z["outer_steps"].tolist()
    def pick(final, group, method):
        i = (steps.index(max(steps)) if final else 0, g.index(group), m.index(method))
        return z["endpoints"][i], z["full_tree_distance"][i] <= .02, z["union_distance"][i] <= .04
    PANELS = [("(a) Base", *pick(False, "separate_nft", "a64")), ("(b) CFG", *pick(False, "separate_nft", "a64_init_ab32_w3")),
              ("(c) NFT", *pick(True, "separate_nft", "a64")), ("(d) NFT + CFG", *pick(True, "separate_nft", "a64_init_ab32_w3")),
              ("(e) Update guided field", *pick(True, "joint_guided_field", "a64_trained_ab32_w3"))]
config = reward_config(4.)
target = select_branch_segments(config.branch_class, config.branch_path, config.branch_levels)
ends = np.array([p for s in target for p in (s.start, s.end)])
c = (ends.min(0) + ends.max(0)) / 2
ZOOM = (c - np.array([0, .05]), float((ends.max(0) - ends.min(0)).max()) / 2 + .19)
print("zoom", ZOOM)

def scene(view, radius, n=800):
    tx, ty = view[0] + np.linspace(-radius, radius, n), view[1] + np.linspace(-radius, radius, n)
    txy = torch.tensor(np.stack(np.meshgrid(tx, ty, indexing="xy"), -1), dtype=torch.float32).reshape(-1, 2)
    tree = {}
    with torch.no_grad():
        for cls in () if THIN else ("AB", "A"):
            logp = torch.cat([gt(cls, torch.device("cpu")).logp(ch) for ch in txy.split(1024)]).reshape(n, n)
            tree[cls] = -(logp.max() - logp).clamp_min(0).sqrt().numpy()
    # Nearest centerline of the full tree; `band` is the on-tree metric region d_tree <= 0.02.
    segs = tree_segments("AB"); a, b = (torch.tensor(np.array([getattr(s, k) for s in segs]), dtype=torch.float32) for k in ("start", "end"))
    ab = b - a
    d2 = [(ch[:, None] - a - ((ch[:, None] - a) * ab).sum(-1, keepdim=True).div(ab.square().sum(-1, keepdim=True)).clamp(0, 1) * ab)
          .square().sum(-1).min(1) for ch in txy.split(4096)]
    dist = torch.cat([d.values for d in d2]).sqrt().numpy().reshape(n, n); near = torch.cat([d.indices for d in d2]).numpy()
    paths = {(s.cls, s.path) for s in target}
    is_target = np.array([(s.cls, s.path) in paths for s in segs])[near].reshape(n, n)
    is_a = np.array([s.cls == "A" for s in segs])[near].reshape(n, n)
    if not THIN:
        tree["target"] = np.where(is_target, tree["A"], -99.)
    else:
        band = .02 - dist
        tree = {"AB": np.where(~is_a, band, -1.), "A": np.where(is_a, band, -1.), "target": np.where(is_target, band, -1.)}
    return dict(view=view, radius=radius, tx=tx, ty=ty, txy=txy, tree=tree, n=n)
SCENES = {"full": scene(*FULL), "zoom": scene(*ZOOM)}

def density(sc, points):
    """Density of the endpoints re-noised to SIGMA, normalised on the grid."""
    pts = torch.from_numpy(points).float()
    logp = torch.cat([torch.logsumexp(-torch.cdist(ch, pts).square() / (2 * SIGMA**2), 1) for ch in sc["txy"].split(4096)])
    return torch.exp(logp - logp.max()).numpy().reshape(sc["n"], sc["n"])

def mass_levels(points, masses):
    """Density thresholds whose super-level sets hold the given probability masses (computed on the full view)."""
    p = density(SCENES["full"], points).ravel(); order = np.sort(p)[::-1]; cum = np.cumsum(order) / order.sum()
    return [float(order[np.searchsorted(cum, q)]) for q in masses]

def backdrop(ax, sc, lw_scale=1.):
    for cls, fill, edge in (("AB", GREY, GREY_EDGE), ("A", TREE, TREE_EDGE)):
        ax.contourf(sc["tx"], sc["ty"], sc["tree"][cls], levels=[LEVEL, 9], colors=[fill], zorder=1)
        ax.contour(sc["tx"], sc["ty"], sc["tree"][cls], levels=[LEVEL], colors=[edge], linewidths=.45, linestyles="solid", zorder=2)
    ax.contourf(sc["tx"], sc["ty"], sc["tree"]["target"], levels=[LEVEL, 9], colors=[TARGET], zorder=3)
    ax.contour(sc["tx"], sc["ty"], sc["tree"]["target"], levels=[LEVEL], colors=[TARGET_EDGE], linewidths=.6, linestyles="solid", zorder=3.1)
    v, r = sc["view"], sc["radius"]
    ax.set(xlim=(v[0] - r, v[0] + r), ylim=(v[1] - r, v[1] + r), aspect="equal"); ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values(): side.set_linewidth(.6)

def draw(name, style, zoom):
    """style: 'points' = on/off-tree coloured points, no density; 'density' = sparse filled density + points."""
    fig = plt.figure(figsize=(13.6, 3.5))
    if zoom:
        gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, .07, 1, 1, 1], left=.004, right=.996, top=.9, bottom=.125, wspace=.03)
        axes = [fig.add_subplot(gs[i]) for i in (0, 1, 3, 4, 5)]
    else:
        gs = fig.add_gridspec(1, 5, left=.004, right=.996, top=.9, bottom=.125, wspace=.03)
        axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for k, (ax, (title, pts, on, hit)) in enumerate(zip(axes, PANELS)):
        zoomed = zoom and k >= 2
        sc = SCENES["zoom" if zoomed else "full"]; scale = FULL[1] / sc["radius"]
        if style == "density":
            lv = mass_levels(pts, (.9, .5))
            ax.contourf(sc["tx"], sc["ty"], density(sc, pts), levels=[lv[0], lv[1], 2], colors=["#F4D9B0", "#E8B168"], alpha=.75, zorder=.5)
            ax.scatter(*pts[:256].T, s=4 * scale, c=POINT, alpha=.75, linewidths=0, zorder=6, rasterized=True)
        else:
            k_on, k_off = 3.2 * scale, 4.2 * scale
            ax.scatter(*pts[on].T, s=k_on, c=ON, alpha=.55, linewidths=0, zorder=6, rasterized=True)
            ax.scatter(*pts[~on].T, s=k_off, c=OFF, alpha=.85, linewidths=0, zorder=7, rasterized=True)
        backdrop(ax, sc, scale ** .5)
        ax.text(.03, .97, f"hit {hit.mean():.2f}\non-tree {on.mean():.2f}", transform=ax.transAxes, fontsize=13.5, va="top", linespacing=1.05,
                zorder=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1.5))
        if zoom and k < 2:
            ax.add_patch(Rectangle(ZOOM[0] - ZOOM[1], 2 * ZOOM[1], 2 * ZOOM[1], fill=False, ec="#555555", lw=.9, ls=(0, (3, 2)), zorder=8))
        ax.set_title(title, fontsize=17, pad=6)
    line = lambda color, label, **kw: Line2D([], [], color=color, label=label, **kw)
    keys = [Patch(facecolor=TREE, edgecolor=TREE_EDGE, label="On-tree region ($d_{\\rm tree}\\leq 0.02$)" if THIN else "Data distribution"), Patch(facecolor=TARGET, edgecolor=TARGET_EDGE, label="High-reward branches")]
    if style == "density":
        keys += [Patch(facecolor="#F4D9B0", label="90% of re-noised samples"), Patch(facecolor="#E8B168", label="50%"),
                 line(POINT, "Samples", lw=0, marker="o", markersize=5)]
    else:
        keys += [line(ON, "On-tree samples", lw=0, marker="o", markersize=5), line(OFF, "Off-tree samples", lw=0, marker="o", markersize=5)]
    if zoom:
        keys.append(Patch(facecolor="none", edgecolor="#555555", ls=(0, (3, 2)), label="View of (c)–(e)"))
    fig.legend(handles=keys, loc="lower center", ncol=len(keys), frameon=False, fontsize=16, handlelength=1.6, handletextpad=.5,
               columnspacing=1.5, borderaxespad=.1)
    fig.savefig(OUT / f"{name}.png", dpi=200); fig.savefig(OUT / f"{name}.pdf", dpi=300); plt.close(fig)

OUT.mkdir(parents=True, exist_ok=True)
draw("figure1_samplers_zoom" if THIN else "figure1_samplers_zoom_filled", "points", True)
if "density" in sys.argv:
    draw("figure1_samplers_zoom_density" if THIN else "figure1_samplers_zoom_density_filled", "density", True)
