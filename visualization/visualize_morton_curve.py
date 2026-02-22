#!/usr/bin/env python3
"""
Morton (Z-order) Space-Filling Curve & Barnes-Hut N-Body Visualization

This script produces two animations:
  1. Morton curve traversal on a 3D octree domain, showing how the curve
     visits each cell while preserving spatial locality.
  2. A small N-body simulation comparing the MPI domain decomposition
     (colored by MPI rank) against the evolving particle positions.

Outputs are saved as GIF files in the `visualization/` directory.

Usage:
    python3 visualization/visualize_morton_curve.py

Requirements:
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for GIF generation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib import cm
import os

# ─────────────────────────────────────────────────────────────────────────────
#  Morton Curve Helpers (mirrors C++ implementation)
# ─────────────────────────────────────────────────────────────────────────────

def spread_bits_2d(v: int) -> int:
    """Spread bits for 2D Morton code (used in 2D slice visualization)."""
    v &= 0xFFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v

def morton_encode_2d(x: int, y: int) -> int:
    return spread_bits_2d(x) | (spread_bits_2d(y) << 1)

def spread_bits_3d(v: int) -> int:
    """Spread bits for 3D Morton code (mirrors C++ spreadBits3D)."""
    v &= 0x1FFFFF  # 21 bits
    x = v
    x = (x | (x << 32)) & 0x001F00000000FFFF
    x = (x | (x << 16)) & 0x001F0000FF0000FF
    x = (x | (x << 8))  & 0x100F00F00F00F00F
    x = (x | (x << 4))  & 0x10C30C30C30C30C3
    x = (x | (x << 2))  & 0x1249249249249249
    return x

def morton_encode_3d(x: int, y: int, z: int) -> int:
    return spread_bits_3d(x) | (spread_bits_3d(y) << 1) | (spread_bits_3d(z) << 2)

def compact_bits_3d(v: int) -> int:
    v &= 0x1249249249249249
    v = (v | (v >> 2))  & 0x10C30C30C30C30C3
    v = (v | (v >> 4))  & 0x100F00F00F00F00F
    v = (v | (v >> 8))  & 0x001F0000FF0000FF
    v = (v | (v >> 16)) & 0x001F00000000FFFF
    v = (v | (v >> 32)) & 0x00000000001FFFFF
    return v

def morton_decode_3d(code: int):
    return (compact_bits_3d(code),
            compact_bits_3d(code >> 1),
            compact_bits_3d(code >> 2))


# ─────────────────────────────────────────────────────────────────────────────
#  3D Arrow for matplotlib (for drawing the curve path)
# ─────────────────────────────────────────────────────────────────────────────

class Arrow3D(FancyArrowPatch):
    """Draw a 3D arrow on an Axes3D."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        xs_2d, ys_2d, _ = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs_2d[0], ys_2d[0]), (xs_2d[1], ys_2d[1]))
        return min(zs)


# ─────────────────────────────────────────────────────────────────────────────
#  1) Morton Curve Visualization (3D)
# ─────────────────────────────────────────────────────────────────────────────

def generate_morton_curve_3d(level: int = 2):
    """
    Generate the Morton Z-order curve visiting every cell of a 2^level grid
    in 3D.  Returns arrays of (x, y, z) cell-center coordinates in
    Morton order.
    """
    n = 1 << level          # grid cells per dimension
    total = n ** 3
    codes = []
    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                codes.append((morton_encode_3d(ix, iy, iz), ix, iy, iz))
    codes.sort(key=lambda c: c[0])
    # cell centers (offset by 0.5 so they sit in the middle of the cell)
    xs = np.array([c[1] + 0.5 for c in codes])
    ys = np.array([c[2] + 0.5 for c in codes])
    zs = np.array([c[3] + 0.5 for c in codes])
    return xs, ys, zs, n


def draw_octree_wireframe(ax, n, color="lightgray", lw=0.3):
    """Draw the octree grid wireframe."""
    for i in range(n + 1):
        for j in range(n + 1):
            ax.plot([i, i], [j, j], [0, n], color=color, lw=lw)
            ax.plot([i, i], [0, n], [j, j], color=color, lw=lw)
            ax.plot([0, n], [i, i], [j, j], color=color, lw=lw)


def animate_morton_curve(save_path: str = "visualization/morton_curve_3d.gif"):
    """Create an animated GIF of the 3D Morton curve traversal."""
    print("Generating Morton curve 3D animation …")
    xs, ys, zs, n = generate_morton_curve_3d(level=2)  # 4×4×4 = 64 cells
    total = len(xs)

    fig = plt.figure(figsize=(9, 8), facecolor="#fafafa")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05)

    # Color map: position along the curve → color (plasma is more vivid)
    cmap = cm.plasma
    norm = Normalize(vmin=0, vmax=total - 1)
    colors = [cmap(norm(i)) for i in range(total)]

    # Reveal schedule: 1 cell per frame for first 64, then 20 extra rotation frames
    ROTATION_FRAMES = 25
    n_frames = total + ROTATION_FRAMES

    def _update(frame):
        ax.cla()
        draw_octree_wireframe(ax, n, color="#cccccc", lw=0.25)

        revealed = min(total, frame + 1)

        # Already-visited cells (slightly transparent, smaller)
        if revealed > 1:
            ax.scatter(xs[:revealed - 1], ys[:revealed - 1], zs[:revealed - 1],
                       c=[colors[i] for i in range(revealed - 1)],
                       s=90, edgecolors="#444444", linewidths=0.3,
                       depthshade=True, alpha=0.75, zorder=5)

        # Draw curve path as a single polyline (much faster than individual segments)
        if revealed > 1:
            ax.plot(xs[:revealed], ys[:revealed], zs[:revealed],
                    color="#555555", lw=1.2, alpha=0.5)

        # Highlight current cell with a glow
        if revealed > 0 and revealed <= total:
            idx = revealed - 1
            # Outer glow
            ax.scatter([xs[idx]], [ys[idx]], [zs[idx]],
                       c="gold", s=350, alpha=0.35, zorder=9)
            # Inner marker
            ax.scatter([xs[idx]], [ys[idx]], [zs[idx]],
                       c="red", s=160, marker="o", edgecolors="darkred",
                       linewidths=1.2, zorder=10)

        # Fixed axis limits
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_zlim(0, n)
        ax.set_xlabel("X", fontsize=10, labelpad=6)
        ax.set_ylabel("Y", fontsize=10, labelpad=6)
        ax.set_zlabel("Z", fontsize=10, labelpad=6)
        ax.tick_params(labelsize=7)

        pct = int(100 * revealed / total)
        ax.set_title(f"Morton (Z-order) Curve Traversal\n"
                     f"Cell {revealed}/{total}  ({pct}%)",
                     fontsize=13, fontweight="bold", pad=12)

        # Smooth rotation
        ax.view_init(elev=25, azim=30 + frame * 1.5)
        ax.set_box_aspect([1, 1, 1])

    anim = animation.FuncAnimation(fig, _update, frames=n_frames, interval=100, blit=False)
    anim.save(save_path, writer="pillow", fps=12, dpi=110)
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  2) N-Body Simulation Visualization with MPI Domain Decomposition
# ─────────────────────────────────────────────────────────────────────────────

G = 6.67430e-11  # gravitational constant
MIN_DIST = 0.5

def generate_particles(n: int, seed: int = 42):
    """Generate random particles in a cube centered at origin."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, (n, 3))
    vel = np.zeros((n, 3))
    mass = rng.uniform(1e8, 1e10, n)
    return pos, vel, mass


def compute_forces_brute(pos, mass):
    """Brute-force O(N²) gravitational force computation."""
    n = len(mass)
    force = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)
            if dist < MIN_DIST:
                continue
            f_mag = G * mass[i] * mass[j] / (dist * dist)
            force[i] += f_mag * diff / dist
    return force


def morton_sort_and_partition(pos, num_ranks: int):
    """
    Sort particles by Morton code and partition into `num_ranks` domains.
    Returns:
        sorted_indices: permutation that sorts particles by Morton code
        rank_assignment: array mapping each particle to its MPI rank
    """
    min_bound = pos.min(axis=0)
    max_bound = pos.max(axis=0)
    ranges = max_bound - min_bound
    ranges[ranges <= 0] = 1.0

    # Normalize to [0, 1]
    norm_pos = (pos - min_bound) / ranges

    # Compute Morton codes at resolution=10 (1024 cells per dim, plenty for vis)
    resolution = 10
    scale = (1 << resolution) - 1
    codes = []
    for i in range(len(pos)):
        ix = int(np.clip(norm_pos[i, 0], 0, 1) * scale)
        iy = int(np.clip(norm_pos[i, 1], 0, 1) * scale)
        iz = int(np.clip(norm_pos[i, 2], 0, 1) * scale)
        codes.append(morton_encode_3d(ix, iy, iz))

    codes = np.array(codes, dtype=np.uint64)
    sorted_indices = np.argsort(codes)

    # Equal partitioning along sorted order
    rank_assignment = np.zeros(len(pos), dtype=int)
    per_rank = len(pos) // num_ranks
    remainder = len(pos) % num_ranks
    offset = 0
    for r in range(num_ranks):
        count = per_rank + (1 if r < remainder else 0)
        for k in range(count):
            rank_assignment[sorted_indices[offset + k]] = r
        offset += count

    return sorted_indices, rank_assignment


def _draw_wireframe_box(ax, bmin, bmax, color="#888888", lw=0.5, alpha=0.3):
    """Draw a 3D wireframe box."""
    corners = np.array([
        [bmin[0], bmin[1], bmin[2]],
        [bmax[0], bmin[1], bmin[2]],
        [bmax[0], bmax[1], bmin[2]],
        [bmin[0], bmax[1], bmin[2]],
        [bmin[0], bmin[1], bmax[2]],
        [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmax[2]],
        [bmin[0], bmax[1], bmax[2]],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # pillars
    ]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color=color, lw=lw, alpha=alpha)


def animate_simulation(save_path: str = "visualization/nbody_morton_simulation.gif"):
    """
    Run a small N-body simulation and animate:
      - Particles colored by MPI rank (Morton curve domain decomposition)
      - Smooth fading traces behind each particle
      - Stable fixed axes
      - Longer, finer-stepped simulation for a polished look
    """
    print("Generating N-body simulation animation …")
    N_PARTICLES = 60
    NUM_RANKS = 4
    N_STEPS = 120           # longer simulation
    DT = 0.035              # finer time step → smoother motion
    TRAIL_LEN = 12          # how many past positions to show as trace
    REPARTITION_EVERY = 8   # re-partition ranks along Morton curve

    pos, vel, mass = generate_particles(N_PARTICLES, seed=42)

    # Pre-compute all frames
    print("  Computing simulation …", end="", flush=True)
    frames_pos = [pos.copy()]
    frames_rank = []

    _, rank_assign = morton_sort_and_partition(pos, NUM_RANKS)
    frames_rank.append(rank_assign.copy())

    for step in range(N_STEPS):
        force = compute_forces_brute(pos, mass)
        acc = force / mass[:, None]
        vel += acc * DT
        pos += vel * DT
        frames_pos.append(pos.copy())

        if step % REPARTITION_EVERY == 0:
            _, rank_assign = morton_sort_and_partition(pos, NUM_RANKS)
        frames_rank.append(rank_assign.copy())
    print(" done")

    # ── Global bounds (fixed axes across entire animation) ──
    all_pos = np.concatenate(frames_pos, axis=0)
    global_min = all_pos.min(axis=0)
    global_max = all_pos.max(axis=0)
    pad = 0.12 * (global_max - global_min).max()
    axis_min = global_min - pad
    axis_max = global_max + pad

    # ── Pre-build per-frame trace data ──
    # For each frame, collect the past TRAIL_LEN positions into flat arrays
    # with pre-computed colors + alpha so the render loop is just scatter calls.
    rank_rgb = np.array([
        [0.906, 0.298, 0.235],  # red
        [0.204, 0.596, 0.859],  # blue
        [0.180, 0.800, 0.443],  # green
        [0.953, 0.612, 0.071],  # orange
    ])
    rank_colors_hex = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    rank_labels = [f"Rank {i}" for i in range(NUM_RANKS)]

    print("  Pre-building trace data …", end="", flush=True)
    # For every frame, gather trace scatter arrays (one combined scatter call)
    trace_cache = []
    for frame in range(len(frames_pos)):
        t_start = max(0, frame - TRAIL_LEN)
        if t_start == frame:
            trace_cache.append(None)
            continue
        all_x, all_y, all_z, all_c, all_s = [], [], [], [], []
        for t in range(t_start, frame):
            age = (t - t_start + 1) / (frame - t_start + 1)  # 0→1
            alpha = 0.04 + 0.22 * age
            size = 3 + 16 * age
            p_t = frames_pos[t]
            ra_t = frames_rank[t]
            rgba = np.zeros((N_PARTICLES, 4))
            rgba[:, :3] = rank_rgb[ra_t]
            rgba[:, 3] = alpha
            all_x.append(p_t[:, 0])
            all_y.append(p_t[:, 1])
            all_z.append(p_t[:, 2])
            all_c.append(rgba)
            all_s.append(np.full(N_PARTICLES, size))
        trace_cache.append({
            "x": np.concatenate(all_x),
            "y": np.concatenate(all_y),
            "z": np.concatenate(all_z),
            "c": np.concatenate(all_c, axis=0),
            "s": np.concatenate(all_s),
        })
    print(" done")

    # ── Build animation ──
    fig = plt.figure(figsize=(9, 7), facecolor="#fafafa")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)

    def _update(frame):
        ax.cla()

        p = frames_pos[frame]
        ra = frames_rank[frame]

        # Fixed axis limits
        ax.set_xlim(axis_min[0], axis_max[0])
        ax.set_ylim(axis_min[1], axis_max[1])
        ax.set_zlim(axis_min[2], axis_max[2])

        # Domain bounding box
        _draw_wireframe_box(ax, axis_min, axis_max, color="#aaaaaa", lw=0.5, alpha=0.2)

        # ── Traces: single scatter call with pre-built arrays ──
        tc = trace_cache[frame]
        if tc is not None:
            ax.scatter(tc["x"], tc["y"], tc["z"],
                       c=tc["c"], s=tc["s"],
                       depthshade=False, linewidths=0)

        # ── Current particles (one scatter per rank for the legend) ──
        for r in range(NUM_RANKS):
            mask = ra == r
            if mask.any():
                ax.scatter(p[mask, 0], p[mask, 1], p[mask, 2],
                           c=rank_colors_hex[r], s=48, alpha=0.93,
                           edgecolors="#222222", linewidths=0.35,
                           label=rank_labels[r], depthshade=True)

        # Styling
        ax.set_xlabel("X", fontsize=9, labelpad=5)
        ax.set_ylabel("Y", fontsize=9, labelpad=5)
        ax.set_zlabel("Z", fontsize=9, labelpad=5)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Barnes-Hut N-Body — Morton Domain Decomposition\n"
                     f"Step {frame}/{N_STEPS}  ·  {N_PARTICLES} particles  ·  {NUM_RANKS} MPI ranks",
                     fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8,
                  edgecolor="#cccccc", fancybox=True)
        ax.view_init(elev=22, azim=30 + frame * 0.5)

    total_frames = len(frames_pos)
    print(f"  Rendering {total_frames} frames …")
    anim = animation.FuncAnimation(fig, _update, frames=total_frames, interval=80, blit=False)
    anim.save(save_path, writer="pillow", fps=14, dpi=90)
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  2D Morton curve illustration (simpler, for README thumbnail)
# ─────────────────────────────────────────────────────────────────────────────

def draw_morton_curve_2d(save_path: str = "visualization/morton_curve_2d.png"):
    """Static 2D Morton curve illustration for the README."""
    print("Generating 2D Morton curve illustration …")
    level = 3  # 8×8 grid
    n = 1 << level
    total = n * n

    codes = []
    for iy in range(n):
        for ix in range(n):
            codes.append((morton_encode_2d(ix, iy), ix, iy))
    codes.sort(key=lambda c: c[0])

    xs = [c[1] + 0.5 for c in codes]
    ys = [c[2] + 0.5 for c in codes]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="white")
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=total - 1)

    # Grid
    for i in range(n + 1):
        ax.axhline(i, color="lightgray", lw=0.5)
        ax.axvline(i, color="lightgray", lw=0.5)

    # Curve path
    for i in range(1, total):
        ax.annotate("", xy=(xs[i], ys[i]), xytext=(xs[i - 1], ys[i - 1]),
                     arrowprops=dict(arrowstyle="->", color=cmap(norm(i)), lw=1.5))

    # Cell markers
    ax.scatter(xs, ys, c=[cmap(norm(i)) for i in range(total)],
               s=60, edgecolors="black", linewidths=0.4, zorder=5)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_title("Morton (Z-order) Space-Filling Curve — 2D", fontsize=13, fontweight="bold")

    # Color bar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Curve position", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("visualization", exist_ok=True)

    # 1. Static 2D Morton curve (fast, for README thumbnail)
    draw_morton_curve_2d()

    # 2. Animated 3D Morton curve traversal
    animate_morton_curve()

    # 3. N-body simulation with Morton domain decomposition
    animate_simulation()

    print("\nAll visualizations generated successfully!")
    print("Files:")
    print("  • visualization/morton_curve_2d.png")
    print("  • visualization/morton_curve_3d.gif")
    print("  • visualization/nbody_morton_simulation.gif")
