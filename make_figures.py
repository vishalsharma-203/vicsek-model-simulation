"""
Generates the figures used in README.md.

    python make_figures.py

Outputs (in images/):
    flock.gif              animation of the flock forming (eta = 0.5)
    regimes.png            snapshots at low, intermediate and high noise
    order_vs_time.png      order parameter vs time for four noise values
    phase_transition.png   average order parameter vs noise
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import vicsek as v

os.makedirs("images", exist_ok=True)

BG = "#0d1117"       # dark background, matches GitHub dark theme
FG = "#e6edf3"
ACCENT = "#58a6ff"
GRID = "#30363d"


def style_dark(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color=GRID, alpha=0.5, linewidth=0.5)


def draw_flock(ax, pos, theta):
    ax.quiver(pos[:, 0], pos[:, 1], np.cos(theta), np.sin(theta),
              theta % (2 * np.pi), cmap="hsv", clim=(0, 2 * np.pi),
              pivot="mid", scale=28, width=0.005)
    ax.set_xlim(0, v.L)
    ax.set_ylim(0, v.L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)


def run(eta, steps, seed=1):
    rng = np.random.default_rng(seed)
    pos, theta = v.init_system(v.N, v.L, rng)
    for _ in range(steps):
        pos, theta = v.step(pos, theta, eta, rng)
    return pos, theta


# ---------------------------------------------------------------
# 1. Animated GIF
# ---------------------------------------------------------------
def make_gif(eta=0.5, frames=160, sub=2):
    rng = np.random.default_rng(1)
    pos, theta = v.init_system(v.N, v.L, rng)
    state = {"pos": pos, "theta": theta}

    fig, ax = plt.subplots(figsize=(5, 5), dpi=70)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

    def update(frame):
        for _ in range(sub):
            state["pos"], state["theta"] = v.step(state["pos"], state["theta"], eta, rng)
        ax.clear()
        draw_flock(ax, state["pos"], state["theta"])
        phi = v.order_parameter(state["theta"])
        ax.set_title(f"noise = {eta}    order = {phi:.2f}", color=FG, fontsize=11)

    ani = FuncAnimation(fig, update, frames=frames)
    ani.save("images/flock.gif", writer=PillowWriter(fps=20))
    plt.close(fig)
    print("flock.gif done")


# ---------------------------------------------------------------
# 2. Snapshots at three noise levels
# ---------------------------------------------------------------
def make_regimes(etas=(0.5, 3.0, 6.0), steps=500):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), dpi=130)
    fig.patch.set_facecolor(BG)
    for ax, eta in zip(axes, etas):
        pos, theta = run(eta, steps)
        draw_flock(ax, pos, theta)
        ax.set_title(f"noise = {eta}\norder = {v.order_parameter(theta):.2f}",
                     color=FG, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.80, wspace=0.04)
    fig.savefig("images/regimes.png", facecolor=BG)
    plt.close(fig)
    print("regimes.png done")


# ---------------------------------------------------------------
# 3. Order parameter vs time
# ---------------------------------------------------------------
def make_phi_time(etas=(0.5, 2.0, 4.0, 6.0), steps=500):
    colors = ["#3fb950", "#58a6ff", "#d29922", "#f85149"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    fig.patch.set_facecolor(BG)
    style_dark(ax)
    for eta, c in zip(etas, colors):
        rng = np.random.default_rng(1)
        pos, theta = v.init_system(v.N, v.L, rng)
        phis = []
        for _ in range(steps):
            pos, theta = v.step(pos, theta, eta, rng)
            phis.append(v.order_parameter(theta))
        ax.plot(phis, color=c, label=f"noise = {eta}", linewidth=1.6)
    ax.set_xlabel("time step")
    ax.set_ylabel("order parameter")
    ax.set_ylim(0, 1.05)
    leg = ax.legend(facecolor=BG, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.tight_layout()
    fig.savefig("images/order_vs_time.png", facecolor=BG)
    plt.close(fig)
    print("order_vs_time.png done")


# ---------------------------------------------------------------
# 4. Phase transition
# ---------------------------------------------------------------
def make_sweep(n_eta=30, settle=300, measure=300):
    etas = np.linspace(0.1, 2 * np.pi, n_eta)
    means, stds = [], []
    for eta in etas:
        rng = np.random.default_rng(1)
        pos, theta = v.init_system(v.N, v.L, rng)
        for _ in range(settle):
            pos, theta = v.step(pos, theta, eta, rng)
        vals = []
        for _ in range(measure):
            pos, theta = v.step(pos, theta, eta, rng)
            vals.append(v.order_parameter(theta))
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        print(f"eta = {eta:.2f}  <phi> = {means[-1]:.3f}  std = {stds[-1]:.3f}", flush=True)
    np.savetxt("images/sweep_data.csv", np.column_stack((etas, means, stds)),
               delimiter=",", header="eta,mean_phi,std_phi", comments="")

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    fig.patch.set_facecolor(BG)
    style_dark(ax)
    ax.errorbar(etas, means, yerr=stds, fmt="o-", color=ACCENT, ecolor="#8b949e",
                capsize=2, markersize=4, linewidth=1.6)
    ax.axhline(1 / np.sqrt(v.N), color="#f85149", linestyle="--", linewidth=1)
    ax.text(0.15, 1 / np.sqrt(v.N) + 0.025, r"random level $1/\sqrt{N}$", color="#f85149", fontsize=9)
    ax.set_xlabel("noise strength")
    ax.set_ylabel("average order parameter")
    ax.set_title(f"Vicsek model: N = {v.N}, density = {v.N / v.L**2:.0f}")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("images/phase_transition.png", facecolor=BG)
    plt.close(fig)
    print("phase_transition.png done")


if __name__ == "__main__":
    make_regimes()
    make_phi_time()
    make_gif()
    make_sweep()
