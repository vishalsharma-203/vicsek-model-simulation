"""
Vicsek model of collective motion (Vicsek et al., PRL 75, 1226, 1995)

Run from the terminal:
    python vicsek.py animate   # watch the flock move
    python vicsek.py phi       # order parameter vs time for a few noise values
    python vicsek.py sweep     # order parameter vs noise (the phase transition)

Needs: numpy, scipy, matplotlib   (pip install numpy scipy matplotlib)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree

# ---------------------------------------------------------------
# Parameters (same as the original paper)
# ---------------------------------------------------------------
N = 300      # number of particles
L = 5.0      # side of the square box (density = N / L^2 = 12)
V0 = 0.03    # speed of every particle (constant)
R = 1.0      # interaction radius
# Noise strength eta is passed in separately, because we vary it.
# Each angle gets noise drawn uniformly from [-eta/2, +eta/2].
# eta = 2*pi means the noise is completely random.


# ---------------------------------------------------------------
# Step 1: initial condition
# ---------------------------------------------------------------
def init_system(n, box, rng):
    """Random positions in the box and random directions."""
    pos = rng.random((n, 2)) * box          # shape (n, 2): x and y of each particle
    theta = rng.random(n) * 2 * np.pi       # shape (n,): direction angle of each particle
    return pos, theta


# ---------------------------------------------------------------
# Steps 2-4: one time step of the model
# ---------------------------------------------------------------
def step(pos, theta, eta, rng):
    n = len(theta)

    # Step 2: find neighbours within radius R, with periodic boundaries.
    # boxsize=L makes the tree treat the box as wrapping around.
    tree = cKDTree(pos, boxsize=L)
    neighbors = tree.query_ball_point(pos, R)   # list of index lists (includes the particle itself)

    # Step 3: average direction of the neighbours.
    # We average the vectors (cos, sin), then convert back to an angle.
    mean_theta = np.empty(n)
    for i, nb in enumerate(neighbors):
        mean_theta[i] = np.arctan2(np.sin(theta[nb]).sum(), np.cos(theta[nb]).sum())

    # Step 4: add noise, then move every particle by V0 in its new direction.
    noise = eta * (rng.random(n) - 0.5)         # uniform in [-eta/2, +eta/2]
    new_theta = mean_theta + noise
    new_pos = pos + V0 * np.column_stack((np.cos(new_theta), np.sin(new_theta)))
    new_pos %= L                                 # wrap around the box
    new_pos[new_pos >= L] = 0.0                  # guard against floating-point edge case
    return new_pos, new_theta


# ---------------------------------------------------------------
# Step 6: order parameter
# ---------------------------------------------------------------
def order_parameter(theta):
    """Length of the average unit-velocity vector. 1 = all aligned, ~0 = random."""
    return np.hypot(np.cos(theta).sum(), np.sin(theta).sum()) / len(theta)


# ---------------------------------------------------------------
# Step 5: animation
# ---------------------------------------------------------------
def animate(eta=0.5, steps=1000, save_gif=None):
    rng = np.random.default_rng(1)
    pos, theta = init_system(N, L, rng)
    state = {"pos": pos, "theta": theta}

    fig, ax = plt.subplots(figsize=(6, 6))
    quiv = ax.quiver(pos[:, 0], pos[:, 1], np.cos(theta), np.sin(theta),
                     theta % (2 * np.pi), cmap="hsv", clim=(0, 2 * np.pi),
                     pivot="mid", scale=25, width=0.004)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    title = ax.set_title("")

    def update(frame):
        state["pos"], state["theta"] = step(state["pos"], state["theta"], eta, rng)
        p, t = state["pos"], state["theta"]
        quiv.set_offsets(p)
        quiv.set_UVC(np.cos(t), np.sin(t), t % (2 * np.pi))
        title.set_text(f"eta = {eta}   step = {frame}   order = {order_parameter(t):.2f}")
        return quiv, title

    ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=False)
    if save_gif:
        ani.save(save_gif, writer="pillow", fps=30)
    else:
        plt.show()


# ---------------------------------------------------------------
# Step 6: order parameter vs time
# ---------------------------------------------------------------
def phi_vs_time(etas=(0.5, 2.0, 4.0, 6.0), steps=500):
    plt.figure(figsize=(7, 4))
    for eta in etas:
        rng = np.random.default_rng(1)
        pos, theta = init_system(N, L, rng)
        phis = []
        for _ in range(steps):
            pos, theta = step(pos, theta, eta, rng)
            phis.append(order_parameter(theta))
        plt.plot(phis, label=f"eta = {eta}")
    plt.xlabel("time step")
    plt.ylabel("order parameter  phi")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# Step 7: phase transition
# ---------------------------------------------------------------
def sweep(n_eta=20, settle=300, measure=300):
    etas = np.linspace(0.1, 2 * np.pi, n_eta)
    mean_phi = []
    for eta in etas:
        rng = np.random.default_rng(1)
        pos, theta = init_system(N, L, rng)
        for _ in range(settle):                  # let the system reach steady state
            pos, theta = step(pos, theta, eta, rng)
        values = []
        for _ in range(measure):                 # then measure
            pos, theta = step(pos, theta, eta, rng)
            values.append(order_parameter(theta))
        mean_phi.append(np.mean(values))
        print(f"eta = {eta:.2f}   <phi> = {mean_phi[-1]:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(etas, mean_phi, "o-")
    plt.xlabel("noise strength  eta")
    plt.ylabel("average order parameter  <phi>")
    plt.title(f"Vicsek model, N = {N}, density = {N / L**2:.0f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "animate"
    if mode == "animate":
        animate()
    elif mode == "phi":
        phi_vs_time()
    elif mode == "sweep":
        sweep()
    else:
        print("Use: python vicsek.py animate | phi | sweep")
