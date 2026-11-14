<div align="center">

# Vicsek Model of Collective Motion

**How does a crowd of blind followers become a flock?**
A from-scratch Python simulation of self-propelled particles that align with their neighbours, and the order–disorder transition that appears as noise increases.

<img src="images/flock.gif" alt="Animation of 300 particles self-organising into a flock" width="420">

`Python` · `NumPy` · `SciPy` · `Matplotlib`

</div>

---

## The idea

Birds, fish, bacteria and migrating cells often move as one, even though no individual is in charge. In 1995, Vicsek and collaborators showed that a very simple rule is enough to reproduce this:

> **Each particle moves at constant speed, and at every step turns to the average direction of the particles within a fixed radius, plus a little random noise.**

When the noise is small, the whole group ends up moving together. When it is large, the motion is random. Between the two, the system changes character, much like a phase transition in a magnet or a liquid. This repository reproduces that behaviour.

## Results

### Three regimes

Arrows show the direction of motion, and colour encodes the angle. At low noise nearly everyone shares one colour (one flock), at intermediate noise there are local groups, and at high noise the directions are random.

<img src="images/regimes.png" alt="Snapshots at noise 0.5, 3.0 and 6.0">

### Measuring order

The **order parameter** is the length of the average direction vector,

$$\varphi = \frac{1}{N}\left|\sum_{i=1}^{N} \hat{\mathbf v}_i\right|$$

It equals 1 when all particles move the same way and about $1/\sqrt{N}\approx 0.06$ for random directions. Starting from random directions, low-noise systems order quickly and high-noise ones never do:

<img src="images/order_vs_time.png" alt="Order parameter versus time for four noise values" width="620">

### Order versus noise

Averaging $\varphi$ in the steady state for 30 noise values (error bars are the fluctuations over time):

<img src="images/phase_transition.png" alt="Average order parameter versus noise strength" width="620">

Order falls from about 1.0 at near-zero noise to the random level at the maximum noise of $2\pi$. For this system size the decrease is smooth, with the largest fluctuations around noise ≈ 4.5–5.5, where the system is closest to losing order. The raw numbers are in [`images/sweep_data.csv`](images/sweep_data.csv).

> **Note:** this is a single run per noise value with N = 300, so the curve is rounded by finite-size effects. Checking how it changes with system size is one of the next steps below.

## How it works

Each time step has four parts, implemented in `step()` in [`vicsek.py`](vicsek.py):

1. **Find neighbours.** A KD-tree (`scipy.spatial.cKDTree`) finds all particles within radius `R`. Setting `boxsize=L` makes the box periodic, so particles wrap around the edges.
2. **Average their directions.** Angles can't be averaged directly (10° and 350° would average to 180°), so each angle is converted to a unit vector, the vectors are summed, and the angle of the sum is taken with `arctan2`.
3. **Add noise.** A random angle uniform in $[-\eta/2, +\eta/2]$ is added. At $\eta = 2\pi$ the noise is fully random.
4. **Move.** Every particle steps a distance `V0` along its new direction, with positions wrapped back into the box.

### Parameters

| Symbol | Meaning | Value |
|---|---|---|
| `N` | number of particles | 300 |
| `L` | side of the periodic box | 5.0 (density 12) |
| `V0` | speed of every particle | 0.03 |
| `R` | interaction radius | 1.0 |
| `eta` | noise strength | varied from 0 to 2π |

## Run it yourself

```bash
git clone https://github.com/vishalsharma-203/vicsek-model.git
cd vicsek-model
pip install -r requirements.txt

python vicsek.py animate     # live animation
python vicsek.py phi         # order parameter vs time
python vicsek.py sweep       # order vs noise (about a minute)
python make_figures.py       # regenerate every figure in images/
```

## Repository contents

| File | Purpose |
|---|---|
| `vicsek.py` | the model, with an animation and the two analyses |
| `make_figures.py` | regenerates the figures in `images/` |
| `images/` | GIF, plots and the raw sweep data |

## Next steps

- [ ] Repeat the noise sweep for N = 100, 300, 1000 at fixed density and study finite-size effects
- [ ] Average over several random seeds and add proper error bars
- [ ] Vary the density and see how the transition moves
- [ ] Speed up the neighbour averaging (vectorised or sparse-matrix version)
- [ ] Try variants, such as vectorial noise or an obstacle in the box

## Reference

T. Vicsek, A. Czirók, E. Ben-Jacob, I. Cohen, O. Shochet, *Novel type of phase transition in a system of self-driven particles*, Physical Review Letters **75**, 1226 (1995).

## Author

**Vishal Sharma**, BS-MS Physics student at NISER Bhubaneswar, interested in biological physics and the collective behaviour of many-body systems.

Released under the [MIT License](LICENSE).
