import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------
# Figure 4.5
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=300)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Draw clean axes from origin (0, 0)
arrow_kw = dict(arrowstyle='->', color='black', lw=1.3, mutation_scale=14)
ax.annotate('', xy=(1.05, 0.0), xytext=(0.0, 0.0), arrowprops=arrow_kw)
ax.annotate('', xy=(0.0, 1.05), xytext=(0.0, 0.0), arrowprops=arrow_kw)

ax.text(1.08, 0.0, r'$x$', fontsize=12, ha='left', va='center')
ax.text(0.0, 1.08, r'$t$', fontsize=12, ha='center', va='bottom')

# Red curve f*(x)
# Shape: starts at (0, 0.15), inflection around 0.45, at x0=0.65 it is around 0.60
x = np.linspace(0.0, 0.96, 300)
# Use smooth cubic polynomial:
# f(0) = 0.16, f'(0) = 1.3
# inflection around x=0.48 with slope 0.3
# f(0.96) = 0.90
x_inf = 0.48
f_star = 0.54 + 2.4 * ((x - x_inf)**3) + 0.28 * (x - x_inf)
ax.plot(x, f_star, color='red', lw=1.8, zorder=3)
ax.text(0.92, f_star[-1] + 0.05, r'$f^\star(x)$', color='black', fontsize=11, ha='center')

# Slicing line at x0 = 0.65
x0 = 0.65
mu_t = 0.54 + 2.4 * ((x0 - x_inf)**3) + 0.28 * (x0 - x_inf) # approx 0.60

ax.plot([x0, x0], [0, 0.96], color='#5A7288', lw=1.2, zorder=2)

# Rotated Gaussian bell curve
t_vals = np.linspace(0.15, 0.95, 400)
sigma = 0.070
density = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((t_vals - mu_t) / sigma)**2)
scale_x = 0.018
x_density = x0 + scale_x * density

ax.plot(x_density, t_vals, color='#0044EE', lw=1.6, zorder=4)

# Text label p(t | x0, w, sigma^2)
ax.text(x0 + 0.015, mu_t - 0.16, r'$p(t \mid x_0, \mathbf{w}, \sigma^2)$', fontsize=10.5, color='black')

ax.set_xlim(-0.02, 1.15)
ax.set_ylim(-0.02, 1.15)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
fig.savefig('/home/student/.gemini/antigravity/brain/f1a31a89-892f-49e9-a93b-0248283a4d0b/scratch/test_fig4_5_v2.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------------
# Figure 4.6
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8), dpi=300)
q_values = [0.3, 1, 2, 10]
diff = np.linspace(-2, 2, 500)

for ax, q in zip(axes.flat, q_values):
    loss = np.abs(diff) ** q
    ax.plot(diff, loss, color='red', lw=1.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 2.05)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel(r'$f - t$', fontsize=11)
    if q == 1:
        ax.set_ylabel(r'$|f - t|^1$', fontsize=11)
    elif q == 0.3:
        ax.set_ylabel(r'$|f - t|^{0.3}$', fontsize=11)
    else:
        ax.set_ylabel(rf'$|f - t|^{{{q}}}$', fontsize=11)
    ax.text(0.0, 1.7, rf'$q = {q}$', fontsize=11, ha='center', va='center')
    ax.tick_params(direction='out', top=False, right=False)
    for s in ax.spines.values():
        s.set_linewidth(0.8)

plt.tight_layout()
fig.savefig('/home/student/.gemini/antigravity/brain/f1a31a89-892f-49e9-a93b-0248283a4d0b/scratch/test_fig4_6.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved test_fig4_5_v2.png and test_fig4_6.png')
