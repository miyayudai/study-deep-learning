import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=300)

# Axes styling with arrows
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Draw black coordinate axes with arrows
arrow_kw = dict(arrowstyle='->', color='black', lw=1.5, mutation_scale=15)
ax.annotate('', xy=(1.02, 0.0), xytext=(-0.02, 0.0), arrowprops=arrow_kw)
ax.annotate('', xy=(0.0, 1.02), xytext=(0.0, -0.02), arrowprops=arrow_kw)

ax.text(1.05, 0.0, r'$x$', fontsize=13, ha='left', va='center')
ax.text(0.0, 1.05, r'$t$', fontsize=13, ha='center', va='bottom')

# Regression curve f*(x)
# Cubic S-curve: passes through (0, 0.1), has inflection around x=0.45, rises to (0.92, 0.90)
x = np.linspace(0.0, 0.92, 300)
# A smooth curve f*(x):
# Let f*(x) = 0.55 + 1.2 * (x - 0.45)**3 + 0.15 * (x - 0.45)
# at x=0: 0.55 + 1.2*(-0.45)**3 + 0.15*(-0.45) = 0.55 - 0.109 - 0.0675 = 0.37...
# Let's adjust parameters so f*(0) ~ 0.15, f*(0.45) ~ 0.55, f*(0.92) ~ 0.90
f_star = 0.12 + 0.9 * x + 0.3 * np.sin(2.5 * x) # smooth curve
# Let's shape it to match textbook:
# Textbook red curve: starts at (0, 0.15), curves up concave downwards until x ~ 0.4,
# flattens near x ~ 0.45, then curves concave upwards towards top right.
# That is exactly f*(x) = 0.55 + 2.2 * (x - 0.45)**3 + 0.45*(x - 0.45) + 0.08
x_c = x - 0.45
f_star = 0.55 + 2.5 * (x_c**3) + 0.25 * x_c
ax.plot(x, f_star, color='red', lw=2.0, zorder=3)
ax.text(0.88, f_star[-1] + 0.05, r'$f^\star(x)$', color='black', fontsize=12, ha='center')

# Slicing line at x0
x0 = 0.50
# Find f*(x0)
x0_c = x0 - 0.45
mu_t = 0.55 + 2.5 * (x0_c**3) + 0.25 * x0_c

# Vertical grey-blue line
ax.plot([x0, x0], [0, 0.96], color='#556688', lw=1.5, zorder=2)

# Blue Gaussian density along t-axis:
# t runs from 0.1 to 0.95
t_vals = np.linspace(0.1, 0.95, 300)
sigma = 0.075
density = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((t_vals - mu_t) / sigma)**2)
# Scale density horizontally
scale_x = 0.016
x_density = x0 + scale_x * density

ax.plot(x_density, t_vals, color='#0044FF', lw=1.8, zorder=4)

# Label p(t | x0, w, sigma^2)
ax.text(x0 + 0.02, mu_t - 0.18, r'$p(t \mid x_0, \mathbf{w}, \sigma^2)$', fontsize=11, color='black')

ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.05, 1.1)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
out_path = '/home/student/.gemini/antigravity/brain/f1a31a89-892f-49e9-a93b-0248283a4d0b/scratch/test_fig4_5.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print('Saved', out_path)
