import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

# ── helper: draw decision boundary on an axis ─────────────────────────────
def plot_boundary(ax, model, X, y, title, show_sv=False,
                  cmap_bg=None, cmap_pts=None, alpha_bg=0.25):
    """
    Draws the filled decision-region background, the decision boundary line,
    margin lines (for linear kernel only), and optionally the support vectors.
    """
    if cmap_bg  is None: cmap_bg  = ListedColormap(['#FFAAAA','#AAAAFF'])
    if cmap_pts is None: cmap_pts = ListedColormap(['#CC0000','#0000CC'])

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02   # mesh resolution

    # Build a dense grid of points that covers the plot area
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Ask the model to classify every grid point
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=alpha_bg, cmap=cmap_bg)   # background shading
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts,
               edgecolors='k', linewidths=0.5, s=40, zorder=3)

    # For linear kernel: draw the decision boundary and the two margin lines
    if hasattr(model, 'decision_function'):
        try:
            Z2 = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contour(xx, yy, Z2, levels=[-1, 0, 1],
                       linestyles=['--', '-', '--'],
                       colors=['gray', 'black', 'gray'],
                       linewidths=[1.2, 2.0, 1.2], zorder=4)
        except Exception:
            pass

    # Highlight support vectors with a thick yellow ring
    if show_sv:
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=160, facecolors='none',
                   edgecolors='gold', linewidths=2.5, zorder=5,
                   label=f'Support vectors ({len(sv)})')
        ax.legend(fontsize=8, loc='upper left')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xticks([]); ax.set_yticks([])


# ============================================================
#  STEP 1 — CREATE LINEARLY SEPARABLE DATA
# ============================================================
# Two clusters of 80 points each, well separated along both axes.
# Class 0: centred at (-1.5, -1.5)
# Class 1: centred at (+1.5, +1.5)
# np.random.randn gives Gaussian noise around each centre.

np.random.seed(42)
n = 80
X_lin = np.vstack([
    np.random.randn(n, 2) + [-1.5, -1.5],   # class 0  — bottom-left
    np.random.randn(n, 2) + [ 1.5,  1.5],   # class 1  — top-right
])
y_lin = np.array([0]*n + [1]*n)

print("=" * 64)
print("STEP 1 — LINEARLY SEPARABLE DATA CREATED")
print("=" * 64)
print(f"  Total points : {len(X_lin)}  ({n} per class)")
print(f"  Class 0 centre: (-1.5, -1.5)  |  Class 1 centre: (+1.5, +1.5)")
print(f"  These two blobs CAN be separated by a straight line.")


# ============================================================
#  STEP 2 — TRAIN LINEAR SVM ON SEPARABLE DATA
# ============================================================
# SVC(kernel='linear') finds the hyperplane (a line in 2D) that:
#   1. Separates the two classes
#   2. Maximises the MARGIN — the gap between the closest points
#      of each class and the decision boundary.
#
# The margin is the "safety zone". A wider margin = more confident
# predictions for new, unseen points near the boundary.
#
# Support Vectors are the training points that sit exactly on the
# margin lines. They are the ONLY points that define the boundary —
# all other points could be removed and the result wouldn't change.
# That is the elegant insight behind SVM.

svm_lin = SVC(kernel='linear', C=1.0)
svm_lin.fit(X_lin, y_lin)

sv = svm_lin.support_vectors_
print("\n" + "=" * 64)
print("STEP 2 — LINEAR SVM TRAINED")
print("=" * 64)
print(f"  Support vectors found : {len(sv)}")
print(f"  (only these {len(sv)} points define the entire decision boundary)")
print(f"  Weight vector w       : {svm_lin.coef_[0].round(4)}")
print(f"  Bias b                : {svm_lin.intercept_[0]:.4f}")
print(f"  Train accuracy        : {svm_lin.score(X_lin, y_lin)*100:.1f}%")


# ============================================================
#  STEP 3 — CREATE NON-LINEAR DATA (Concentric Circles)
# ============================================================
# make_circles() generates two interleaved rings of points.
# Inner ring = class 0, outer ring = class 1.
# NO straight line can ever separate these two classes —
# the data is fundamentally non-linear.
# noise=0.1 adds a small amount of Gaussian noise to each point.

X_circ, y_circ = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)

# SVM is sensitive to feature scale — always normalise before SVC
scaler = StandardScaler()
X_circ_scaled = scaler.fit_transform(X_circ)

print("\n" + "=" * 64)
print("STEP 3 — CONCENTRIC CIRCLES DATA CREATED")
print("=" * 64)
print(f"  Total points : 200  (100 inner, 100 outer)")
print(f"  Data scaled  : yes  (mean=0, std=1 per feature)")
print(f"  Can a straight line separate these? NO — we need a curve.")


# ============================================================
#  STEP 4 — LINEAR SVM ON CIRCLES DATA (Expected Failure)
# ============================================================
# A linear kernel can only draw a straight line.
# For concentric circles, no matter how you tilt that line,
# you will always cut through both classes.
# Accuracy will be around 50% — no better than random guessing.

svm_circ_lin = SVC(kernel='linear', C=1.0)
svm_circ_lin.fit(X_circ_scaled, y_circ)
lin_acc = svm_circ_lin.score(X_circ_scaled, y_circ)

print("\n" + "=" * 64)
print("STEP 4 — LINEAR SVM ON CIRCLES  (Expected to Fail)")
print("=" * 64)
print(f"  Train accuracy : {lin_acc*100:.1f}%  ← ~50% means the model is guessing!")
print(f"  A straight line CANNOT separate two rings. Physics won't allow it.")


# ============================================================
#  STEP 5 — RBF KERNEL ON CIRCLES DATA (Kernel Trick)
# ============================================================
# The RBF (Radial Basis Function) kernel uses the KERNEL TRICK:
# it implicitly maps the data into an infinite-dimensional space
# where the two rings BECOME linearly separable — without ever
# computing that transformation explicitly (too expensive).
#
# Mathematically: K(x, z) = exp(-γ ||x - z||²)
# Intuitively: it measures "similarity by distance".
# Two points close together get similarity ≈ 1.
# Two points far apart get similarity ≈ 0.
# This lets the model draw curved, circular boundaries.

svm_circ_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_circ_rbf.fit(X_circ_scaled, y_circ)
rbf_acc = svm_circ_rbf.score(X_circ_scaled, y_circ)

print("\n" + "=" * 64)
print("STEP 5 — RBF KERNEL ON CIRCLES  (Kernel Trick Works)")
print("=" * 64)
print(f"  Train accuracy : {rbf_acc*100:.1f}%  ← near-perfect with curved boundary!")
print(f"  The kernel trick 'imagines' the rings in higher dimensions")
print(f"  where a flat hyperplane separates them cleanly.")


# ============================================================
#  STEP 6 — EXPERIMENT: C PARAMETER
# ============================================================
# C controls the trade-off between margin width and training errors.
#
#   Low  C (e.g. 0.1):
#     → Wide margin, allows some misclassifications.
#     → "I'm okay with a few mistakes if the boundary stays smooth."
#     → Better generalisation, less overfitting.
#
#   High C (e.g. 100):
#     → Narrow margin, tries to classify every training point correctly.
#     → "I want zero training errors no matter what."
#     → Risk of overfitting — wiggly boundary that memorises noise.
#
# C = 1 is the default — a good balance for most problems.

C_values = [0.1, 1, 100]
svm_C = [SVC(kernel='rbf', C=c, gamma='scale').fit(X_circ_scaled, y_circ)
         for c in C_values]

print("\n" + "=" * 64)
print("STEP 6 — EFFECT OF C PARAMETER  (RBF kernel on circles)")
print("=" * 64)
print(f"{'C value':>10}  {'Train Acc':>12}  {'Support Vectors':>16}  {'Interpretation'}")
print("-" * 72)
for c, m in zip(C_values, svm_C):
    acc = m.score(X_circ_scaled, y_circ) * 100
    nsv = len(m.support_vectors_)
    interp = ("Wide margin, may miss some" if c < 1
              else "Balanced" if c == 1
              else "Narrow margin, strict fit")
    print(f"{c:>10}  {acc:>11.1f}%  {nsv:>16}  {interp}")


# ============================================================
#  STEP 7 — EXPERIMENT: GAMMA PARAMETER
# ============================================================
# Gamma controls how FAR each training point's influence reaches.
#
#   Low  gamma (e.g. 0.1):
#     → Each point influences a WIDE neighbourhood.
#     → Boundary is smooth, simple, may underfit.
#     → "Every point votes for a large area."
#
#   High gamma (e.g. 10):
#     → Each point only influences a TINY neighbourhood.
#     → Boundary is wiggly, complex, may overfit.
#     → "Each point only cares about its immediate neighbours."
#
# gamma='scale' (default) = 1 / (n_features * X.var())
# — a sensible automatic choice.

gamma_values = [0.1, 'scale', 10]
gamma_labels = ['gamma=0.1', 'gamma=scale', 'gamma=10']
svm_G = [SVC(kernel='rbf', C=1.0, gamma=g).fit(X_circ_scaled, y_circ)
         for g in gamma_values]

print("\n" + "=" * 64)
print("STEP 7 — EFFECT OF GAMMA PARAMETER  (RBF kernel, C=1)")
print("=" * 64)
print(f"{'Gamma':>12}  {'Train Acc':>12}  {'Support Vectors':>16}  {'Boundary shape'}")
print("-" * 72)
for g, lbl, m in zip(gamma_values, gamma_labels, svm_G):
    acc = m.score(X_circ_scaled, y_circ) * 100
    nsv = len(m.support_vectors_)
    interp = ("Smooth, wide influence" if g == 0.1
              else "Balanced (auto)" if g == 'scale'
              else "Wiggly, may overfit")
    print(f"{str(g):>12}  {acc:>11.1f}%  {nsv:>16}  {interp}")


# ============================================================
#  STEP 8 — BIG PLOT
# ============================================================
# Layout: 4 rows × 3 columns = 12 panels
#
#  Row 0: [Linear separable raw] [Linear SVM + margin + SVs] [empty / spacer]
#  Row 1: [Circles raw]          [Linear SVM fails]          [RBF SVM works]
#  Row 2: [C=0.1]                [C=1]                        [C=100]
#  Row 3: [gamma=0.1]            [gamma=scale]                [gamma=10]

fig = plt.figure(figsize=(18, 22))
fig.suptitle("SVM Decision Boundaries — Visualised\n"
             "Linear kernel · RBF kernel · C parameter · γ parameter",
             fontsize=14, y=0.995)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.25)


# ── Row 0, Col 0: Raw linearly separable data ─────────────────
ax00 = fig.add_subplot(gs[0, 0])
cmap_pts = ListedColormap(['#CC0000','#0000CC'])
ax00.scatter(X_lin[:, 0], X_lin[:, 1], c=y_lin, cmap=cmap_pts,
             edgecolors='k', linewidths=0.5, s=40)
ax00.set_title("Raw Data — Linearly Separable\nTwo blobs, easy to split", fontsize=9)
ax00.set_xticks([]); ax00.set_yticks([])


# ── Row 0, Col 1: Linear SVM + margin + support vectors ───────
ax01 = fig.add_subplot(gs[0, 1])
plot_boundary(ax01, svm_lin, X_lin, y_lin,
              "Linear SVM — Decision Boundary + Margin\n"
              "Solid line = boundary · Dashed = margins · Gold = support vectors",
              show_sv=True)


# ── Row 0, Col 2: Support vectors close-up explanation ────────
ax02 = fig.add_subplot(gs[0, 2])
plot_boundary(ax02, svm_lin, X_lin, y_lin,
              f"Support Vectors = only {len(sv)} points define the boundary\n"
              "Remove all other points → boundary unchanged",
              show_sv=True)
# Dim non-support-vector points
sv_idx = svm_lin.support_
mask = np.ones(len(X_lin), dtype=bool); mask[sv_idx] = False
ax02.scatter(X_lin[mask, 0], X_lin[mask, 1],
             c=y_lin[mask], cmap=cmap_pts,
             alpha=0.12, edgecolors='none', s=40, zorder=2)


# ── Row 1, Col 0: Concentric circles raw ──────────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax10.scatter(X_circ_scaled[:, 0], X_circ_scaled[:, 1], c=y_circ,
             cmap=cmap_pts, edgecolors='k', linewidths=0.5, s=40)
ax10.set_title("Concentric Circles — Non-linear Data\nNo straight line can separate these", fontsize=9)
ax10.set_xticks([]); ax10.set_yticks([])


# ── Row 1, Col 1: Linear SVM fails on circles ─────────────────
ax11 = fig.add_subplot(gs[1, 1])
plot_boundary(ax11, svm_circ_lin, X_circ_scaled, y_circ,
              f"Linear SVM on Circles — FAILS\n"
              f"Acc={lin_acc*100:.1f}%  ← barely better than random guessing")


# ── Row 1, Col 2: RBF SVM succeeds on circles ─────────────────
ax12 = fig.add_subplot(gs[1, 2])
plot_boundary(ax12, svm_circ_rbf, X_circ_scaled, y_circ,
              f"RBF SVM on Circles — WORKS\n"
              f"Acc={rbf_acc*100:.1f}%  ← kernel trick maps to higher dimensions",
              show_sv=True)


# ── Row 2: C parameter effect ─────────────────────────────────
C_axes = [fig.add_subplot(gs[2, j]) for j in range(3)]
C_desc = ['Wide margin, tolerates errors\n(may underfit)',
          'Balanced  (default)',
          'Strict, narrow margin\n(may overfit)']
for ax, c, m, desc in zip(C_axes, C_values, svm_C, C_desc):
    acc = m.score(X_circ_scaled, y_circ) * 100
    plot_boundary(ax, m, X_circ_scaled, y_circ,
                  f"C = {c}  |  Acc = {acc:.1f}%\n{desc}")


# ── Row 3: Gamma parameter effect ─────────────────────────────
G_axes = [fig.add_subplot(gs[3, j]) for j in range(3)]
G_desc = ['Smooth, wide influence\n(may underfit)',
          'Auto-scale (default)',
          'Wiggly, tight fit\n(may overfit)']
for ax, g, lbl, m, desc in zip(G_axes, gamma_values, gamma_labels, svm_G, G_desc):
    acc = m.score(X_circ_scaled, y_circ) * 100
    plot_boundary(ax, m, X_circ_scaled, y_circ,
                  f"{lbl}  |  Acc = {acc:.1f}%\n{desc}")


# ── Row labels on the left ─────────────────────────────────────
row_labels = ["Linear SVM\n(separable data)",
              "Kernel Comparison\n(circles data)",
              "C parameter\n(RBF, circles)",
              "γ parameter\n(RBF, C=1)"]
for row, label in enumerate(row_labels):
    ax_ref = fig.axes[row * 3]
    ax_ref.set_ylabel(label, fontsize=9, labelpad=8,
                      rotation=90, va='center', ha='center')

plt.savefig("svm_boundaries_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → svm_boundaries_plot.png")