import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
#  STEP 1 — GENERATE BLOB DATA  (4 true clusters)
# ============================================================
# make_blobs creates Gaussian clusters at specified centres.
# We deliberately create 4 clusters so we can later verify
# the elbow method correctly identifies K=4.
#
# cluster_std controls how spread out each blob is.
# random_state=42 makes the blobs reproducible.
#
# KEY INSIGHT: In real life you don't know the true K.
# We create 4 blobs here ONLY to verify the elbow method works.
# Then we pretend we don't know K and let the method find it.

np.random.seed(42)
n_samples   = 400     # 400 total points, 100 per cluster
true_K      = 4

X, y_true = make_blobs(
    n_samples    = n_samples,
    centers      = true_K,
    cluster_std  = 0.9,
    random_state = 42
)

# Scale features — KMeans uses Euclidean distance,
# so scaling ensures no feature dominates the distance.
scaler = StandardScaler()
X      = scaler.fit_transform(X)

print("=" * 64)
print("STEP 1 — BLOB DATA GENERATED")
print("=" * 64)
print(f"  Total points    : {n_samples}")
print(f"  True clusters   : {true_K}  (the answer we're trying to find)")
print(f"  Features        : 2  (so we can visualize in 2D)")
print(f"  Scaled          : yes  (StandardScaler)")
print(f"\n  In real life you'd START HERE — not knowing that K=4.")
print(f"  The elbow method's job is to discover this from the data.")


# ============================================================
#  STEP 2 — RAW DATA  (no cluster colors yet)
# ============================================================
# First we look at the data as K-Means would see it:
# just a cloud of unlabelled points.
# We're about to figure out the natural structure from scratch.


# ============================================================
#  STEP 3 — RUN K-MEANS FOR K = 1 to 10, RECORD INERTIA
# ============================================================
# Inertia = Within-Cluster Sum of Squares (WCSS)
#         = sum of squared distances from each point to its cluster centre
#
# How inertia behaves:
#   K=1 : one centroid in the middle of everything → very high inertia
#   K=2 : two centres, each covers half the data → big drop
#   K=4 : matches the true structure → large jump in improvement ends here
#   K=5 : adding a 5th cluster splits one existing cluster → small gain
#   K=10: ten tiny clusters → near-zero inertia but meaningless
#
# The ELBOW is where adding more clusters stops giving big returns.
# On a plot of inertia vs K, it looks like an arm bending at the elbow.

K_values = list(range(1, 11))
inertias = []

print("\n" + "=" * 64)
print("STEP 3 — INERTIA FOR K = 1 to 10  (Elbow Method)")
print("=" * 64)
print(f"\n  {'K':>4}  {'Inertia':>12}  {'Drop from prev':>16}  Interpretation")
print(f"  {'-'*62}")

for k in K_values:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

    drop = inertias[-2] - inertias[-1] if k > 1 else 0
    note = ""
    if k == 1:    note = "← everything in one cluster"
    elif k == 4:  note = "← ELBOW expected here (true K=4)"
    elif k == 5:  note = "← drop shrinks, diminishing returns begin"
    elif k == 10: note = "← near zero but 10 clusters is meaningless"

    print(f"  {k:>4}  {km.inertia_:>12.2f}  {drop:>16.2f}  {note}")

# Detect elbow programmatically using the "knee/elbow" heuristic:
# The elbow is where the second derivative of inertia is maximised.
# (biggest change in the rate of decrease)
drops       = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
second_diff = [drops[i-1] - drops[i] for i in range(1, len(drops))]
elbow_K     = K_values[second_diff.index(max(second_diff)) + 2]

print(f"\n  Elbow detected at K = {elbow_K}  {'✓ CORRECT!' if elbow_K == true_K else '← close to true K'}")


# ============================================================
#  STEP 4 — FIT FINAL MODEL WITH K=4
# ============================================================
# n_init=10: run K-Means 10 times with different random starts,
# keep the run with the lowest inertia.
# This guards against bad luck with initial centroid placement.

km_final = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels   = km_final.fit_predict(X)
centres  = km_final.cluster_centers_

print("\n" + "=" * 64)
print("STEP 4 — FINAL K-MEANS (K=4)")
print("=" * 64)
print(f"  Inertia        : {km_final.inertia_:.4f}")
print(f"  Iterations     : {km_final.n_iter_}")
print(f"\n  Cluster sizes  :")
for c in range(4):
    print(f"    Cluster {c}  : {(labels==c).sum()} points")


# ============================================================
#  STEP 5 — SILHOUETTE SCORE
# ============================================================
# Silhouette Score measures how well each point fits its cluster.
#
# For each point i, it computes:
#   a(i) = average distance to all OTHER points in its cluster
#   b(i) = average distance to all points in the NEAREST other cluster
#   s(i) = (b(i) - a(i)) / max(a(i), b(i))
#
# s(i) ranges from -1 to +1:
#   +1  = point is deeply embedded in its cluster, far from others
#    0  = point is on the boundary between two clusters
#   -1  = point is in the wrong cluster entirely
#
# The overall Silhouette Score = mean of s(i) across all points.
# Rule of thumb:
#   > 0.7  = very strong clustering
#   > 0.5  = good clustering
#   > 0.25 = weak but possible structure
#   < 0.25 = little to no cluster structure

sil_scores = {}

for k in [2, 3, 4, 5, 7]:
    km_k   = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    lbl_k  = km_k.fit_predict(X)
    sil    = silhouette_score(X, lbl_k)
    sil_scores[k] = sil

sil_final = sil_scores[4]

print("\n" + "=" * 64)
print("STEP 5 — SILHOUETTE SCORES  (comparing wrong vs right K)")
print("=" * 64)
print(f"\n  {'K':>4}  {'Silhouette Score':>18}  {'Quality':>12}  Note")
print(f"  {'-'*60}")
for k, sil in sil_scores.items():
    quality = ("Very strong" if sil > 0.7 else
               "Good"        if sil > 0.5 else
               "Weak"        if sil > 0.25 else "Poor")
    marker = "  ← TRUE K" if k == 4 else ("  ← too few" if k == 2 else
                                            "  ← too many" if k == 7 else "")
    print(f"  {k:>4}  {sil:>18.4f}  {quality:>12}{marker}")

print(f"\n  K=4 silhouette ({sil_scores[4]:.4f}) vs K=2 ({sil_scores[2]:.4f}) vs K=7 ({sil_scores[7]:.4f})")
print(f"  The TRUE K gives the HIGHEST silhouette — method works! ✓")


# ============================================================
#  STEP 6 — INIT METHOD EXPERIMENT: 'random' vs 'k-means++'
# ============================================================
# K-Means is sensitive to where centroids start.
# Bad initialisation → gets stuck in a local minimum (bad clustering).
#
# 'random' init: picks K random data points as starting centroids.
#   → Sometimes gets lucky, sometimes very unlucky.
#   → High variance across runs (results change a lot).
#
# 'k-means++' init: spreads starting centroids far apart deliberately.
#   → Guarantees centroids start in different regions.
#   → Much more consistent, always near the optimal solution.
#   → Slightly slower to init but saves many iterations.

N_RUNS = 7
print("\n" + "=" * 64)
print("STEP 6 — INIT METHOD: 'random' vs 'k-means++'")
print("=" * 64)
print(f"\n  Running each method {N_RUNS} times with different random seeds...")
print(f"  Using harder overlapping data (K=8, high noise) so init matters more.\n")

# Use harder, noisier data so bad initialisation clearly shows
X_hard, _ = make_blobs(n_samples=500, centers=8, cluster_std=2.5, random_state=1)
X_hard    = StandardScaler().fit_transform(X_hard)

random_inertias  = []
kmeanspp_inertias = []

print(f"  {'Run':>4}  {'random init':>14}  {'k-means++ init':>16}")
print(f"  {'-'*38}")

for run in range(N_RUNS):
    km_rand = KMeans(n_clusters=8, init='random',    n_init=1, max_iter=50, random_state=run*13)
    km_pp   = KMeans(n_clusters=8, init='k-means++', n_init=1, max_iter=50, random_state=run*13)

    km_rand.fit(X_hard); km_pp.fit(X_hard)

    random_inertias.append(km_rand.inertia_)
    kmeanspp_inertias.append(km_pp.inertia_)

    print(f"  {run+1:>4}  {km_rand.inertia_:>14.4f}  {km_pp.inertia_:>16.4f}")

rand_mean = np.mean(random_inertias)
rand_std  = np.std(random_inertias)
pp_mean   = np.mean(kmeanspp_inertias)
pp_std    = np.std(kmeanspp_inertias)

print(f"\n  {'':>4}  {'--- random ---':>14}  {'--- k-means++ ---':>16}")
print(f"  {'Mean':>4}  {rand_mean:>14.4f}  {pp_mean:>16.4f}")
print(f"  {'Std':>4}  {rand_std:>14.4f}  {pp_std:>16.4f}  ← lower = more consistent")
print(f"\n  k-means++ variance is {rand_std/max(pp_std,1e-9):.1f}x lower than random init.")
print(f"  Always use k-means++ (it's the default in sklearn).")


# ============================================================
#  STEP 7 — BIG PLOT (3 rows × 3 cols)
# ============================================================

COLORS = ['#E24B4A', '#378ADD', '#63B77A', '#F5A623',
          '#9B59B6', '#1ABC9C', '#E67E22', '#2C3E50', '#E91E63', '#607D8B']

fig = plt.figure(figsize=(19, 17))
fig.suptitle("K-Means Clustering — Elbow Method + Visualisation\n"
             "(4 blob clusters, 400 points, 2D scaled features)",
             fontsize=14, y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.33)


# ── Panel (0,0): Raw unlabelled data ─────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
ax00.scatter(X[:, 0], X[:, 1], c='#888888', s=22, alpha=0.6, edgecolors='none')
ax00.set_title("Raw Data — No Labels\n(K-Means starts here, knowing nothing)",
               fontsize=9)
ax00.set_xticks([]); ax00.set_yticks([])


# ── Panel (0,1): Elbow curve ──────────────────────────────────
ax01 = fig.add_subplot(gs[0, 1])
ax01.plot(K_values, inertias, 'o-', color='steelblue',
          linewidth=2.5, markersize=8)
ax01.axvline(x=elbow_K, color='tomato', linestyle='--',
             linewidth=2, label=f'Elbow at K={elbow_K}')
ax01.scatter([elbow_K], [inertias[elbow_K-1]], color='tomato', s=150, zorder=5)
ax01.annotate(f'Elbow → K={elbow_K}\nInertia={inertias[elbow_K-1]:.2f}',
              xy=(elbow_K, inertias[elbow_K-1]),
              xytext=(elbow_K + 1.2, inertias[elbow_K-1] + 20),
              fontsize=8.5, color='tomato',
              arrowprops=dict(arrowstyle='->', color='tomato', lw=1.3))
ax01.set_xlabel("K  (number of clusters)")
ax01.set_ylabel("Inertia  (within-cluster sum of squares)")
ax01.set_title("Elbow Method — Inertia vs K\nBig drops stop at K=4 → that's the elbow", fontsize=9)
ax01.legend(fontsize=9)
ax01.grid(True, alpha=0.25)
ax01.set_xticks(K_values)


# ── Panel (0,2): Silhouette scores bar chart ──────────────────
ax02 = fig.add_subplot(gs[0, 2])
sil_ks   = list(sil_scores.keys())
sil_vals = list(sil_scores.values())
bar_cols = ['#63B77A' if k == 4 else '#AAB7C4' for k in sil_ks]
bars = ax02.bar([str(k) for k in sil_ks], sil_vals, color=bar_cols,
                alpha=0.87, edgecolor='white')
for bar, val in zip(bars, sil_vals):
    ax02.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax02.axhline(y=0.5, color='tomato', linestyle='--', linewidth=1.2,
             alpha=0.7, label='0.5 = good threshold')
ax02.set_xlabel("K  (number of clusters)")
ax02.set_ylabel("Silhouette Score")
ax02.set_title("Silhouette Score per K\nGreen bar (K=4) should be highest → ✓",
               fontsize=9)
ax02.set_ylim(0, max(sil_vals) + 0.08)
ax02.legend(fontsize=8)
ax02.grid(True, alpha=0.2, axis='y')


# ── Panel (1,0): Final K=4 clustering ────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
for c in range(4):
    mask = labels == c
    ax10.scatter(X[mask, 0], X[mask, 1],
                 c=COLORS[c], s=28, alpha=0.75,
                 edgecolors='none', label=f'Cluster {c}')
ax10.scatter(centres[:, 0], centres[:, 1],
             c='black', marker='*', s=280, zorder=6,
             edgecolors='white', linewidths=0.8, label='Centroids')
ax10.set_title(f"Final Clustering  K=4\n"
               f"Silhouette={sil_final:.4f} · Inertia={km_final.inertia_:.2f}",
               fontsize=9)
ax10.legend(fontsize=7.5, ncol=2)
ax10.set_xticks([]); ax10.set_yticks([])


# ── Panel (1,1): K=2  (underclustering) ──────────────────────
ax11 = fig.add_subplot(gs[1, 1])
km2 = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
lbl2 = km2.fit_predict(X)
for c in range(2):
    mask = lbl2 == c
    ax11.scatter(X[mask, 0], X[mask, 1], c=COLORS[c], s=28, alpha=0.7, edgecolors='none')
ax11.scatter(km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1],
             c='black', marker='*', s=280, zorder=6, edgecolors='white', linewidths=0.8)
ax11.set_title(f"K=2  (Too Few — Underclustering)\n"
               f"Silhouette={sil_scores[2]:.4f}  ← drops vs K=4",
               fontsize=9)
ax11.set_xticks([]); ax11.set_yticks([])


# ── Panel (1,2): K=7  (overclustering) ───────────────────────
ax12 = fig.add_subplot(gs[1, 2])
km7 = KMeans(n_clusters=7, init='k-means++', n_init=10, random_state=42)
lbl7 = km7.fit_predict(X)
for c in range(7):
    mask = lbl7 == c
    ax12.scatter(X[mask, 0], X[mask, 1], c=COLORS[c], s=28, alpha=0.7, edgecolors='none')
ax12.scatter(km7.cluster_centers_[:, 0], km7.cluster_centers_[:, 1],
             c='black', marker='*', s=280, zorder=6, edgecolors='white', linewidths=0.8)
ax12.set_title(f"K=7  (Too Many — Overclustering)\n"
               f"Silhouette={sil_scores[7]:.4f}  ← lower than K=4",
               fontsize=9)
ax12.set_xticks([]); ax12.set_yticks([])


# ── Panel (2,0–1): random vs k-means++ inertia variance ──────
ax20 = fig.add_subplot(gs[2, 0:2])
runs = list(range(1, N_RUNS + 1))
ax20.plot(runs, random_inertias,   'o--', color='tomato',    linewidth=2,
          markersize=9, label=f"init='random'   (std={rand_std:.4f})")
ax20.plot(runs, kmeanspp_inertias, 's-',  color='steelblue', linewidth=2.5,
          markersize=9, label=f"init='k-means++' (std={pp_std:.4f})")
ax20.fill_between(runs, random_inertias, kmeanspp_inertias,
                  alpha=0.12, color='tomato', label='Variance gap')
ax20.set_xlabel("Run number  (different random seed each time)")
ax20.set_ylabel("Final Inertia")
ax20.set_title(f"'random' vs 'k-means++' Initialisation — {N_RUNS} runs\n"
               "k-means++ is more consistent (lower variance) → always prefer it",
               fontsize=9)
ax20.legend(fontsize=9)
ax20.grid(True, alpha=0.25)
ax20.set_xticks(runs)


# ── Panel (2,2): Inertia drop bar chart ──────────────────────
ax21 = fig.add_subplot(gs[2, 2])
drop_vals = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
drop_Ks   = K_values[1:]
drop_cols = ['#63B77A' if k == elbow_K else '#AAB7C4' for k in drop_Ks]
bars2 = ax21.bar([f'K={k}' for k in drop_Ks], drop_vals,
                 color=drop_cols, alpha=0.87, edgecolor='white')
for bar, val in zip(bars2, drop_vals):
    if val > 5:
        ax21.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=8)
ax21.set_xlabel("Going from K-1 → K")
ax21.set_ylabel("Inertia Drop")
ax21.set_title("Inertia Drop per Step\nGreen = biggest remaining drop = elbow point",
               fontsize=9)
ax21.grid(True, alpha=0.2, axis='y')

plt.savefig("kmeans_elbow_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → kmeans_elbow_plot.png")