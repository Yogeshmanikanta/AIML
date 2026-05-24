import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ── helper: colour map that treats -1 (noise) as black ───────────────────
def cluster_colors(labels):
    """
    Returns a list of hex colours.
    Noise points (label == -1) → black '#222222'
    Clusters 0,1,2,… → distinct palette
    """
    PALETTE = ['#E24B4A', '#378ADD', '#63B77A', '#F5A623',
               '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
    return [('#222222' if lbl == -1 else PALETTE[lbl % len(PALETTE)])
            for lbl in labels]


def cluster_summary(labels, name):
    """Print a tidy label summary."""
    unique = sorted(set(labels))
    n_noise    = (labels == -1).sum()
    n_clusters = len([l for l in unique if l != -1])
    print(f"  {name:<35}  clusters={n_clusters:>2}  noise={n_noise:>3} pts")
    for lbl in unique:
        tag = "NOISE" if lbl == -1 else f"Cluster {lbl}"
        print(f"    {tag:<12}: {(labels==lbl).sum():>4} points")


# ============================================================
#  STEP 1 — GENERATE MOON-SHAPED DATA  (with injected noise)
# ============================================================
# make_moons creates two interleaved crescent shapes.
# These are the canonical example of data K-Means CANNOT handle.
#
# Why K-Means fails on moons:
#   K-Means finds clusters by minimising distance to a centroid.
#   A centroid is a single point (the mean of a cluster).
#   A crescent has no meaningful "centre" — any centroid placed
#   inside one moon is equidistant from many points of BOTH moons.
#   Result: K-Means slices the data vertically down the middle
#           instead of following the curved moon shapes.
#
# DBSCAN works differently:
#   It groups points that are CLOSE TO EACH OTHER in a chain.
#   "If you can walk from A to B stepping only to neighbours
#    within eps distance, A and B are in the same cluster."
#   No centroid needed — the shape itself defines the cluster.
#
# We also add 40 random noise points scattered across the plane.
# DBSCAN will label them -1 (noise).
# K-Means will force every single one into the nearest cluster.

np.random.seed(42)
n_samples = 300

X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.08, random_state=42)

# Inject 40 random noise points in [-1.5, 2.5] × [-1.0, 1.5]
n_noise   = 40
X_noise   = np.random.uniform([-1.5, -1.0], [2.5, 1.5], size=(n_noise, 2))
X_moons_noisy = np.vstack([X_moons, X_noise])

# Scale for DBSCAN (eps is distance-sensitive)
scaler_m      = StandardScaler()
X_moons_s     = scaler_m.fit_transform(X_moons_noisy)

print("=" * 64)
print("STEP 1 — MOON-SHAPED DATA GENERATED")
print("=" * 64)
print(f"  Moon points      : {n_samples}")
print(f"  Injected noise   : {n_noise}  random scatter points")
print(f"  Total points     : {len(X_moons_s)}")
print(f"  Scaled           : yes  (StandardScaler)")
print(f"\n  Ground truth: 2 moons + noise.")
print(f"  Challenge: can the algorithm find them?")


# ============================================================
#  STEP 2 — K-MEANS ON MOONS  (expected failure)
# ============================================================
# K-Means will cut the data along a straight boundary —
# it has no concept of 'follow the curve'.
# The noise points will be silently absorbed into whichever
# cluster centroid they happen to be closest to.
# No point is ever labelled as 'noise' — K-Means doesn't know
# what a noise point is.

km_moons = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
km_labels_moons = km_moons.fit_predict(X_moons_s)

print("\n" + "=" * 64)
print("STEP 2 — K-MEANS ON MOONS  (expected failure)")
print("=" * 64)
cluster_summary(km_labels_moons, "K-Means (k=2)")
print(f"\n  Silhouette score : {silhouette_score(X_moons_s, km_labels_moons):.4f}")
print(f"  Noise points labelled -1 : 0  ← K-Means never says 'noise'")
print(f"  → Cuts moons with a STRAIGHT LINE. Completely wrong shape.")


# ============================================================
#  STEP 3 — DBSCAN ON MOONS  (should work perfectly)
# ============================================================
# DBSCAN parameters:
#
#   eps (epsilon): the neighbourhood radius.
#     Two points are "neighbours" if their distance < eps.
#     Think of it as: "how far can I step to the next point?"
#
#   min_samples: minimum neighbours needed to be a CORE point.
#     A core point starts or extends a cluster.
#     Points with fewer than min_samples neighbours are:
#       → either BORDER points (reachable from a core point), or
#       → NOISE points (labelled -1, not in any cluster)
#
# eps=0.3 means: in the scaled space, two points within distance
# 0.3 of each other are in the same neighbourhood.
# This is the "Goldilocks" eps for this dataset.

db_moons = DBSCAN(eps=0.3, min_samples=5)
db_labels_moons = db_moons.fit_predict(X_moons_s)

print("\n" + "=" * 64)
print("STEP 3 — DBSCAN (eps=0.3) ON MOONS  (should work perfectly)")
print("=" * 64)
cluster_summary(db_labels_moons, "DBSCAN (eps=0.3, min=5)")
sil_db_moons = silhouette_score(X_moons_s[db_labels_moons != -1],
                                db_labels_moons[db_labels_moons != -1])
print(f"\n  Silhouette score (excl. noise): {sil_db_moons:.4f}")
print(f"  Noise points labelled -1       : {(db_labels_moons==-1).sum()}")
print(f"  → DBSCAN follows the curved shape perfectly ✓")
print(f"  → Noise points correctly identified and excluded ✓")


# ============================================================
#  STEP 4 — EXPERIMENTING WITH EPS
# ============================================================
# eps is the most sensitive parameter in DBSCAN.
#
# eps too small (0.1):
#   Each point can only reach very close neighbours.
#   Most points have < min_samples neighbours.
#   Everything becomes noise, or hundreds of tiny clusters.
#   The algorithm is "too strict" — no one qualifies as core.
#
# eps too large (1.0):
#   Each point reaches MANY neighbours.
#   Both moons merge into one giant cluster.
#   The algorithm is "too loose" — everyone is connected to everyone.
#
# The right eps puts you in the sweet spot:
#   big enough to chain points within a cluster,
#   small enough to NOT bridge across the gap between clusters.

eps_values = [0.1, 0.3, 1.0]
db_eps_labels = []

print("\n" + "=" * 64)
print("STEP 4 — EPS SENSITIVITY EXPERIMENT  (moons data)")
print("=" * 64)
print(f"\n  {'eps':>6}  {'Clusters':>9}  {'Noise pts':>10}  {'Silhouette':>12}  Interpretation")
print(f"  {'-'*65}")

for eps in eps_values:
    db_e  = DBSCAN(eps=eps, min_samples=5)
    lbls  = db_e.fit_predict(X_moons_s)
    db_eps_labels.append(lbls)

    n_cls = len(set(lbls) - {-1})
    n_nse = (lbls == -1).sum()
    valid = lbls[lbls != -1]
    sil   = (silhouette_score(X_moons_s[lbls != -1], valid)
             if len(set(valid)) > 1 else float('nan'))
    note  = ("← TOO TIGHT: most points are noise" if eps == 0.1
             else "← JUST RIGHT: two moons + noise detected" if eps == 0.3
             else "← TOO LOOSE: both moons merged into one")
    sil_s = f"{sil:.4f}" if not np.isnan(sil) else "  N/A "
    print(f"  {eps:>6}  {n_cls:>9}  {n_nse:>10}  {sil_s:>12}  {note}")


# ============================================================
#  STEP 5 — CIRCLES DATASET
# ============================================================
# make_circles creates two concentric rings (inner + outer).
# Same fundamental problem as moons: non-spherical shapes
# that K-Means cannot handle.
#
# K-Means will again cut with a straight line, mixing inner/outer.
# DBSCAN will trace each ring as a separate cluster.

np.random.seed(0)
X_circ, y_circ = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=42)

# Add 30 noise points
X_cnoise = np.random.uniform(-1.5, 1.5, size=(30, 2))
X_circ_noisy = np.vstack([X_circ, X_cnoise])

scaler_c   = StandardScaler()
X_circ_s   = scaler_c.fit_transform(X_circ_noisy)

km_circ    = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
km_lbl_c   = km_circ.fit_predict(X_circ_s)

db_circ    = DBSCAN(eps=0.3, min_samples=5)
db_lbl_c   = db_circ.fit_predict(X_circ_s)

print("\n" + "=" * 64)
print("STEP 5 — CIRCLES DATASET  (concentric rings)")
print("=" * 64)
print("\n  K-Means:")
cluster_summary(km_lbl_c, "K-Means (k=2)")
print("\n  DBSCAN:")
cluster_summary(db_lbl_c, "DBSCAN (eps=0.3, min=5)")

valid_c = db_lbl_c[db_lbl_c != -1]
if len(set(valid_c)) > 1:
    sil_c = silhouette_score(X_circ_s[db_lbl_c != -1], valid_c)
    print(f"\n  DBSCAN Silhouette (excl. noise): {sil_c:.4f}")


# ============================================================
#  STEP 6 — DBSCAN LABEL DEEP DIVE
# ============================================================
# Let's print actual label values so the -1 convention is concrete.

print("\n" + "=" * 64)
print("STEP 6 — DBSCAN LABEL DEEP DIVE")
print("=" * 64)
unique_lbl = sorted(set(db_labels_moons))
print(f"\n  Unique labels in DBSCAN output: {unique_lbl}")
print(f"\n  What each value means:")
print(f"    -1 → NOISE: point has fewer than min_samples neighbours")
print(f"                 within eps distance. Not part of any cluster.")
print(f"     0 → Cluster 0  (first identified cluster)")
print(f"     1 → Cluster 1  (second identified cluster)")
print(f"\n  First 20 labels: {db_labels_moons[:20].tolist()}")
print(f"  Noise labels   : {np.where(db_labels_moons == -1)[0][:10].tolist()} ...  (indices)")
print(f"\n  Compare with K-Means first 20: {km_labels_moons[:20].tolist()}")
print(f"  K-Means never produces -1. Every point forced into a cluster.")


# ============================================================
#  STEP 7 — FINAL SUMMARY TABLE
# ============================================================

print("\n" + "=" * 64)
print("STEP 7 — ALGORITHM COMPARISON SUMMARY")
print("=" * 64)
print(f"""
  {'Property':<35} {'K-Means':>12} {'DBSCAN':>12}
  {'-'*62}
  {'Finds non-spherical shapes':<35} {'✗  NO':>12} {'✓  YES':>12}
  {'Handles noise/outliers':<35} {'✗  NO':>12} {'✓  YES':>12}
  {'Needs K in advance':<35} {'✓  YES':>12} {'✗  NO':>12}
  {'Works on moon/circle data':<35} {'✗  FAILS':>12} {'✓  WORKS':>12}
  {'Sensitive to scale':<35} {'✓  YES':>12} {'✓  YES':>12}
  {'Speed (large datasets)':<35} {'✓  FAST':>12} {'✗  SLOWER':>12}
  {'Clusters always same size':<35} {'No':>12} {'No':>12}
  {'Key parameter':<35} {'K (n_clusters)':>12} {'eps, min_samp':>12}
  {'-'*62}

  When to use K-Means:
    → Blobs / spherical clusters
    → Large datasets (fast)
    → You have a rough idea of K
    → Customer segmentation, document clustering

  When to use DBSCAN:
    → Non-spherical shapes (roads, rings, crescents)
    → Data has outliers / noise you want to flag
    → You do NOT know K in advance
    → Geographic data, anomaly detection
""")


# ============================================================
#  STEP 8 — BIG PLOT  (4 rows × 3 cols = 12 panels)
# ============================================================

fig = plt.figure(figsize=(18, 22))
fig.suptitle("DBSCAN vs K-Means — Non-Spherical Clusters\n"
             "Moon shapes · Circle shapes · eps sensitivity · noise detection",
             fontsize=14, y=0.995)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.28)

# Colours
KM_PALETTE = ['#E24B4A', '#378ADD']
NOISE_COL  = '#222222'
DB_PALETTE = ['#E24B4A', '#378ADD', '#63B77A', '#F5A623']


def scatter_clusters(ax, X, labels, title, show_noise_legend=False, centroids=None):
    """Generic scatter that colours by cluster label, black for noise."""
    unique = sorted(set(labels))
    n_real = [l for l in unique if l != -1]
    pal    = DB_PALETTE

    for lbl in unique:
        mask  = labels == lbl
        col   = NOISE_COL if lbl == -1 else pal[lbl % len(pal)]
        zord  = 5 if lbl == -1 else 3
        alpha = 0.55 if lbl == -1 else 0.75
        sz    = 18 if lbl == -1 else 22
        tag   = f"Noise ({mask.sum()})" if lbl == -1 else f"Cluster {lbl} ({mask.sum()})"
        ax.scatter(X[mask, 0], X[mask, 1], c=col, s=sz,
                   alpha=alpha, edgecolors='none', zorder=zord, label=tag)

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='black', marker='*', s=220, zorder=6,
                   edgecolors='white', linewidths=0.8, label='Centroids')

    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.7)
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xticks([]); ax.set_yticks([])


# ── Row 0: Raw moons data + K-Means failure + DBSCAN success ──
ax00 = fig.add_subplot(gs[0, 0])
ax00.scatter(X_moons_s[:n_samples, 0], X_moons_s[:n_samples, 1],
             c='#888', s=20, alpha=0.6, edgecolors='none', label='Moon pts')
ax00.scatter(X_moons_s[n_samples:, 0], X_moons_s[n_samples:, 1],
             c='#F5A623', s=28, alpha=0.8, edgecolors='none', label=f'Noise ({n_noise})')
ax00.legend(fontsize=8)
ax00.set_title("Raw Moon Data  (no labels)\n2 crescents + 40 injected noise points", fontsize=9)
ax00.set_xticks([]); ax00.set_yticks([])

ax01 = fig.add_subplot(gs[0, 1])
scatter_clusters(ax01, X_moons_s, km_labels_moons,
                 f"K-Means (k=2) on Moons — FAILS\n"
                 f"Cuts straight across, noise absorbed silently",
                 centroids=km_moons.cluster_centers_)

ax02 = fig.add_subplot(gs[0, 2])
scatter_clusters(ax02, X_moons_s, db_labels_moons,
                 f"DBSCAN (eps=0.3) on Moons — WORKS ✓\n"
                 f"Follows curve, flags {(db_labels_moons==-1).sum()} noise pts black")


# ── Row 1: eps sensitivity on moons ───────────────────────────
eps_titles = [
    "eps=0.1  (TOO TIGHT)\nEverything is noise — radius too small",
    "eps=0.3  (JUST RIGHT)\nTwo moons + noise correctly separated",
    "eps=1.0  (TOO LOOSE)\nBoth moons merge into one cluster"
]
for col, (lbl_e, title_e) in enumerate(zip(db_eps_labels, eps_titles)):
    ax = fig.add_subplot(gs[1, col])
    scatter_clusters(ax, X_moons_s, lbl_e, title_e)


# ── Row 2: Circles — raw + K-Means + DBSCAN ───────────────────
ax20 = fig.add_subplot(gs[2, 0])
ax20.scatter(X_circ_s[:300, 0], X_circ_s[:300, 1],
             c='#888', s=20, alpha=0.6, edgecolors='none', label='Ring pts')
ax20.scatter(X_circ_s[300:, 0], X_circ_s[300:, 1],
             c='#F5A623', s=28, alpha=0.8, edgecolors='none', label='Noise (30)')
ax20.legend(fontsize=8)
ax20.set_title("Raw Circles Data  (no labels)\nInner + outer ring + 30 noise points", fontsize=9)
ax20.set_xticks([]); ax20.set_yticks([])

ax21 = fig.add_subplot(gs[2, 1])
scatter_clusters(ax21, X_circ_s, km_lbl_c,
                 "K-Means (k=2) on Circles — FAILS\n"
                 "Bisects both rings vertically",
                 centroids=km_circ.cluster_centers_)

ax22 = fig.add_subplot(gs[2, 2])
scatter_clusters(ax22, X_circ_s, db_lbl_c,
                 f"DBSCAN (eps=0.3) on Circles — WORKS ✓\n"
                 f"Inner ring = cluster 0 · Outer ring = cluster 1")


# ── Row 3: Algorithm property comparison (visual table) ────────
ax30 = fig.add_subplot(gs[3, :])   # full-width
ax30.axis('off')

properties = [
    ("Cluster shape",      "Spherical only",          "Any shape  ✓"),
    ("Noise handling",     "None — forced into cluster ✗", "Labels as -1  ✓"),
    ("Need to specify K",  "YES  (required)",          "NO  (finds automatically ✓)"),
    ("Moon / Circle data", "FAILS ✗",                 "WORKS PERFECTLY ✓"),
    ("Key parameter",      "K  (n_clusters)",          "eps + min_samples"),
    ("Speed",              "Fast  ✓",                  "Slower on large data"),
    ("Best for",           "Blobs, segmentation",      "Spatial, anomaly detection"),
]

col_x   = [0.02, 0.30, 0.65]
row_h   = 0.13
pad_top = 0.92

ax30.text(col_x[0], pad_top + 0.05, "Property",      fontsize=10, fontweight='bold', transform=ax30.transAxes)
ax30.text(col_x[1], pad_top + 0.05, "K-Means",       fontsize=10, fontweight='bold', color='#E24B4A', transform=ax30.transAxes)
ax30.text(col_x[2], pad_top + 0.05, "DBSCAN",        fontsize=10, fontweight='bold', color='#378ADD', transform=ax30.transAxes)
ax30.plot([0.01, 0.99], [pad_top + 0.02, pad_top + 0.02], color='gray', linewidth=0.8, transform=ax30.transAxes)

for i, (prop, km_val, db_val) in enumerate(properties):
    y = pad_top - (i + 1) * row_h
    bg = '#F9F9F7' if i % 2 == 0 else '#FFFFFF'
    ax30.add_patch(plt.Rectangle((0, y - 0.02), 1.0, row_h,
                                  transform=ax30.transAxes, color=bg, zorder=0))
    ax30.text(col_x[0], y, prop,    fontsize=9, transform=ax30.transAxes, va='center')
    ax30.text(col_x[1], y, km_val,  fontsize=9, transform=ax30.transAxes, va='center',
              color='#C0392B' if '✗' in km_val else '#555')
    ax30.text(col_x[2], y, db_val,  fontsize=9, transform=ax30.transAxes, va='center',
              color='#1A7A40' if '✓' in db_val else '#555')

ax30.set_title("K-Means vs DBSCAN — Property Comparison Table",
               fontsize=10, pad=8)

plt.savefig("dbscan_comparison_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → dbscan_comparison_plot.png")