import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
#  STEP 1 — LOAD THE DIGITS DATASET
# ============================================================
# 1,797 handwritten digit images (0-9), each 8×8 pixels.
# Flattened into a 64-feature vector per image.
#
# Why this dataset is perfect for PCA:
#   64 features sounds manageable, but many pixels are CORRELATED.
#   Corner pixels are almost always black across all digits.
#   Adjacent pixels tend to move together (edges, curves).
#   PCA finds these correlations and collapses redundant features
#   into a smaller set of "principal components" that capture
#   the most variance with the fewest dimensions.
#
# What PCA does mathematically:
#   1. Centres the data (zero mean)
#   2. Finds directions of maximum variance (eigenvectors)
#   3. Projects data onto those directions
#   4. Orders components by how much variance they explain
#
# Result: 64 correlated pixels → K uncorrelated components
# where K << 64 but still captures most of the information.

digits = load_digits()
X      = digits.data     # shape: (1797, 64)
y      = digits.target   # 0-9 digit labels

print("=" * 64)
print("STEP 1 — DIGITS DATASET LOADED")
print("=" * 64)
print(f"  Total images   : {X.shape[0]}")
print(f"  Features/image : {X.shape[1]}  (8×8 pixels, flattened)")
print(f"  Classes        : {sorted(set(y))}  (digits 0–9)")
print(f"  Samples/class  : ~{len(y)//10}")
print(f"\n  Each image is 8×8 = 64 pixel values (0=black, 16=white).")
print(f"  Challenge: 64 features is high-dimensional. PCA compresses it.")


# ============================================================
#  STEP 2 — SCALE + PCA TO 2 COMPONENTS
# ============================================================
# StandardScaler first — PCA is variance-based, so features with
# larger numerical ranges would dominate without scaling.
# Pixel values are already 0–16 (similar range), but we scale
# anyway as best practice.
#
# PCA(n_components=2) keeps only the top 2 directions of variance.
# This loses a lot of information (64 → 2 is aggressive),
# but it lets us VISUALISE in 2D — which is the whole point here.
# We'll find the optimal components for accuracy in Step 5.

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

pca_2 = PCA(n_components=2, random_state=42)
X_train_2d = pca_2.fit_transform(X_train_scaled)
X_test_2d  = pca_2.transform(X_test_scaled)

var_2 = pca_2.explained_variance_ratio_
cumvar_2 = var_2.sum()

print("\n" + "=" * 64)
print("STEP 2 — PCA TO 2 COMPONENTS  (for visualization)")
print("=" * 64)
print(f"  Original features    : 64")
print(f"  After PCA            : 2")
print(f"  Compression ratio    : {64/2:.0f}× smaller")
print(f"\n  Explained variance:")
print(f"    Component 1 : {var_2[0]*100:.2f}%")
print(f"    Component 2 : {var_2[1]*100:.2f}%")
print(f"    Total kept  : {cumvar_2*100:.2f}%  ← remaining {100-cumvar_2*100:.2f}% lost")
print(f"\n  Note: 2 components keep only {cumvar_2*100:.1f}% of variance.")
print(f"  This is lossy compression — good for visualization, not prediction.")


# ============================================================
#  STEP 3 — EXPLAINED VARIANCE ACROSS ALL COMPONENTS
# ============================================================
# Fit PCA with all 64 components so we can plot the full
# variance curve and identify the 95% threshold.
#
# Scree plot: each bar = one principal component.
# The curve drops steeply at first (first few components carry
# most variance) then flattens (later components add little).
# This "elbow" shape is where PCA shows its power:
# a few components do most of the work.

pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)

evr       = pca_full.explained_variance_ratio_     # per-component variance
cumsum_evr = np.cumsum(evr)                         # running total

# Find how many components needed for 95% variance
n_95 = np.argmax(cumsum_evr >= 0.95) + 1

print("\n" + "=" * 64)
print("STEP 3 — VARIANCE EXPLAINED BY EACH COMPONENT")
print("=" * 64)
print(f"  {'Components':>12}  {'Cumulative Variance':>20}")
print(f"  {'-'*35}")
for n in [1, 2, 5, 10, 20, 30, n_95, 64]:
    n = min(n, 64)
    flag = "  ← 95% threshold" if n == n_95 else ""
    print(f"  {n:>12}  {cumsum_evr[n-1]*100:>19.2f}%{flag}")

print(f"\n  Components needed for 95% variance : {n_95}")
print(f"  Compression achieved               : {64} → {n_95}  ({100*(1-n_95/64):.0f}% fewer features)")
print(f"  First 2 components alone           : {cumvar_2*100:.1f}%  (enough to see clusters)")


# ============================================================
#  STEP 4 — PCA WITH 95% VARIANCE THRESHOLD
# ============================================================
# PCA(n_components=0.95) automatically selects however many
# components are needed to keep 95% of the variance.
# sklearn figures out the right number for us.
#
# This is the practical way to use PCA in a ML pipeline:
# don't pick an arbitrary number — pick a variance threshold.

pca_95 = PCA(n_components=0.95, random_state=42)
X_train_95 = pca_95.fit_transform(X_train_scaled)
X_test_95  = pca_95.transform(X_test_scaled)

print("\n" + "=" * 64)
print("STEP 4 — PCA(n_components=0.95)  — AUTO-SELECT COMPONENTS")
print("=" * 64)
print(f"  sklearn auto-selected : {pca_95.n_components_} components")
print(f"  Variance retained     : {pca_95.explained_variance_ratio_.sum()*100:.2f}%")
print(f"  Shape: {X_train_scaled.shape}  →  {X_train_95.shape}")


# ============================================================
#  STEP 5 — TRAIN LOGISTIC REGRESSION: RAW vs PCA-95% vs PCA-2D
# ============================================================
# Now the key experiment: does PCA hurt accuracy?
# We train the same model on three versions of the data:
#   1. Raw scaled   : 64 features  — full information
#   2. PCA 95%      : ~{n_95} features — 95% of information
#   3. PCA 2D       : 2 features   — only for visualization
#
# Expected results:
#   Raw → best accuracy  (all info available)
#   PCA 95% → nearly same accuracy  (5% loss is mostly noise)
#   PCA 2D → noticeably worse  (28% info loss is too much)
#
# PCA also acts as a noise filter:
#   The dropped components often capture noise, not signal.
#   So PCA-95% sometimes MATCHES or BEATS raw accuracy.

configs = [
    ("Raw (64 features)",       X_train_scaled, X_test_scaled),
    (f"PCA 95% ({pca_95.n_components_} features)",   X_train_95,    X_test_95),
    ("PCA 2D  (2 features)",    X_train_2d,    X_test_2d),
]

results = {}
print("\n" + "=" * 64)
print("STEP 5 — LOGISTIC REGRESSION: RAW vs PCA COMPARISON")
print("=" * 64)
print(f"\n  {'Config':<30}  {'Train time':>12}  {'Predict time':>13}  {'Accuracy':>10}")
print(f"  {'-'*70}")

for name, Xtr, Xte in configs:
    lr = LogisticRegression(max_iter=5000, random_state=42)

    t0 = time.perf_counter()
    lr.fit(Xtr, y_train)
    train_t = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    y_pred = lr.predict(Xte)
    pred_t = (time.perf_counter() - t1) * 1000

    acc = accuracy_score(y_test, y_pred)
    results[name] = {"acc": acc, "train_ms": train_t, "pred_ms": pred_t, "pred": y_pred}

    print(f"  {name:<30}  {train_t:>10.1f}ms  {pred_t:>11.1f}ms  {acc*100:>9.2f}%")

raw_acc   = results["Raw (64 features)"]["acc"]
pca95_acc = results[f"PCA 95% ({pca_95.n_components_} features)"]["acc"]
pca2_acc  = results["PCA 2D  (2 features)"]["acc"]

raw_t   = results["Raw (64 features)"]["train_ms"]
pca95_t = results[f"PCA 95% ({pca_95.n_components_} features)"]["train_ms"]

acc_drop   = (raw_acc - pca95_acc) * 100
speedup    = raw_t / pca95_t if pca95_t > 0 else float('inf')

print(f"\n  Accuracy drop (Raw → PCA 95%)  : {acc_drop:+.2f}%")
print(f"  Training speedup               : {speedup:.1f}×  faster with PCA 95%")


# ============================================================
#  STEP 6 — THE TRADEOFF VERDICT
# ============================================================

print("\n" + "=" * 64)
print("STEP 6 — WAS THE TRADEOFF WORTH IT?")
print("=" * 64)
print(f"""
  COMPRESSION SUMMARY
  ───────────────────
  Raw features        : 64
  PCA 95% components  : {pca_95.n_components_}     ({100*(1-pca_95.n_components_/64):.0f}% fewer features)
  PCA 2D components   : 2      (97% fewer features — only for visualization)

  ACCURACY TRADEOFF
  ─────────────────
  Raw 64 features    : {raw_acc*100:.2f}%
  PCA 95% features   : {pca95_acc*100:.2f}%   ({acc_drop:+.2f}% change)
  PCA 2D  features   : {pca2_acc*100:.2f}%

  SPEED TRADEOFF
  ──────────────
  Raw training time  : {raw_t:.1f} ms
  PCA 95% train time : {pca95_t:.1f} ms  ({speedup:.1f}× faster)

  VERDICT
  ───────
  {'✓ WORTH IT' if abs(acc_drop) < 2 else '~ BORDERLINE' if abs(acc_drop) < 5 else '✗ TOO COSTLY'}:
  PCA 95% kept {pca95_acc*100:.1f}% accuracy while using only {pca_95.n_components_} of 64 features.
  Accuracy {'barely changed' if abs(acc_drop) < 1 else f'dropped by {abs(acc_drop):.1f}%'}.

  PCA also acts as a NOISE FILTER:
    The dropped 5% of variance is mostly random pixel noise.
    Removing it can sometimes IMPROVE generalisation slightly.

  When PCA is essential in real ML pipelines:
    → Thousands of features (text, genomics, images)
    → Training is too slow without compression
    → You want to visualise high-dimensional data in 2D/3D
    → Storage or inference cost matters (edge devices)

  When NOT to use PCA:
    → Interpretability matters (components are uninterpretable)
    → Features are already few and uncorrelated
    → Tree models (Random Forest, XGBoost) — they handle high-dim fine
""")


# ============================================================
#  STEP 7 — CLASSIFICATION REPORT FOR BEST MODEL (PCA 95%)
# ============================================================

print("=" * 64)
print(f"STEP 7 — CLASSIFICATION REPORT  (PCA 95%, {pca_95.n_components_} features)")
print("=" * 64)
print(classification_report(y_test,
                             results[f"PCA 95% ({pca_95.n_components_} features)"]["pred"],
                             target_names=[str(i) for i in range(10)]))


# ============================================================
#  STEP 8 — PLOTS
# ============================================================

DIGIT_COLORS = [
    '#E24B4A','#378ADD','#63B77A','#F5A623','#9B59B6',
    '#1ABC9C','#E67E22','#34495E','#C0392B','#7F8C8D'
]

fig = plt.figure(figsize=(20, 18))
fig.suptitle("PCA — Compression + Visualisation\n"
             "Digits Dataset  (1797 images · 64 features · 10 classes)",
             fontsize=14, y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)


# ── Panel (0,0–1): 2D PCA scatter coloured by digit ──────────
ax00 = fig.add_subplot(gs[0, 0:2])
X_all_2d = pca_2.transform(scaler.transform(X))   # project ALL points to 2D

for digit in range(10):
    mask = y == digit
    ax00.scatter(X_all_2d[mask, 0], X_all_2d[mask, 1],
                 c=DIGIT_COLORS[digit], s=12, alpha=0.55,
                 edgecolors='none', label=str(digit))

ax00.set_xlabel(f"PC1  ({var_2[0]*100:.1f}% variance)")
ax00.set_ylabel(f"PC2  ({var_2[1]*100:.1f}% variance)")
ax00.set_title(f"2D PCA Projection — All 1797 Digits\n"
               f"Colour = digit class  |  Total variance kept: {cumvar_2*100:.1f}%\n"
               f"Clusters forming = PCA has found structure even with 2 components",
               fontsize=9)
ax00.legend(title="Digit", ncol=5, fontsize=7.5, loc='lower right',
            markerscale=2, framealpha=0.8)
ax00.grid(True, alpha=0.15)


# ── Panel (0,2): Sample digit images ─────────────────────────
ax02 = fig.add_subplot(gs[0, 2])
ax02.axis('off')
ax02.set_title("Sample Images from Dataset\n(each is 8×8 = 64 pixels)", fontsize=9)
# Show one sample of each digit in a 2×5 grid
inner_gs = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0, 2],
                                             hspace=0.1, wspace=0.1)
for d in range(10):
    idx  = np.where(y == d)[0][0]
    axd  = fig.add_subplot(inner_gs[d // 5, d % 5])
    axd.imshow(X[idx].reshape(8, 8), cmap='gray_r', interpolation='nearest')
    axd.set_title(str(d), fontsize=9, pad=1)
    axd.axis('off')


# ── Panel (1,0): Scree plot — per-component variance ─────────
ax10 = fig.add_subplot(gs[1, 0])
comp_x = np.arange(1, len(evr) + 1)
ax10.bar(comp_x[:30], evr[:30] * 100, color='steelblue',
         alpha=0.75, edgecolor='white', label='Individual variance')
ax10.set_xlabel("Principal Component")
ax10.set_ylabel("Variance Explained (%)")
ax10.set_title("Scree Plot — Variance per Component\n"
               "First few components carry most information (steep drop)", fontsize=9)
ax10.grid(True, alpha=0.2, axis='y')
ax10.set_xlim(0, 31)


# ── Panel (1,1): Cumulative variance curve ────────────────────
ax11 = fig.add_subplot(gs[1, 1])
ax11.plot(comp_x, cumsum_evr * 100, 'o-', color='seagreen',
          linewidth=2, markersize=3)
ax11.axhline(y=95, color='tomato', linestyle='--', linewidth=1.5,
             label='95% threshold')
ax11.axvline(x=n_95, color='tomato', linestyle='--', linewidth=1.5)
ax11.scatter([n_95], [cumsum_evr[n_95-1]*100], color='tomato', s=100, zorder=5)
ax11.annotate(f'{n_95} components\nfor 95% variance',
              xy=(n_95, cumsum_evr[n_95-1]*100),
              xytext=(n_95 + 6, 80),
              fontsize=8.5, color='tomato',
              arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2))
ax11.fill_between(comp_x[:n_95], cumsum_evr[:n_95]*100,
                  alpha=0.12, color='seagreen', label=f'Kept ({n_95} components)')
ax11.set_xlabel("Number of Principal Components")
ax11.set_ylabel("Cumulative Variance Explained (%)")
ax11.set_title(f"Cumulative Variance — How Many Components Needed?\n"
               f"95% threshold = {n_95} components  (down from 64)", fontsize=9)
ax11.legend(fontsize=8.5)
ax11.grid(True, alpha=0.2)
ax11.set_ylim(0, 103)


# ── Panel (1,2): Accuracy & speed tradeoff ───────────────────
ax12 = fig.add_subplot(gs[1, 2])
config_labels = ['Raw\n64 feat', f'PCA 95%\n{pca_95.n_components_} feat', 'PCA 2D\n2 feat']
accs  = [raw_acc*100, pca95_acc*100, pca2_acc*100]
times = [results["Raw (64 features)"]["train_ms"],
         results[f"PCA 95% ({pca_95.n_components_} features)"]["train_ms"],
         results["PCA 2D  (2 features)"]["train_ms"]]
bar_c = ['#E24B4A', '#63B77A', '#AAB7C4']

ax12b = ax12.twinx()
b1 = ax12.bar(np.arange(3) - 0.2, accs,  0.35, color=bar_c, alpha=0.82,
              edgecolor='white', label='Accuracy')
b2 = ax12b.bar(np.arange(3) + 0.2, times, 0.35, color=['#88AABB']*3,
               alpha=0.65, edgecolor='white', label='Train time (ms)')
for bar, val in zip(b1, accs):
    ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
              f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for bar, val in zip(b2, times):
    ax12b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.0f}ms', ha='center', va='bottom', fontsize=7.5, color='#336688')
ax12.set_xticks([0, 1, 2])
ax12.set_xticklabels(config_labels, fontsize=9)
ax12.set_ylabel("Test Accuracy (%)", color='#222')
ax12b.set_ylabel("Training Time (ms)", color='#336688')
ax12.set_ylim(50, 107)
ax12.set_title("Accuracy vs Training Speed\nGreen = best tradeoff", fontsize=9)
ax12.grid(True, alpha=0.2, axis='y')


# ── Panel (2,0): PCA reconstruction — original vs compressed ──
ax_row2 = [fig.add_subplot(gs[2, j]) for j in range(3)]

# Show original vs reconstructed images for a few digits
sample_digits = [0, 1, 2, 3, 4, 5, 6, 7]
n_show = 8
orig    = X_train_scaled[:n_show]
comp_95 = pca_95.transform(orig)
recon_95 = pca_95.inverse_transform(comp_95)

pca_2_r = PCA(n_components=2, random_state=42).fit(X_train_scaled)
comp_2  = pca_2_r.transform(orig)
recon_2 = pca_2_r.inverse_transform(comp_2)

# undo scaling for display
orig_px   = scaler.inverse_transform(orig)
recon95px = scaler.inverse_transform(recon_95)
recon2px  = scaler.inverse_transform(recon_2)

for col_idx, (ax_r, title_r, data_r) in enumerate(zip(
        ax_row2,
        ["Original  (64 features)",
         f"Reconstructed  (PCA 95%, {pca_95.n_components_} components)",
         "Reconstructed  (PCA 2D, 2 components — lossy)"],
        [orig_px, recon95px, recon2px])):
    inner = gridspec.GridSpecFromSubplotSpec(2, 4,
                subplot_spec=gs[2, col_idx], hspace=0.05, wspace=0.05)
    for idx in range(n_show):
        axi = fig.add_subplot(inner[idx // 4, idx % 4])
        img = np.clip(data_r[idx].reshape(8, 8), 0, 16)
        axi.imshow(img, cmap='gray_r', interpolation='nearest')
        axi.axis('off')
    ax_r.axis('off')
    ax_r.set_title(title_r, fontsize=8.5, pad=18)

plt.savefig("pca_compression_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → pca_compression_plot.png")