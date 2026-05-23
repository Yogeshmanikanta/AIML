import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
#  STEP 1 — LOAD THE WINE DATASET
# ============================================================
# 178 wines from 3 Italian wineries, 13 chemical features each.
# We've used this before for Random Forest — now KNN gets a turn.
#
# Why KNN needs scaling more than almost any other model:
#   KNN works by measuring DISTANCE between points.
#   "Which K training points are closest to this new point?"
#
#   Problem: 'alcohol' ranges from 11–15,
#             but 'proline' ranges from 278–1680.
#   Without scaling, proline distances are 100x bigger than alcohol.
#   The model effectively IGNORES alcohol entirely and decides
#   everything based on proline — even if alcohol is more informative.
#
#   After scaling: every feature has mean=0 and std=1.
#   All features contribute equally to distance calculations.

wine = load_wine()
X    = wine.data       # shape: (178, 13)
y    = wine.target     # 0, 1, or 2

print("=" * 64)
print("STEP 1 — WINE DATASET LOADED")
print("=" * 64)
print(f"  Samples   : {X.shape[0]}")
print(f"  Features  : {X.shape[1]}")
print(f"  Classes   : {list(wine.target_names)}")
print(f"\n  Feature ranges BEFORE scaling (why scaling is critical):")
print(f"  {'Feature':<30} {'Min':>8} {'Max':>8} {'Range':>8}")
print(f"  {'-'*57}")
for i, name in enumerate(wine.feature_names):
    mn, mx = X[:, i].min(), X[:, i].max()
    print(f"  {name:<30} {mn:>8.2f} {mx:>8.2f} {mx-mn:>8.2f}")
print(f"\n  Without scaling, high-range features like 'proline' (range ~1400)")
print(f"  completely dominate KNN distance calculations.")


# ============================================================
#  STEP 2 — TRAIN / TEST SPLIT
# ============================================================
# 80% training (142 wines), 20% test (36 wines).
# stratify=y keeps all 3 classes balanced in both splits.

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 64)
print("STEP 2 — TRAIN / TEST SPLIT")
print("=" * 64)
print(f"  Training : {X_train_raw.shape[0]} wines")
print(f"  Test     : {X_test_raw.shape[0]} wines")


# ============================================================
#  STEP 3 — SCALE THE FEATURES  (WITH scaling — the right way)
# ============================================================
# StandardScaler transforms each feature:
#   x_scaled = (x - mean) / std
#
# CRITICAL RULE: fit the scaler ONLY on training data.
# Then transform both train and test with those training statistics.
#
# If you fit on the full dataset first (data leakage!), the model
# "peeks" at the test set's statistics during training — cheating!
#
# After scaling: every feature has mean≈0 and std≈1.
# Now alcohol and proline contribute equally to distance.

scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train_raw)   # fit + transform on train
X_test   = scaler.transform(X_test_raw)        # transform only on test

print("\n" + "=" * 64)
print("STEP 3 — FEATURE SCALING  (StandardScaler)")
print("=" * 64)
print(f"  Scaler fitted on TRAINING data only  (no data leakage)")
print(f"\n  Feature ranges AFTER scaling:")
print(f"  {'Feature':<30} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*48}")
for i, name in enumerate(wine.feature_names):
    print(f"  {name:<30} {X_train[:,i].mean():>8.4f} {X_train[:,i].std():>8.4f}")
print(f"\n  All features now have mean≈0 and std≈1.")
print(f"  KNN will now treat all 13 features equally in distance.")


# ============================================================
#  STEP 4 — TRAIN KNN FOR MULTIPLE K VALUES
# ============================================================
# K is the "number of neighbours" to vote on each new point.
#
#   K = 1  : Only the single nearest neighbour votes.
#             The model memorises training data exactly.
#             Train accuracy = 100%, but wildly overfits on test.
#
#   K = 3  : Top 3 nearest neighbours vote (majority wins).
#             Less noisy than K=1, still fairly flexible.
#
#   K = 21 : 21 neighbours vote — smoother, more robust boundary.
#             Outliers and noise get outvoted by the majority.
#
#   K = 51 : Very many neighbours — boundary is very smooth.
#             May start underfitting (too many votes blur the boundary).
#
#   K = N (all samples): Every point votes — always predicts majority class.
#
# Sweet spot: usually where train and test accuracy curves cross
# and the test accuracy is at its maximum.

K_values = [1, 3, 5, 7, 11, 21, 51]
train_accs_scaled = []
test_accs_scaled  = []

print("\n" + "=" * 64)
print("STEP 4 — KNN FOR K = 1 to 51  (Scaled Features)")
print("=" * 64)
print(f"\n  {'K':>6}  {'Train Acc':>12}  {'Test Acc':>12}  {'Overfit Gap':>13}  Note")
print(f"  {'-'*65}")

for k in K_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    tr_acc = accuracy_score(y_train, knn.predict(X_train))
    te_acc = accuracy_score(y_test,  knn.predict(X_test))
    gap    = tr_acc - te_acc

    train_accs_scaled.append(tr_acc * 100)
    test_accs_scaled.append(te_acc * 100)

    note = ""
    if k == 1:   note = "← K=1 always 100% train (memorises everything)"
    if k == 51:  note = "← Large K, boundary smooths out"

    print(f"  {k:>6}  {tr_acc*100:>11.1f}%  {te_acc*100:>11.1f}%  {gap*100:>12.1f}%  {note}")

best_k_idx = int(np.argmax(test_accs_scaled))
best_k     = K_values[best_k_idx]
best_acc   = test_accs_scaled[best_k_idx]

print(f"\n  Best K by test accuracy : K = {best_k}  ({best_acc:.1f}%)")


# ============================================================
#  STEP 5 — CLASSIFICATION REPORT FOR BEST K
# ============================================================
# Full precision, recall, F1 breakdown for the best K value.
# This tells us not just overall accuracy but per-class performance.

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_best = best_knn.predict(X_test)

print("\n" + "=" * 64)
print(f"STEP 5 — CLASSIFICATION REPORT  (Best K = {best_k})")
print("=" * 64)
print(classification_report(y_test, y_pred_best,
                             target_names=wine.target_names))


# ============================================================
#  STEP 6 — THE SCALING EXPERIMENT  (Skip scaling — see damage)
# ============================================================
# Now we deliberately skip StandardScaler and run the same K values
# on raw, unscaled features.
#
# Expected result: accuracy drops significantly, especially for
# features with large ranges like 'proline' (278–1680) and
# 'magnesium' (70–162) that will dominate the distance metric.
#
# This experiment makes the scaling lesson unforgettable.
# Numbers don't lie — the accuracy drop IS the lesson.

print("\n" + "=" * 64)
print("STEP 6 — SCALING EXPERIMENT  (Unscaled vs Scaled Comparison)")
print("=" * 64)
print(f"\n  Training KNN on RAW (unscaled) features...")
print(f"\n  {'K':>6}  {'Scaled Test':>13}  {'Unscaled Test':>15}  {'Accuracy Drop':>15}")
print(f"  {'-'*55}")

train_accs_unscaled = []
test_accs_unscaled  = []

for i, k in enumerate(K_values):
    knn_raw = KNeighborsClassifier(n_neighbors=k)
    knn_raw.fit(X_train_raw, y_train)

    tr_acc_raw = accuracy_score(y_train, knn_raw.predict(X_train_raw))
    te_acc_raw = accuracy_score(y_test,  knn_raw.predict(X_test_raw))

    train_accs_unscaled.append(tr_acc_raw * 100)
    test_accs_unscaled.append(te_acc_raw * 100)

    drop   = test_accs_scaled[i] - te_acc_raw * 100
    marker = "  ← BIG DROP!" if drop > 10 else ("  ← notable" if drop > 3 else "")
    print(f"  {k:>6}  {test_accs_scaled[i]:>12.1f}%  {te_acc_raw*100:>14.1f}%  {drop:>14.1f}%{marker}")

avg_drop = np.mean(test_accs_scaled) - np.mean(test_accs_unscaled)
print(f"\n  Average accuracy drop from skipping scaling: {avg_drop:.1f}%")
print(f"\n  WHY does this happen?")
print(f"  'proline' has range ~1400 — its distances tower over all other features.")
print(f"  KNN effectively becomes '1-feature nearest neighbour on proline alone'.")
print(f"  The other 12 features contribute almost NOTHING to the distance.")


# ============================================================
#  STEP 7 — PLOTS
# ============================================================

fig = plt.figure(figsize=(18, 14))
fig.suptitle("KNN — Build Intuition  (Wine Dataset, 13 features, 3 classes)",
             fontsize=14, y=0.99)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)


# ── Panel 1: Train vs Test accuracy (scaled) ─────────────────
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(K_values, train_accs_scaled, 's--', color='steelblue',
         linewidth=2, markersize=7, label='Train Accuracy (scaled)')
ax1.plot(K_values, test_accs_scaled, 'o-', color='seagreen',
         linewidth=2.5, markersize=8, label='Test Accuracy (scaled)')

# Shade overfitting zone (K too small) and underfitting zone (K too large)
ax1.axvspan(K_values[0]-1,  best_k,          alpha=0.06, color='tomato',
            label='← Overfit zone (K too small)')
ax1.axvspan(best_k,         K_values[-1]+2,  alpha=0.06, color='orange',
            label='Underfit zone (K too large) →')

# Mark best K
ax1.axvline(x=best_k, color='seagreen', linestyle=':', linewidth=1.5, alpha=0.8)
ax1.scatter([best_k], [best_acc], color='seagreen', s=150, zorder=5)
ax1.annotate(f'Best K={best_k}\n{best_acc:.1f}%',
             xy=(best_k, best_acc),
             xytext=(best_k + 3, best_acc - 4),
             fontsize=9, color='seagreen',
             arrowprops=dict(arrowstyle='->', color='seagreen', lw=1.2))

ax1.set_xlabel("K  (number of neighbours)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Train vs Test Accuracy — Scaled Features\n"
              "Sweet spot: where test accuracy peaks")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.25)
ax1.set_ylim(60, 103)
ax1.set_xticks(K_values)


# ── Panel 2: Scaled vs Unscaled test accuracy ─────────────────
ax2 = fig.add_subplot(gs[0, 1])

ax2.plot(K_values, test_accs_scaled,   'o-',  color='seagreen',
         linewidth=2.5, markersize=8,  label='Scaled   (StandardScaler ✓)')
ax2.plot(K_values, test_accs_unscaled, 's--', color='tomato',
         linewidth=2,   markersize=7,  label='Unscaled (raw features ✗)')

# Fill the accuracy gap between scaled and unscaled
ax2.fill_between(K_values, test_accs_scaled, test_accs_unscaled,
                 alpha=0.15, color='red', label='Accuracy lost by skipping scaling')

ax2.set_xlabel("K  (number of neighbours)")
ax2.set_ylabel("Test Accuracy (%)")
ax2.set_title("Scaled vs Unscaled — The Scaling Experiment\n"
              "Red gap = accuracy destroyed by skipping StandardScaler")
ax2.legend(fontsize=8.5)
ax2.grid(True, alpha=0.25)
ax2.set_ylim(30, 105)
ax2.set_xticks(K_values)


# ── Panel 3: Overfit gap bar chart ────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

gaps_scaled   = [tr - te for tr, te in zip(train_accs_scaled,   test_accs_scaled)]
gaps_unscaled = [tr - te for tr, te in zip(train_accs_unscaled, test_accs_unscaled)]

x     = np.arange(len(K_values))
width = 0.35
b1 = ax3.bar(x - width/2, gaps_scaled,   width, label='Scaled',
             color='steelblue', alpha=0.85, edgecolor='white')
b2 = ax3.bar(x + width/2, gaps_unscaled, width, label='Unscaled',
             color='tomato',    alpha=0.85, edgecolor='white')

for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0.5:
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 f'{h:.1f}', ha='center', va='bottom', fontsize=8)

ax3.set_xticks(x); ax3.set_xticklabels([f'K={k}' for k in K_values], fontsize=8)
ax3.set_ylabel("Train − Test Accuracy (%)")
ax3.set_title("Overfit Gap per K  (Train − Test)\n"
              "K=1 has biggest gap — memorises training data")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2, axis='y')
ax3.axhline(y=0, color='gray', linewidth=0.8)


# ── Panel 4: Feature range comparison (why scaling matters) ───
ax4 = fig.add_subplot(gs[1, 1])

feat_ranges   = X[:, :].max(axis=0) - X[:, :].min(axis=0)
feat_names_short = [n[:12] for n in wine.feature_names]
sort_fi       = np.argsort(feat_ranges)[::-1]

colors_range = ['#E24B4A' if feat_ranges[i] > 200 else
                '#F5A623' if feat_ranges[i] > 20  else
                '#378ADD'
                for i in sort_fi]

bars4 = ax4.barh([feat_names_short[i] for i in sort_fi[::-1]],
                 [feat_ranges[i] for i in sort_fi[::-1]],
                 color=[colors_range[j] for j in range(len(sort_fi)-1, -1, -1)],
                 alpha=0.85, edgecolor='white', height=0.6)

ax4.set_xlabel("Feature Range  (Max − Min, raw units)")
ax4.set_title("Why Scaling is Mandatory for KNN\n"
              "Red = dominates distance  ·  Blue = gets ignored without scaling")
ax4.grid(True, alpha=0.2, axis='x')

# Annotate the biggest offender
max_idx = sort_fi[0]
ax4.annotate(f"'{wine.feature_names[max_idx]}'\nrange ≈ {feat_ranges[max_idx]:.0f}\n← dominates KNN!",
             xy=(feat_ranges[max_idx], len(sort_fi) - 0.5),
             xytext=(feat_ranges[max_idx] * 0.5, len(sort_fi) - 2.5),
             fontsize=8, color='#C0392B',
             arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2))

plt.savefig("knn_intuition_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → knn_intuition_plot.png")