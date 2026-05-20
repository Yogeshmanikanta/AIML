import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
#  STEP 1 — LOAD THE WINE DATASET  (more challenging than Iris)
# ============================================================
# The Wine dataset has 178 wine samples from 3 Italian wineries.
# Each wine has 13 chemical measurements:
#   alcohol, malic acid, ash, magnesium, flavanoids, color intensity, etc.
#
# Why Wine instead of Iris?
#   - Iris is too easy — both models score ~97% and the gap is tiny.
#   - Wine has 13 features (vs 4), more noise, and trickier class boundaries.
#   - The overfitting problem and Random Forest's advantage show up MUCH
#     more clearly here — which is exactly the point of this exercise.
#
# Goal: predict which of the 3 wineries a wine came from.

wine = load_wine()

X = wine.data     # shape: (178, 13)  — 178 wines, 13 chemical features
y = wine.target   # 0, 1, or 2        — which winery it came from

print("=" * 62)
print("STEP 1 — WINE DATASET OVERVIEW")
print("=" * 62)
print(f"Total wines       : {X.shape[0]}")
print(f"Features          : {X.shape[1]}  chemical measurements")
print(f"Feature names     : {', '.join(wine.feature_names[:5])}, ...")
print(f"Classes           : {list(wine.target_names)}")
print(f"Samples per class : {[(y==i).sum() for i in range(3)]}")
print(f"\nWhy harder than Iris: 13 features, unequal class sizes,")
print(f"overlapping chemical profiles → more room for overfitting.")


# ============================================================
#  STEP 2 — TRAIN / TEST SPLIT
# ============================================================
# 80% = 142 wines for training, 20% = 36 wines for testing.
# stratify=y keeps the same class ratio in both splits.
# random_state=42 makes results reproducible every run.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 62)
print("STEP 2 — TRAIN / TEST SPLIT")
print("=" * 62)
print(f"Training wines    : {X_train.shape[0]}  (80%)")
print(f"Test wines        : {X_test.shape[0]}   (20%)")


# ============================================================
#  STEP 3 — SINGLE DECISION TREE  (no depth limit)
# ============================================================
# A single Decision Tree with no max_depth will grow until every
# training sample sits in its own leaf node — perfect train accuracy.
#
# But with 13 features and only 142 training samples, the tree
# memorizes very specific combinations that don't exist in real life.
# Result: great training score, weaker test score — classic overfitting.
#
# This is the VARIANCE problem: small changes in training data
# would produce a completely different tree.

dt = DecisionTreeClassifier(random_state=42)   # no depth limit
dt.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_test_acc  = accuracy_score(y_test,  dt.predict(X_test))

print("\n" + "=" * 62)
print("STEP 3 — SINGLE DECISION TREE  (no depth limit)")
print("=" * 62)
print(f"Tree depth        : {dt.get_depth()}  levels")
print(f"Leaf nodes        : {dt.get_n_leaves()}")
print(f"Train Accuracy    : {dt_train_acc * 100:.2f}%  ← memorized everything")
print(f"Test  Accuracy    : {dt_test_acc  * 100:.2f}%  ← drops on unseen data")
print(f"Overfit Gap       : {(dt_train_acc - dt_test_acc)*100:.2f}%  ← this is the problem")


# ============================================================
#  STEP 4 — RANDOM FOREST  (100 trees + OOB score)
# ============================================================
# A Random Forest fixes the single tree's variance problem by:
#
#   1. BAGGING: each of the 100 trees trains on a random bootstrap
#      sample (random 63% of training data, with replacement).
#      Different trees see different data → they make different mistakes.
#
#   2. FEATURE RANDOMNESS: at each split, only sqrt(13)≈3-4 features
#      are considered (not all 13). This forces trees to be different
#      from each other and prevents one dominant feature from controlling
#      every tree.
#
#   3. VOTING: for prediction, all 100 trees vote. The majority class wins.
#      Individual tree mistakes cancel out — wisdom of the crowd.
#
# oob_score=True → uses Out-Of-Bag samples (the ~37% each tree didn't
# train on) as a FREE internal validation set.
# OOB score ≈ cross-validation accuracy without extra computation.

rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,       # free validation using unused bootstrap samples
    random_state=42,
    n_jobs=-1             # use all CPU cores
)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc  = accuracy_score(y_test,  rf.predict(X_test))
rf_oob_acc   = rf.oob_score_    # accuracy on out-of-bag samples

print("\n" + "=" * 62)
print("STEP 4 — RANDOM FOREST  (100 trees)")
print("=" * 62)
print(f"Number of trees   : 100")
print(f"Features per split: sqrt(13) ≈ {int(np.sqrt(13))}  (random subset each split)")
print(f"Train Accuracy    : {rf_train_acc * 100:.2f}%")
print(f"Test  Accuracy    : {rf_test_acc  * 100:.2f}%  ← significantly better!")
print(f"OOB   Score       : {rf_oob_acc   * 100:.2f}%  ← free estimate using unused samples")
print(f"Overfit Gap       : {(rf_train_acc - rf_test_acc)*100:.2f}%  ← much smaller gap")
print(f"\nOOB vs Test diff  : {abs(rf_oob_acc - rf_test_acc)*100:.2f}%  ← OOB is a reliable proxy!")


# ============================================================
#  STEP 5 — HEAD-TO-HEAD COMPARISON
# ============================================================

print("\n" + "=" * 62)
print("STEP 5 — HEAD-TO-HEAD COMPARISON")
print("=" * 62)
print(f"\n{'Metric':<25} {'Decision Tree':>16} {'Random Forest':>16}")
print("-" * 59)
print(f"{'Tree depth':<25} {dt.get_depth():>16} {'avg ~5-8':>16}")
print(f"{'Total trees':<25} {'1':>16} {'100':>16}")
print(f"{'Train Accuracy':<25} {dt_train_acc*100:>15.2f}% {rf_train_acc*100:>15.2f}%")
print(f"{'Test  Accuracy':<25} {dt_test_acc*100:>15.2f}% {rf_test_acc*100:>15.2f}%")
print(f"{'Overfit Gap':<25} {(dt_train_acc-dt_test_acc)*100:>15.2f}% {(rf_train_acc-rf_test_acc)*100:>15.2f}%")
print(f"{'OOB Score':<25} {'N/A':>16} {rf_oob_acc*100:>15.2f}%")
print("-" * 59)
print(f"{'Overfits more?':<25} {'YES ✗':>16} {'No  ✓':>16}")
print(f"{'Generalizes better?':<25} {'No  ✗':>16} {'YES ✓':>16}")


# ============================================================
#  STEP 6 — FEATURE IMPORTANCES: Tree vs Forest
# ============================================================
# Single tree importance = how much each feature reduced Gini
#   across splits IN THAT ONE TREE.
#   Problem: unstable! Train the tree again with a different random_state
#   and the importances change dramatically.
#
# Random Forest importance = AVERAGE Gini reduction across ALL 100 trees.
#   Much more stable and trustworthy.
#   Features that consistently help across many trees get high scores.
#   Noisy features that only helped by chance average out to near zero.

dt_imp = dt.feature_importances_
rf_imp = rf.feature_importances_
feat_names = wine.feature_names

# Sort by RF importance (the more trustworthy one)
sort_idx = np.argsort(rf_imp)[::-1]

print("\n" + "=" * 62)
print("STEP 6 — FEATURE IMPORTANCES")
print("=" * 62)
print(f"{'Feature':<30} {'Single Tree':>12} {'Random Forest':>14}")
print("-" * 58)
for i in sort_idx:
    dt_bar = "█" * int(dt_imp[i] * 30)
    rf_bar = "█" * int(rf_imp[i] * 30)
    print(f"  {feat_names[i]:<28} {dt_imp[i]:>10.4f}   {rf_imp[i]:>10.4f}")
print("-" * 58)
print(f"\n  RF top feature  → '{feat_names[sort_idx[0]]}'  ({rf_imp[sort_idx[0]]:.4f})")
print(f"  DT top feature  → '{feat_names[np.argmax(dt_imp)]}'  ({dt_imp.max():.4f})")
stable = "AGREE" if sort_idx[0] == np.argmax(dt_imp) else "DISAGREE — RF is more trustworthy"
print(f"  Do they agree?    {stable}")


# ============================================================
#  STEP 7 — ACCURACY vs NUMBER OF TREES
# ============================================================
# Key question: at what point does adding more trees stop helping?
#
# Expected pattern:
#   1 tree   → high variance, similar to single DecisionTree
#   10 trees → big accuracy jump, variance drops sharply
#   50 trees → most of the gain already achieved
#   100 trees → nearly converged, small improvements
#   200 trees → diminishing returns — barely better than 100
#
# This is the LAW OF DIMINISHING RETURNS for Random Forests.
# After ~50-100 trees, each new tree adds less and less value.
# You're just spending more compute time for marginal gains.

n_estimators_list = [1, 5, 10, 20, 50, 100, 150, 200]
train_accs, test_accs, oob_accs = [], [], []

print("\n" + "=" * 62)
print("STEP 7 — ACCURACY vs NUMBER OF TREES")
print("=" * 62)
print(f"\n{'n_estimators':>14} {'Train Acc':>12} {'Test Acc':>12} {'OOB Score':>12}")
print("-" * 54)

for n in n_estimators_list:
    rf_n = RandomForestClassifier(
        n_estimators=n,
        oob_score=(n > 1),    # OOB needs at least 2 trees
        random_state=42,
        n_jobs=-1
    )
    rf_n.fit(X_train, y_train)

    tr_acc = accuracy_score(y_train, rf_n.predict(X_train))
    te_acc = accuracy_score(y_test,  rf_n.predict(X_test))
    ob_acc = rf_n.oob_score_ if n > 1 else float('nan')

    train_accs.append(tr_acc * 100)
    test_accs.append(te_acc  * 100)
    oob_accs.append(ob_acc   * 100)

    oob_str = f"{ob_acc*100:>11.2f}%" if n > 1 else "         N/A"
    print(f"{n:>14}  {tr_acc*100:>11.2f}%  {te_acc*100:>11.2f}%  {oob_str}")

print("-" * 54)
peak_idx  = np.argmax(test_accs)
peak_n    = n_estimators_list[peak_idx]
print(f"\n  Peak test accuracy at n_estimators = {peak_n}  ({test_accs[peak_idx]:.2f}%)")
print(f"  After that: adding trees gives diminishing returns.")


# ============================================================
#  STEP 8 — PLOTS  (4-panel figure)
# ============================================================

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Random Forest vs Decision Tree  —  Wine Dataset  (13 features, 3 classes)",
             fontsize=14, y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)


# ── Panel 1: Accuracy vs n_estimators ────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(n_estimators_list, train_accs, 's--',
         color='steelblue', linewidth=2, markersize=6, label='Train Accuracy')
ax1.plot(n_estimators_list, test_accs,  'o-',
         color='seagreen',  linewidth=2.5, markersize=7, label='Test Accuracy')
ax1.plot(n_estimators_list, oob_accs,   '^:',
         color='darkorange', linewidth=1.8, markersize=6, label='OOB Score', alpha=0.85)

# Mark single DT baseline
ax1.axhline(y=dt_test_acc*100, color='tomato', linestyle='--',
            linewidth=1.5, label=f'Single DT baseline ({dt_test_acc*100:.1f}%)')

# Mark peak
ax1.axvline(x=peak_n, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax1.scatter([peak_n], [test_accs[peak_idx]], color='seagreen', s=120, zorder=5)
ax1.annotate(f'Peak @ n={peak_n}\n{test_accs[peak_idx]:.1f}%',
             xy=(peak_n, test_accs[peak_idx]),
             xytext=(peak_n + 15, test_accs[peak_idx] - 2.5),
             fontsize=8, color='seagreen',
             arrowprops=dict(arrowstyle='->', color='seagreen', lw=1.2))

ax1.set_xlabel("Number of Trees  (n_estimators)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy vs Number of Trees\n(diminishing returns after the peak)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.25)
ax1.set_ylim(min(test_accs) - 5, 103)


# ── Panel 2: Feature importances comparison ───────────────────
ax2 = fig.add_subplot(gs[0, 1])

x_feat = np.arange(len(feat_names))
w      = 0.38
bars1 = ax2.bar(x_feat - w/2, dt_imp[sort_idx], w,
                label='Single Tree',   color='tomato',    alpha=0.8)
bars2 = ax2.bar(x_feat + w/2, rf_imp[sort_idx], w,
                label='Random Forest', color='steelblue', alpha=0.8)

ax2.set_xticks(x_feat)
ax2.set_xticklabels([feat_names[i][:12] for i in sort_idx],
                     rotation=45, ha='right', fontsize=7.5)
ax2.set_ylabel("Importance Score")
ax2.set_title("Feature Importances: Single Tree vs Random Forest\n(RF = averaged over 100 trees = more stable)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2, axis='y')


# ── Panel 3: Train vs Test bar chart (DT vs RF 100) ───────────
ax3 = fig.add_subplot(gs[1, 0])

categories = ['Single\nDecision Tree', 'Random Forest\n(100 trees)']
tr_vals    = [dt_train_acc * 100, rf_train_acc * 100]
te_vals    = [dt_test_acc  * 100, rf_test_acc  * 100]
x3         = np.arange(2)
wb         = 0.32

b1 = ax3.bar(x3 - wb/2, tr_vals, wb,
             label='Train Accuracy', color='steelblue', alpha=0.85)
b2 = ax3.bar(x3 + wb/2, te_vals, wb,
             label='Test Accuracy',  color='seagreen',  alpha=0.85)

for bar in list(b1) + list(b2):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9.5)

# Draw overfit gap arrows
for xi, tr, te in zip(x3, tr_vals, te_vals):
    ax3.annotate('', xy=(xi, te), xytext=(xi, tr),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax3.text(xi + 0.18, (tr + te)/2, f'gap\n{tr-te:.1f}%',
             fontsize=8, color='gray', va='center')

ax3.set_xticks(x3)
ax3.set_xticklabels(categories, fontsize=10)
ax3.set_ylim(80, 108)
ax3.set_ylabel("Accuracy (%)")
ax3.set_title("Overfitting Gap: Single Tree vs Random Forest\n(smaller gap = less overfitting)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2, axis='y')


# ── Panel 4: OOB score vs Test score ─────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

valid_idx = [i for i, n in enumerate(n_estimators_list) if n > 1]
oob_valid  = [oob_accs[i]  for i in valid_idx]
test_valid = [test_accs[i] for i in valid_idx]
n_valid    = [n_estimators_list[i] for i in valid_idx]

ax4.plot(n_valid, test_valid, 'o-',  color='seagreen',  linewidth=2.5,
         markersize=7, label='Test Accuracy  (real holdout)')
ax4.plot(n_valid, oob_valid,  '^--', color='darkorange', linewidth=2,
         markersize=7, label='OOB Score  (free estimate)', alpha=0.9)

# Fill the gap between them
ax4.fill_between(n_valid, oob_valid, test_valid,
                 alpha=0.12, color='purple',
                 label='Gap between OOB and Test')

ax4.set_xlabel("Number of Trees")
ax4.set_ylabel("Accuracy (%)")
ax4.set_title("OOB Score vs Test Accuracy\n(OOB tracks test well → no extra validation needed)")
ax4.legend(fontsize=8.5)
ax4.grid(True, alpha=0.25)
ax4.set_ylim(min(oob_valid) - 3, 103)


plt.savefig("random_forest_vs_tree_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → random_forest_vs_tree_plot.png")