import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
#  STEP 1 — LOAD THE IRIS DATASET
# ============================================================
# The Iris dataset is one of the most famous datasets in ML.
# Collected in 1936 by statistician Ronald Fisher.
#
# It contains measurements of 150 iris flowers from 3 species:
#   - Setosa     (class 0) — 50 flowers
#   - Versicolor (class 1) — 50 flowers
#   - Virginica  (class 2) — 50 flowers
#
# Each flower has 4 measurements (features):
#   1. Sepal Length (cm)
#   2. Sepal Width  (cm)
#   3. Petal Length (cm)  ← usually most important
#   4. Petal Width  (cm)  ← usually most important
#
# Goal: given the 4 measurements, predict which species it is.
# This is a MULTI-CLASS problem (3 classes), unlike breast cancer (2 classes).

iris = load_iris()

X = iris.data          # shape: (150, 4)  — 150 flowers, 4 features
y = iris.target        # shape: (150,)    — 0, 1, or 2

print("=" * 60)
print("STEP 1 — IRIS DATASET OVERVIEW")
print("=" * 60)
print(f"Total flowers     : {X.shape[0]}")
print(f"Features          : {X.shape[1]}  → {list(iris.feature_names)}")
print(f"Classes           : {list(iris.target_names)}")
print(f"Samples per class : {[(y==i).sum() for i in range(3)]}  (perfectly balanced)")


# ============================================================
#  STEP 2 — TRAIN / TEST SPLIT
# ============================================================
# 80% = 120 flowers for training, 20% = 30 flowers for testing.
# stratify=y ensures all 3 species are represented equally in both sets.
# Without stratify, one species could accidentally be missing from the test set.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("STEP 2 — TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Training flowers : {X_train.shape[0]}  (80%)")
print(f"Test flowers     : {X_test.shape[0]}   (20%)")


# ============================================================
#  STEP 3 — DECISION TREE WITH NO DEPTH LIMIT (Overfitting)
# ============================================================
# A Decision Tree works by asking a series of yes/no questions:
#   "Is petal length < 2.45 cm?"  → Yes → Setosa
#                                 → No  → ask next question...
#
# With NO depth limit, the tree keeps splitting until every single
# training sample is correctly classified — even if it has to create
# a split just for 1 or 2 flowers.
#
# This causes OVERFITTING:
#   - Training accuracy = 100% (memorized every flower)
#   - Test accuracy drops because it memorized noise, not patterns
#
# Think of it like a student who memorizes exam answers word-for-word
# instead of understanding the concept — fails on new questions.

tree_full = DecisionTreeClassifier(random_state=42)   # no max_depth = unlimited
tree_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, tree_full.predict(X_train))
test_acc_full  = accuracy_score(y_test,  tree_full.predict(X_test))

print("\n" + "=" * 60)
print("STEP 3 — UNLIMITED DEPTH TREE  (Overfitting example)")
print("=" * 60)
print(f"Tree depth reached : {tree_full.get_depth()}  levels deep")
print(f"Total leaf nodes   : {tree_full.get_n_leaves()}  (one per unique path)")
print(f"Train Accuracy     : {train_acc_full * 100:.2f}%  ← memorized training data!")
print(f"Test  Accuracy     : {test_acc_full  * 100:.2f}%  ← struggles on new data")
print(f"Gap (overfit)      : {(train_acc_full - test_acc_full)*100:.2f}%  ← this gap = overfitting")


# ============================================================
#  STEP 4 — DECISION TREE WITH MAX DEPTH = 3 (Generalization)
# ============================================================
# max_depth=3 means the tree can ask at most 3 questions in a row.
# This forces the tree to find the MOST IMPORTANT splits only —
# it can't afford to waste splits on edge cases.
#
# This is called REGULARIZATION — limiting model complexity
# so it learns general patterns instead of memorizing specifics.
#
# With depth=3 on Iris, we often get BETTER test accuracy than
# the unlimited tree, because the simple structure generalizes well.
#
# Think of it like a student who understands the 3 core concepts
# deeply — they perform better on new questions than the memorizer.

tree_d3 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_d3.fit(X_train, y_train)

train_acc_d3 = accuracy_score(y_train, tree_d3.predict(X_train))
test_acc_d3  = accuracy_score(y_test,  tree_d3.predict(X_test))

print("\n" + "=" * 60)
print("STEP 4 — MAX DEPTH = 3 TREE  (Good generalization)")
print("=" * 60)
print(f"Tree depth        : {tree_d3.get_depth()}  levels deep  (capped at 3)")
print(f"Total leaf nodes  : {tree_d3.get_n_leaves()}  (much simpler than {tree_full.get_n_leaves()})")
print(f"Train Accuracy    : {train_acc_d3 * 100:.2f}%")
print(f"Test  Accuracy    : {test_acc_d3  * 100:.2f}%  ← same or better than unlimited!")
print(f"Gap (overfit)     : {(train_acc_d3 - test_acc_d3)*100:.2f}%  ← much smaller gap = healthier model")


# ============================================================
#  STEP 5 — COMPARE BOTH TREES
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 — SIDE-BY-SIDE COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'No Limit':>12} {'max_depth=3':>14}")
print("-" * 53)
print(f"{'Tree Depth':<25} {tree_full.get_depth():>12} {tree_d3.get_depth():>14}")
print(f"{'Leaf Nodes':<25} {tree_full.get_n_leaves():>12} {tree_d3.get_n_leaves():>14}")
print(f"{'Train Accuracy':<25} {train_acc_full*100:>11.2f}% {train_acc_d3*100:>13.2f}%")
print(f"{'Test Accuracy':<25} {test_acc_full*100:>11.2f}% {test_acc_d3*100:>13.2f}%")
print(f"{'Overfit Gap':<25} {(train_acc_full-test_acc_full)*100:>11.2f}% {(train_acc_d3-test_acc_d3)*100:>13.2f}%")
print("-" * 53)
winner = "max_depth=3" if test_acc_d3 >= test_acc_full else "No Limit"
print(f"{'Better test model':<25} {'':>12} {winner:>14}  ✓")


# ============================================================
#  STEP 6 — FEATURE IMPORTANCES
# ============================================================
# After training, the tree knows which features it used most.
# feature_importances_ gives a score (0 to 1) for each feature.
# Higher score = that feature was used to make the most impactful splits.
#
# Importance is calculated by how much each feature reduced "impurity"
# (Gini index) across all the splits it was used in.
# A feature that cleanly separates many samples gets a high score.
#
# For Iris, petal measurements almost always dominate sepal measurements
# because petals vary much more between species than sepals do.

importances   = tree_d3.feature_importances_
feature_names = iris.feature_names
sorted_idx    = np.argsort(importances)[::-1]   # sort high to low

print("\n" + "=" * 60)
print("STEP 6 — FEATURE IMPORTANCES  (max_depth=3 tree)")
print("=" * 60)
print("How much did each feature contribute to reducing impurity?\n")
for rank, idx in enumerate(sorted_idx):
    bar = "█" * int(importances[idx] * 40)
    print(f"  #{rank+1}  {feature_names[idx]:<25}  {importances[idx]:.4f}  {bar}")

top_feature = feature_names[sorted_idx[0]]
print(f"\n  Most important feature → '{top_feature}'")
print(f"  The tree relied on this for its very first (root) split.")


# ============================================================
#  STEP 7 — TRACE A PATH FROM ROOT TO LEAF
# ============================================================
# Let's manually trace what happens to one test flower as it
# travels through the depth-3 tree, question by question.

sample_idx    = 0
sample        = X_test[sample_idx]
sample_label  = y_test[sample_idx]
predicted     = tree_d3.predict([sample])[0]

print("\n" + "=" * 60)
print("STEP 7 — TRACING A PATH: ROOT → LEAF")
print("=" * 60)
print(f"\nFlower being classified:")
for fname, fval in zip(iris.feature_names, sample):
    print(f"  {fname:<25}: {fval:.2f} cm")
print(f"\nActual species    : {iris.target_names[sample_label]}")
print(f"Predicted species : {iris.target_names[predicted]}")

# Extract the tree structure for manual trace
feat   = tree_d3.tree_.feature
thresh = tree_d3.tree_.threshold

print(f"""
Decision path:
  ROOT  →  Is {iris.feature_names[feat[0]]} <= {thresh[0]:.2f}?
            Flower value: {sample[feat[0]]:.2f}
            Answer: {'YES → left branch' if sample[feat[0]] <= thresh[0] else 'NO  → right branch'}
""")
print("  (Follow the tree visualization below to trace the rest!)")


# ============================================================
#  STEP 8 — PLOTS
# ============================================================
# 4 panels:
#   [Top-Left]  The depth-3 tree visualization — readable and annotated
#   [Top-Right] Feature importance bar chart
#   [Bot-Left]  Train vs Test accuracy comparison bar chart
#   [Bot-Right] The full unlimited tree — to show how messy it gets

fig = plt.figure(figsize=(22, 18))
fig.suptitle("Decision Tree — Iris Dataset", fontsize=15, y=0.98)

# ── Panel 1: Depth-3 tree (readable) ─────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
plot_tree(
    tree_d3,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,          # color nodes by majority class
    rounded=True,         # rounded corners
    fontsize=9,
    ax=ax1
)
ax1.set_title(
    "Decision Tree  (max_depth=3)  — THE READABLE ONE\n"
    "Orange=Setosa  Blue=Versicolor  Green=Virginica\n"
    "Each node shows: split rule | Gini | samples | class distribution",
    fontsize=9
)

# ── Panel 2: Feature importances ─────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
colors_imp = ['#E24B4A' if i == sorted_idx[0] else '#378ADD' for i in range(4)]
bars = ax2.barh(
    [feature_names[i] for i in range(3, -1, -1)],
    [importances[i] for i in range(3, -1, -1)],
    color=[colors_imp[i] for i in range(3, -1, -1)],
    alpha=0.85, edgecolor='white'
)
ax2.set_xlabel("Importance Score  (Gini reduction contribution)")
ax2.set_title("Feature Importances\nRed = most important feature (root split)")
ax2.axvline(x=0, color='gray', linewidth=0.8)
ax2.grid(True, alpha=0.25, axis='x')
for bar, val in zip(bars, [importances[i] for i in range(3, -1, -1)]):
    ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9)

# ── Panel 3: Train vs Test accuracy both trees ───────────────
ax3 = fig.add_subplot(2, 2, 3)
labels_bar  = ['No Limit\n(Overfit)', 'max_depth=3\n(Generalized)']
train_accs  = [train_acc_full * 100, train_acc_d3 * 100]
test_accs   = [test_acc_full  * 100, test_acc_d3  * 100]
x_pos       = np.arange(len(labels_bar))
width_b     = 0.32

b1 = ax3.bar(x_pos - width_b/2, train_accs, width_b,
             label='Train Accuracy', color='steelblue', alpha=0.85)
b2 = ax3.bar(x_pos + width_b/2, test_accs,  width_b,
             label='Test Accuracy',  color='tomato',    alpha=0.85)

for bar in list(b1) + list(b2):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels_bar)
ax3.set_ylim(85, 105)
ax3.set_ylabel("Accuracy (%)")
ax3.set_title("Train vs Test Accuracy\nSmaller gap = less overfitting")
ax3.legend()
ax3.grid(True, alpha=0.25, axis='y')
ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=1)

# ── Panel 4: Full unlimited tree (messy) ─────────────────────
ax4 = fig.add_subplot(2, 2, 4)
plot_tree(
    tree_full,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=5,
    ax=ax4
)
ax4.set_title(
    f"Unlimited Depth Tree  (depth={tree_full.get_depth()}, leaves={tree_full.get_n_leaves()})\n"
    "← This is what overfitting looks like. Too many splits, too complex.",
    fontsize=9
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("decision_tree_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → decision_tree_plot.png")