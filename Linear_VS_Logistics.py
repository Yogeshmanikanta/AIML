import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
#  STEP 1 — LOAD & SPLIT THE BREAST CANCER DATASET
# ============================================================
# Same dataset as before: 569 patients, 30 features, labels 0 or 1.
# We use only 1 feature (mean radius) so we can actually PLOT both
# models as a line over data — makes the visual comparison obvious.
#
# mean radius = the average size of the tumor cell nuclei.
# Larger radius tends to mean malignant; smaller tends to mean benign.
# It's the single most informative feature in the dataset.

data    = load_breast_cancer()
X_full  = data.data
y       = data.target                        # 0 = malignant, 1 = benign

# Use only feature 0 (mean radius) for plotting clarity
X       = X_full[:, 0].reshape(-1, 1)        # shape: (569, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y           # preserve class balance
)

print("=" * 62)
print("STEP 1 — DATASET LOADED")
print("=" * 62)
print(f"Feature used     : mean radius  (feature 0 of 30)")
print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Label range      : 0 (malignant) or 1 (benign) — binary")


# ============================================================
#  STEP 2 — FIT LINEAR REGRESSION (the wrong tool)
# ============================================================
# LinearRegression was designed to predict CONTINUOUS values like
# house prices, temperatures, salaries — numbers with no upper/lower bound.
#
# Here we're FORCING it to predict labels 0 and 1.
# It will try to fit a straight line through the binary data,
# but a straight line has NO concept of "stop at 0" or "stop at 1".
#
# The result: predictions will spill outside [0, 1].
# For example, very large tumors might get predicted as -0.3
# and very small ones as 1.4 — neither makes sense as a probability.
#
# We then ROUND predictions to get 0 or 1 (a crude hack to produce labels).
# This is exactly what you should NOT do in real classification problems.

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Raw predictions — these are NOT probabilities, just real numbers
y_pred_lin_raw    = lin_model.predict(X_test)

# Round to nearest integer (0 or 1) to get class labels
y_pred_lin_rounded = np.round(y_pred_lin_raw).astype(int)
# Clip to 0–1 in case of extreme values like -1 or 2
y_pred_lin_rounded = np.clip(y_pred_lin_rounded, 0, 1)

lin_accuracy = accuracy_score(y_test, y_pred_lin_rounded)
lin_f1       = f1_score(y_test, y_pred_lin_rounded)

print("\n" + "=" * 62)
print("STEP 2 — LINEAR REGRESSION (forced onto classification)")
print("=" * 62)
print(f"Min raw prediction : {y_pred_lin_raw.min():.4f}   ← below 0, impossible probability!")
print(f"Max raw prediction : {y_pred_lin_raw.max():.4f}   ← above 1, impossible probability!")
print(f"Predictions outside [0,1]: {((y_pred_lin_raw < 0) | (y_pred_lin_raw > 1)).sum()} out of {len(y_pred_lin_raw)}")
print(f"\nAfter rounding to 0 or 1:")
print(f"  Accuracy : {lin_accuracy * 100:.2f}%")
print(f"  F1 Score : {lin_f1:.4f}")


# ============================================================
#  STEP 3 — FIT LOGISTIC REGRESSION (the right tool)
# ============================================================
# LogisticRegression was BUILT for binary classification.
# It applies the sigmoid function to keep all outputs strictly
# between 0 and 1 — so every output IS a valid probability.
#
# sigmoid(z) = 1 / (1 + e^(-z))
#   → always outputs between 0 and 1
#   → 0.5 = decision boundary (predict 1 if > 0.5, else 0)
#
# No rounding hack needed. No impossible values. Clean probabilities.

log_model = LogisticRegression(max_iter=10000, random_state=42)
log_model.fit(X_train, y_train)

y_pred_log      = log_model.predict(X_test)
y_prob_log      = log_model.predict_proba(X_test)[:, 1]   # P(benign)

log_accuracy = accuracy_score(y_test, y_pred_log)
log_f1       = f1_score(y_test, y_pred_log)

print("\n" + "=" * 62)
print("STEP 3 — LOGISTIC REGRESSION (the right tool)")
print("=" * 62)
print(f"Min predicted probability : {y_prob_log.min():.4f}   ← always >= 0 ✓")
print(f"Max predicted probability : {y_prob_log.max():.4f}   ← always <= 1 ✓")
print(f"Predictions outside [0,1] : 0 — impossible by design")
print(f"\nResults:")
print(f"  Accuracy : {log_accuracy * 100:.2f}%")
print(f"  F1 Score : {log_f1:.4f}")


# ============================================================
#  STEP 4 — SIDE-BY-SIDE COMPARISON
# ============================================================
# Now let's compare the two models head-to-head on every metric.
#
# F1 Score matters here more than accuracy because the dataset is
# slightly imbalanced (more benign than malignant). F1 balances
# Precision and Recall — it's a harder, fairer test than accuracy.

print("\n" + "=" * 62)
print("STEP 4 — HEAD-TO-HEAD COMPARISON")
print("=" * 62)
print(f"{'Metric':<22} {'Linear Regression':>20} {'Logistic Regression':>20}")
print("-" * 64)
print(f"{'Accuracy':<22} {lin_accuracy*100:>19.2f}% {log_accuracy*100:>19.2f}%")
print(f"{'F1 Score':<22} {lin_f1:>20.4f} {log_f1:>20.4f}")
print(f"{'Values outside [0,1]':<22} {((y_pred_lin_raw<0)|(y_pred_lin_raw>1)).sum():>20} {'0':>20}")
print("-" * 64)
winner_acc = "Logistic" if log_accuracy >= lin_accuracy else "Linear"
winner_f1  = "Logistic" if log_f1 >= lin_f1 else "Linear"
print(f"{'Winner (Accuracy)':<22} {'':>20} {winner_acc:>20} ✓")
print(f"{'Winner (F1)':<22} {'':>20} {winner_f1:>20} ✓")

print('''Linear won the number game here — but for the wrong reasons: The breast cancer data with just 1 feature happens to be almost perfectly linearly separable. The tumors with large radius are mostly malignant, small radius are mostly benign — a straight line accidentally lands near the rig''')

# ============================================================
#  STEP 5 — WHY LOGISTIC REGRESSION WINS (3 sentences)
# ============================================================

print("\n" + "=" * 62)
print("STEP 5 — WHY LOGISTIC REGRESSION WINS")
print("=" * 62)
print("""
1. Linear Regression outputs unbounded real numbers (like -0.3 or
   1.4) which are meaningless as probabilities, whereas Logistic
   Regression uses the sigmoid function to guarantee all outputs
   stay strictly between 0 and 1 — making them valid probabilities.

2. When Linear Regression is forced onto binary labels (0/1), its
   straight-line fit gets distorted by extreme data points on both
   ends, shifting the decision boundary away from the optimal
   position; Logistic Regression's S-curve naturally hugs the data
   and places the boundary exactly where the classes separate.

3. Logistic Regression optimizes Binary Cross-Entropy loss, which
   is specifically designed to penalize wrong confident predictions
   in classification tasks, whereas Linear Regression minimizes MSE
   — a loss function that treats predicting 0.6 for a label of 1
   as almost as bad as predicting 0.4, ignoring the 0/1 structure
   of the problem entirely.
""")


# ============================================================
#  STEP 6 — PLOTS
# ============================================================
# Four plots arranged in a 2x2 grid:
#
#  [Top-Left]  Raw predictions of BOTH models over mean radius.
#              Shows Linear going outside [0,1], Logistic staying inside.
#
#  [Top-Right] Zoomed view of the problematic region where Linear
#              produces values < 0 or > 1 — highlighted in red.
#
#  [Bot-Left]  Confusion matrix for Linear Regression after rounding.
#
#  [Bot-Right] Confusion matrix for Logistic Regression.

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Linear vs Logistic Regression on Breast Cancer Dataset\n"
    "(single feature: mean radius)",
    fontsize=13
)

# Smooth x axis for drawing model curves
x_range     = np.linspace(X.min() - 1, X.max() + 1, 400).reshape(-1, 1)
lin_curve   = lin_model.predict(x_range)
log_curve   = log_model.predict_proba(x_range)[:, 1]   # P(benign)


# ── TOP-LEFT: Both model curves over scattered data ──────────
ax1 = axes[0, 0]

# Data points: green = benign (1), red = malignant (0)
ax1.scatter(X_test[y_test == 1], y_test[y_test == 1],
            color='seagreen', alpha=0.5, s=30, label='Benign (1)', zorder=4)
ax1.scatter(X_test[y_test == 0], y_test[y_test == 0],
            color='tomato', alpha=0.5, s=30, label='Malignant (0)', zorder=4)

# Linear regression line — goes below 0 and above 1
ax1.plot(x_range, lin_curve,
         color='darkorange', linewidth=2, linestyle='--',
         label='Linear Regression (raw)')

# Logistic sigmoid curve — stays in [0, 1]
ax1.plot(x_range, log_curve,
         color='steelblue', linewidth=2.5,
         label='Logistic Regression P(benign)')

# Safe zone band
ax1.axhspan(0, 1, alpha=0.06, color='green', label='Valid probability zone [0,1]')
ax1.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
ax1.axhline(y=1, color='gray', linewidth=0.8, linestyle=':')
ax1.axhline(y=0.5, color='purple', linewidth=1, linestyle='--', alpha=0.6, label='Decision boundary (0.5)')

ax1.set_xlabel("Mean Radius")
ax1.set_ylabel("Predicted value / Probability")
ax1.set_title("Both Models — Raw Output\n(Orange line exits valid zone!)")
ax1.legend(fontsize=7.5)
ax1.grid(True, alpha=0.25)
ax1.set_ylim(-0.5, 1.5)


# ── TOP-RIGHT: Highlight out-of-bounds predictions ───────────
ax2 = axes[0, 1]

# Show where linear regression predicts outside [0, 1]
out_of_bounds_mask = (lin_curve < 0) | (lin_curve > 1)

ax2.plot(x_range, lin_curve, color='darkorange', linewidth=2,
         linestyle='--', label='Linear output')
ax2.plot(x_range, log_curve, color='steelblue', linewidth=2.5,
         label='Logistic output')

# Shade the dangerous out-of-bounds regions
ax2.fill_between(x_range.ravel(), lin_curve,
                 where=(lin_curve < 0),
                 color='red', alpha=0.3, label='Output < 0  ← impossible!')
ax2.fill_between(x_range.ravel(), lin_curve,
                 where=(lin_curve > 1),
                 color='red', alpha=0.3, label='Output > 1  ← impossible!')

ax2.axhline(y=0, color='black', linewidth=1)
ax2.axhline(y=1, color='black', linewidth=1)
ax2.set_xlabel("Mean Radius")
ax2.set_ylabel("Predicted value")
ax2.set_title("Out-of-Bounds Predictions (Red Zones)\nLogistic never leaves [0, 1]")
ax2.legend(fontsize=7.5)
ax2.grid(True, alpha=0.25)
ax2.set_ylim(-0.5, 1.5)


# ── BOT-LEFT: Confusion matrix — Linear ──────────────────────
ax3 = axes[1, 0]
cm_lin = confusion_matrix(y_test, y_pred_lin_rounded)
disp_lin = ConfusionMatrixDisplay(cm_lin, display_labels=data.target_names)
disp_lin.plot(ax=ax3, colorbar=False, cmap='Oranges')
ax3.set_title(
    f"Linear Regression — Confusion Matrix\n"
    f"Accuracy: {lin_accuracy*100:.2f}%  |  F1: {lin_f1:.3f}"
)


# ── BOT-RIGHT: Confusion matrix — Logistic ───────────────────
ax4 = axes[1, 1]
cm_log = confusion_matrix(y_test, y_pred_log)
disp_log = ConfusionMatrixDisplay(cm_log, display_labels=data.target_names)
disp_log.plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title(
    f"Logistic Regression — Confusion Matrix\n"
    f"Accuracy: {log_accuracy*100:.2f}%  |  F1: {log_f1:.3f}"
)

plt.tight_layout()
plt.savefig("linear_vs_logistic_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved -> linear_vs_logistic_plot.png")