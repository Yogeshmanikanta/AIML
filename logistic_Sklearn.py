import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
 classification_report, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # suppress convergence warnings for clean output


# ============================================================
#  STEP 1 — LOAD THE BREAST CANCER DATASET
# ============================================================
# sklearn ships with several classic datasets built-in.
# Breast Cancer Wisconsin is one of them — no downloading needed.
#
# It contains 569 real patient tumor measurements.
# Each patient has 30 features (radius, texture, perimeter, area, etc.)
# extracted from a digitized image of a fine needle aspirate (FNA) of a tumor.
#
# Target labels:
#   0 = Malignant (cancerous)   — 212 patients
#   1 = Benign   (not cancer)   — 357 patients
#
# Our goal: given the 30 measurements, predict if a tumor is malignant or benign.

data = load_breast_cancer()

X = data.data          # shape: (569, 30)  — 569 patients, 30 features each
y = data.target        # shape: (569,)     — 0 or 1 label for each patient

print("=" * 60)
print("STEP 1 — DATASET OVERVIEW")
print("=" * 60)
print(f"Total patients   : {X.shape[0]}")
print(f"Features per row : {X.shape[1]}")
print(f"Feature names    : {', '.join(data.feature_names[:5])} ...")
print(f"Class names      : {list(data.target_names)}")
print(f"Malignant (0)    : {(y == 0).sum()}")
print(f"Benign    (1)    : {(y == 1).sum()}")


# ============================================================
#  STEP 2 — SPLIT INTO TRAIN AND TEST SETS
# ============================================================
# We never train and evaluate on the same data — that would be cheating!
# The model would just memorize answers instead of learning patterns.
#
# train_test_split randomly shuffles and divides data:
#   - 80% → training set  (the model learns from this)
#   - 20% → test set      (we evaluate on this — model never sees it during training)
#
# random_state=42 fixes the shuffle so results are reproducible every run.
# stratify=y ensures both splits have the same ratio of 0s and 1s.
# Without stratify, you might accidentally put all cancer cases in train!

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% goes to test
    random_state=42,     # reproducible split
    stratify=y           # preserve class balance in both splits
)

print("\n" + "=" * 60)
print("STEP 2 — TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Training samples : {X_train.shape[0]}  (80%)")
print(f"Test samples     : {X_test.shape[0]}   (20%)")
print(f"Train — Malignant: {(y_train==0).sum()}  |  Benign: {(y_train==1).sum()}")
print(f"Test  — Malignant: {(y_test==0).sum()}   |  Benign: {(y_test==1).sum()}")


# ============================================================
#  STEP 3 — FIT LOGISTIC REGRESSION MODEL
# ============================================================
# sklearn's LogisticRegression does everything we built manually before:
#   - initializes weights
#   - runs gradient descent (or a faster solver like 'lbfgs')
#   - finds the best w and b that minimize cross-entropy loss
#
# max_iter=10000: allow enough iterations to fully converge.
# C=1.0: regularization strength (prevents overfitting). Lower C = stronger regularization.
#
# .fit() is where all the learning happens.
# After this line, the model has found the optimal weights.

model = LogisticRegression(max_iter=10000, C=1.0, random_state=42)
model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("STEP 3 — MODEL TRAINING")
print("=" * 60)
print("LogisticRegression fitted successfully.")
print(f"Solver used      : lbfgs  (efficient gradient-based optimizer)")
print(f"Iterations ran   : {model.n_iter_[0]}")
print(f"Number of weights: {model.coef_.shape[1]}  (one per feature)")


# ============================================================
#  STEP 4 — PREDICT ON TEST SET & ACCURACY
# ============================================================
# model.predict() runs the forward pass on unseen test data:
#   z     = w · X + b
#   ŷ     = sigmoid(z)
#   label = 1 if ŷ >= 0.5 else 0
#
# Accuracy = (correct predictions) / (total predictions)
# It answers: "What % of patients did we classify correctly?"
#
# Accuracy alone can be misleading for imbalanced datasets —
# that's why we also look at Precision, Recall, and F1 below.

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("STEP 4 — ACCURACY ON TEST SET")
print("=" * 60)
print(f"Correct predictions : {(y_pred == y_test).sum()} / {len(y_test)}")
print(f"Accuracy Score      : {accuracy * 100:.2f}%")


# ============================================================
#  STEP 5 — CONFUSION MATRIX
# ============================================================
# The confusion matrix breaks down predictions into 4 buckets:
#
#                    Predicted: 0       Predicted: 1
#   Actual: 0   |  True Negative (TN) | False Positive (FP) |
#   Actual: 1   |  False Negative (FN)| True Positive  (TP) |
#
# In cancer detection context:
#   TN = Model said "Benign"    — actually Benign    ✓ correct
#   TP = Model said "Malignant" — actually Malignant ✓ correct
#   FP = Model said "Malignant" — actually Benign    ✗ false alarm (patient gets extra tests)
#   FN = Model said "Benign"    — actually Malignant ✗ MISSED CANCER ← worst mistake!
#
# A False Negative is the most dangerous error here.
# Missing a real cancer case could cost someone their life.

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("STEP 5 — CONFUSION MATRIX")
print("=" * 60)
print(f"\n  True Negatives  (TN) = {tn}  → Said Benign,    actually Benign    ✓")
print(f"  False Positives (FP) = {fp}  → Said Malignant, actually Benign    ✗ (false alarm)")
print(f"  False Negatives (FN) = {fn}  → Said Benign,    actually Malignant ✗ MISSED CANCER!")
print(f"  True Positives  (TP) = {tp} → Said Malignant, actually Malignant  ✓")


# ============================================================
#  STEP 6 — PRECISION, RECALL, F1
# ============================================================
# These three metrics together give a much fuller picture than accuracy.
#
# PRECISION = TP / (TP + FP)
#   "Of all the patients we flagged as Malignant, how many truly were?"
#   High precision = fewer false alarms. Patients aren't wrongly scared.
#
# RECALL (Sensitivity) = TP / (TP + FN)
#   "Of all actual Malignant cases, how many did we catch?"
#   High recall = fewer missed cancers. This is the CRITICAL metric here.
#   A recall of 0.95 means we caught 95% of real cancer cases.
#
# F1 SCORE = 2 * (Precision * Recall) / (Precision + Recall)
#   Harmonic mean of precision and recall.
#   Useful single number when you care about both, but recall matters more here.
#
# SUPPORT = number of actual samples in each class in the test set.
#
# For cancer detection:
#   Recall for class 0 (Malignant) is the most important metric.
#   Missing a cancer = deadly. A false alarm = extra tests (bad, but survivable).

print("\n" + "=" * 60)
print("STEP 6 — PRECISION, RECALL, F1  (Classification Report)")
print("=" * 60)
print(classification_report(y_test, y_pred,
                             target_names=data.target_names))


# ============================================================
#  STEP 7 — IS THIS MODEL GOOD ENOUGH FOR CANCER DETECTION?
# ============================================================
# Let's compute the key numbers and answer this honestly.

report = classification_report(y_test, y_pred,
                                target_names=data.target_names,
                                output_dict=True)

malignant_recall    = report['malignant']['recall']
malignant_precision = report['malignant']['precision']
malignant_f1        = report['malignant']['f1-score']

print("\n" + "=" * 60)
print("STEP 7 — IS THE MODEL GOOD ENOUGH FOR CANCER DETECTION?")
print("=" * 60)
print(f"""
The most important metric in cancer detection is RECALL for 
the Malignant class (class 0).

Why? Because a False Negative means we told a cancer patient 
they are fine. That mistake can be fatal.

A False Positive just means extra follow-up tests — costly 
and stressful, but not life-threatening.

── Our model's performance ──────────────────────────────────

  Accuracy              : {accuracy*100:.2f}%
  Malignant Recall      : {malignant_recall*100:.2f}%   ← caught this % of real cancers
  Malignant Precision   : {malignant_precision*100:.2f}%   ← this % of our cancer flags were real
  Malignant F1 Score    : {malignant_f1*100:.2f}%
  Missed cancer cases   : {fn}  (False Negatives)

── Verdict ──────────────────────────────────────────────────

  Recall of {malignant_recall*100:.1f}% is {"STRONG" if malignant_recall > 0.93 else "DECENT but needs improvement"}.
  
  In a real clinical setting, we'd want Recall > 95%, ideally 
  close to 99%. We would also tune the classification threshold 
  (currently 0.5) downward — say to 0.3 — so the model flags 
  more borderline cases as Malignant, trading some precision 
  for higher recall.
  
  For a baseline Logistic Regression with no feature engineering,
  this result is {"impressive and production-competitive." if malignant_recall > 0.93 else "a strong starting point."}
""")


# ============================================================
#  STEP 8 — PLOTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Logistic Regression — Breast Cancer Dataset (sklearn)", fontsize=13)

# ── Confusion Matrix heatmap ──────────────────────────────────
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=data.target_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(
    "Confusion Matrix\n"
    "FN (bottom-left) = missed cancers  ← minimize this!"
)

# ── Precision / Recall / F1 bar chart ────────────────────────
classes  = ['Malignant (0)', 'Benign (1)']
metrics  = ['Precision', 'Recall', 'F1-Score']
colors   = ['#E24B4A', '#378ADD', '#63B77A']
x        = np.arange(len(classes))
width    = 0.22

for j, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [report[cls][metric.lower().replace('-', '-').replace(' ', '-').replace('Precision','precision').replace('Recall','recall').replace('F1-Score','f1-score')]
            for cls in ['malignant', 'benign']]
    axes[1].bar(x + j * width, vals, width, label=metric, color=color, alpha=0.85)

axes[1].set_xticks(x + width)
axes[1].set_xticklabels(classes)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("Score")
axes[1].set_title("Precision / Recall / F1 per Class\n(Recall for Malignant = most critical bar)")
axes[1].legend()
axes[1].axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
axes[1].text(1.35, 0.96, '0.95 target', fontsize=8, color='gray')
axes[1].grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig("logistic_sklearn_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved -> logistic_sklearn_plot.png")