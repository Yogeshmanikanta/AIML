import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)


# ============================================================
#  STEP 1 — LOAD BREAST CANCER DATASET
# ============================================================
# 569 patients, 30 features (tumour measurements), binary target:
#   0 = Malignant (cancerous)   — 212 patients
#   1 = Benign   (not cancer)   — 357 patients
#
# This is our battleground. All four models will compete on the
# EXACT SAME split with the EXACT SAME scaled features.
# No model gets any advantage — pure head-to-head comparison.
#
# Most important metric for cancer detection: RECALL for Malignant (0)
#   = "Of all real cancer patients, what % did we correctly flag?"
#   A False Negative (missed cancer) can cost a life.
#   A False Positive (false alarm) leads to extra tests — uncomfortable
#   but survivable. So we care MORE about recall than precision here.

data = load_breast_cancer()
X, y = data.data, data.target

print("=" * 66)
print("STEP 1 — BREAST CANCER DATASET")
print("=" * 66)
print(f"  Patients  : {X.shape[0]}")
print(f"  Features  : {X.shape[1]}")
print(f"  Malignant : {(y==0).sum()}  (class 0 — the dangerous one)")
print(f"  Benign    : {(y==1).sum()}  (class 1)")
print(f"\n  Critical metric → RECALL for Malignant (class 0)")
print(f"  Because: missing a cancer = False Negative = potentially fatal")


# ============================================================
#  STEP 2 — SPLIT + SCALE
# ============================================================
# Fit scaler on train only — no data leakage to test set.
# Mandatory for SVM and KNN (distance-based).
# Helps Logistic Regression converge faster.
# Random Forest doesn't need it (tree splits are scale-invariant)
# — but we apply it anyway so all models see the same data.

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print("\n" + "=" * 66)
print("STEP 2 — SPLIT + SCALE")
print("=" * 66)
print(f"  Train : {X_train.shape[0]} patients  (80%)")
print(f"  Test  : {X_test.shape[0]}  patients  (20%)")
print(f"  Scaler: StandardScaler fitted on train only")


# ============================================================
#  STEP 3 — DEFINE AND TRAIN ALL FOUR MODELS
# ============================================================
# We time each model's training separately so we can compare speed.
#
# SVC(kernel='rbf')     — finds the maximum-margin curved boundary
#                         in an implicit high-dimensional space.
#                         Excellent on tabular medical data.
#
# KNeighborsClassifier  — no "training" at all! Just memorises data.
#                         Prediction time is slow (searches all points).
#
# LogisticRegression    — fits a sigmoid curve via gradient descent.
#                         Fast, interpretable, strong baseline.
#
# RandomForestClassifier — 100 decision trees vote together.
#                          Robust, handles outliers, no scaling needed.

models_def = {
    "SVM  (RBF kernel)":          SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "KNN  (k=5)":                  KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression":         LogisticRegression(max_iter=10000, random_state=42),
    "Random Forest (100 trees)":   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

print("\n" + "=" * 66)
print("STEP 3 — TRAINING ALL FOUR MODELS")
print("=" * 66)

results    = {}
train_times = {}
pred_times  = {}
trained_models = {}

for name, model in models_def.items():
    # ── Train ────────────────────────────────────────────────
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = (time.perf_counter() - t0) * 1000   # milliseconds

    # ── Predict ──────────────────────────────────────────────
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = (time.perf_counter() - t1) * 1000

    # ── Metrics ──────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0)   # for Malignant
    rec  = recall_score(y_test, y_pred, pos_label=0)      # for Malignant
    f1   = f1_score(y_test, y_pred, pos_label=0)          # for Malignant

    # ── 5-fold cross-validation ──────────────────────────────
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')

    results[name] = {
        "Accuracy":       round(acc * 100,  2),
        "Precision (M)":  round(prec * 100, 2),
        "Recall (M)":     round(rec * 100,  2),
        "F1 (M)":         round(f1 * 100,   2),
        "CV Recall (5F)": round(cv_scores.mean() * 100, 2),
    }
    train_times[name] = round(train_time, 2)
    pred_times[name]  = round(pred_time, 2)
    trained_models[name] = (model, y_pred)

    print(f"\n  ── {name}")
    print(f"     Train time  : {train_time:>8.2f} ms")
    print(f"     Predict time: {pred_time:>8.2f} ms")
    print(f"     Accuracy    : {acc*100:>7.2f}%")
    print(f"     Recall (M)  : {rec*100:>7.2f}%   ← most important metric")
    print(f"     F1 (M)      : {f1*100:>7.2f}%")


# ============================================================
#  STEP 4 — COMPARISON TABLE
# ============================================================
# Build a tidy pandas DataFrame so all numbers sit side-by-side.
# Sort by Recall (Malignant) — the metric that matters most here.

df_results = pd.DataFrame(results).T
df_results.index.name = "Model"
df_results_sorted = df_results.sort_values("Recall (M)", ascending=False)

print("\n" + "=" * 66)
print("STEP 4 — COMPARISON TABLE  (sorted by Recall for Malignant)")
print("=" * 66)
print(df_results_sorted.to_string())

print("\n  (M) = computed for Malignant class (class 0) — the dangerous one")
print(f"  Best Recall  → {df_results_sorted['Recall (M)'].idxmax()}")
print(f"  Best Accuracy→ {df_results_sorted['Accuracy'].idxmax()}")
print(f"  Best F1      → {df_results_sorted['F1 (M)'].idxmax()}")


# ============================================================
#  STEP 5 — TRAINING + PREDICTION SPEED
# ============================================================

print("\n" + "=" * 66)
print("STEP 5 — SPEED COMPARISON")
print("=" * 66)
print(f"\n  {'Model':<35}  {'Train (ms)':>12}  {'Predict (ms)':>14}")
print(f"  {'-'*65}")
for name in models_def:
    print(f"  {name:<35}  {train_times[name]:>12.2f}  {pred_times[name]:>14.2f}")

fastest_train  = min(train_times, key=train_times.get)
slowest_train  = max(train_times, key=train_times.get)
fastest_pred   = min(pred_times,  key=pred_times.get)
slowest_pred   = max(pred_times,  key=pred_times.get)

print(f"\n  Fastest to train  : {fastest_train}")
print(f"  Slowest to train  : {slowest_train}")
print(f"  Fastest to predict: {fastest_pred}")
print(f"  Slowest to predict: {slowest_pred}")
print(f"\n  Why KNN is slow to predict:")
print(f"  It stores ALL training data — at prediction time it computes")
print(f"  distances to every single training point (455 patients here).")
print(f"  Logistic Regression just does one dot product — nearly instant.")


# ============================================================
#  STEP 6 — DEPLOY DECISION
# ============================================================

best_recall_model = df_results_sorted['Recall (M)'].idxmax()
best_recall_score = df_results_sorted['Recall (M)'].max()
best_acc_model    = df_results_sorted['Accuracy'].idxmax()
best_f1_model     = df_results_sorted['F1 (M)'].idxmax()

print("\n" + "=" * 66)
print("STEP 6 — WHICH MODEL WOULD YOU DEPLOY FOR CANCER DETECTION?")
print("=" * 66)
print(f"""
THE CASE FOR RECALL OVER ACCURACY
──────────────────────────────────
In cancer detection, a False Negative (FN) means:
  → Model predicts "Benign" for a patient who actually HAS cancer.
  → Patient goes untreated. Cancer progresses. Potentially fatal.

A False Positive (FP) means:
  → Model predicts "Malignant" for a patient who is actually benign.
  → Patient gets additional tests. Stressful and costly, but survivable.

Conclusion: we want MAXIMUM RECALL for Malignant class (class 0).
We are willing to sacrifice some precision (more false alarms)
to ensure we catch every real cancer case.

RESULTS SUMMARY
───────────────
  Best Recall for Malignant : {best_recall_model}  ({best_recall_score:.1f}%)
  Best Accuracy             : {best_acc_model}
  Best F1 (Malignant)       : {best_f1_model}

DEPLOYMENT RECOMMENDATION
──────────────────────────
Primary choice : {best_recall_model}
Reason         : Highest recall for malignant class — catches the most
                 real cancer cases among all four models tested.

Secondary      : Logistic Regression
Reason         : Interpretable (doctors can see feature weights),
                 fast to train, fast to predict, and competitive recall.
                 In clinical settings, interpretability builds trust.

Do NOT deploy  : KNN as sole model in production
Reason         : Slow prediction time scales badly with more patients.
                 In a busy hospital scanning thousands of results,
                 speed matters alongside accuracy.

In practice    : Run multiple models as an ensemble. Flag any patient
                 that AT LEAST ONE model marks as malignant. Then have
                 a human radiologist make the final call.
                 Machine learning ASSISTS doctors — it does not replace them.
""")


# ============================================================
#  STEP 7 — FULL CLASSIFICATION REPORTS
# ============================================================

print("=" * 66)
print("STEP 7 — FULL CLASSIFICATION REPORTS (all models)")
print("=" * 66)
for name, (model, y_pred) in trained_models.items():
    print(f"\n── {name}")
    print(classification_report(y_test, y_pred,
                                 target_names=['Malignant(0)', 'Benign(1)']))


# ============================================================
#  STEP 8 — PLOTS
# ============================================================

model_names_short = ["SVM\n(RBF)", "KNN\n(k=5)", "Logistic\nReg.", "Random\nForest"]
metrics_to_plot   = ["Accuracy", "Recall (M)", "F1 (M)"]
colors_models     = ['#5B8DB8', '#E47A7A', '#6CBF7F', '#C99A3F']

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Head-to-Head: SVM vs KNN vs Logistic Regression vs Random Forest\n"
             "Breast Cancer Dataset  (30 features · 569 patients · binary classification)",
             fontsize=13, y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)


# ── Panel row 0: Grouped metric bars ─────────────────────────
ax_main = fig.add_subplot(gs[0, :])   # full-width top panel

x      = np.arange(len(model_names_short))
n_met  = len(metrics_to_plot)
width  = 0.22
offsets = np.linspace(-(n_met-1)/2 * width, (n_met-1)/2 * width, n_met)
metric_colors = ['#378ADD', '#E24B4A', '#63B77A']

for j, (metric, mc) in enumerate(zip(metrics_to_plot, metric_colors)):
    vals = [df_results.loc[full_name, metric]
            for full_name in models_def.keys()]
    bars = ax_main.bar(x + offsets[j], vals, width,
                       label=metric, color=mc, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax_main.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)

ax_main.set_xticks(x)
ax_main.set_xticklabels(model_names_short, fontsize=11)
ax_main.set_ylim(60, 107)
ax_main.set_ylabel("Score (%)")
ax_main.set_title("Accuracy vs Recall (Malignant) vs F1 (Malignant)\n"
                   "Red bars = Recall for Malignant class — THE metric that matters for cancer detection",
                   fontsize=10)
ax_main.legend(fontsize=9)
ax_main.grid(True, alpha=0.2, axis='y')
ax_main.axhline(y=95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax_main.text(3.55, 95.3, '95% line', fontsize=8, color='gray')


# ── Panels row 1: Confusion matrices ─────────────────────────
for col, (name, (model, y_pred)) in enumerate(trained_models.items()):
    ax = fig.add_subplot(gs[1, col])
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(cm, display_labels=['Malig.', 'Benign'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    rec_m = recall_score(y_test, y_pred, pos_label=0) * 100
    ax.set_title(f"{model_names_short[col].replace(chr(10),' ')}\n"
                 f"FN={fn} missed cancers · Recall={rec_m:.1f}%",
                 fontsize=8)


# ── Panels row 2 left: Speed comparison ──────────────────────
ax_speed_tr = fig.add_subplot(gs[2, 0:2])
tr_vals = [train_times[n] for n in models_def]
pr_vals = [pred_times[n]  for n in models_def]
x2      = np.arange(4)
b1 = ax_speed_tr.bar(x2 - 0.2, tr_vals, 0.38,
                      label='Train time (ms)', color='steelblue', alpha=0.85)
b2 = ax_speed_tr.bar(x2 + 0.2, pr_vals, 0.38,
                      label='Predict time (ms)', color='darkorange', alpha=0.85)
for bar in list(b1) + list(b2):
    ax_speed_tr.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
ax_speed_tr.set_xticks(x2)
ax_speed_tr.set_xticklabels(model_names_short, fontsize=9)
ax_speed_tr.set_ylabel("Time (ms)")
ax_speed_tr.set_title("Training & Prediction Speed\n"
                       "KNN trains instantly but predicts slowly (searches all points at predict time)",
                       fontsize=9)
ax_speed_tr.legend(fontsize=9)
ax_speed_tr.grid(True, alpha=0.2, axis='y')


# ── Panel row 2 right: Radar-style summary bars ──────────────
ax_rank = fig.add_subplot(gs[2, 2:4])
criteria   = ["Accuracy", "Recall\n(Malignant)", "F1\n(Malignant)",
              "Train\nSpeed", "Predict\nSpeed", "Interpretability"]

# Normalised scores (0-100) for each criterion per model
# Speed and interpretability are manual qualitative scores
interp_scores = {
    "SVM  (RBF kernel)":          [df_results.loc["SVM  (RBF kernel)","Accuracy"],
                                    df_results.loc["SVM  (RBF kernel)","Recall (M)"],
                                    df_results.loc["SVM  (RBF kernel)","F1 (M)"],
                                    70, 80, 30],
    "KNN  (k=5)":                  [df_results.loc["KNN  (k=5)","Accuracy"],
                                    df_results.loc["KNN  (k=5)","Recall (M)"],
                                    df_results.loc["KNN  (k=5)","F1 (M)"],
                                    95, 40, 70],
    "Logistic Regression":         [df_results.loc["Logistic Regression","Accuracy"],
                                    df_results.loc["Logistic Regression","Recall (M)"],
                                    df_results.loc["Logistic Regression","F1 (M)"],
                                    98, 98, 95],
    "Random Forest (100 trees)":   [df_results.loc["Random Forest (100 trees)","Accuracy"],
                                    df_results.loc["Random Forest (100 trees)","Recall (M)"],
                                    df_results.loc["Random Forest (100 trees)","F1 (M)"],
                                    80, 85, 50],
}

x3    = np.arange(len(criteria))
width3 = 0.18
for j, (name, scores) in enumerate(interp_scores.items()):
    offset = (j - 1.5) * width3
    ax_rank.bar(x3 + offset, scores, width3,
                label=model_names_short[j].replace('\n',' '),
                color=colors_models[j], alpha=0.85, edgecolor='white')

ax_rank.set_xticks(x3)
ax_rank.set_xticklabels(criteria, fontsize=8.5)
ax_rank.set_ylim(0, 110)
ax_rank.set_ylabel("Score / 100")
ax_rank.set_title("Multi-Criteria Comparison\n"
                   "(last 2 cols are qualitative: speed & interpretability)",
                   fontsize=9)
ax_rank.legend(fontsize=7.5, ncol=2)
ax_rank.grid(True, alpha=0.2, axis='y')

plt.savefig("model_comparison_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → model_comparison_plot.png")