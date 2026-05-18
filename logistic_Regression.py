import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  PART 1 — VISUALIZE THE SIGMOID FUNCTION
# ============================================================

# The sigmoid "squashes" any real number into the range (0, 1)
# This is what makes it perfect for outputting probabilities

def sigmoid(z):
    """
    Sigmoid function: converts any number z into a probability.
    
    Formula: σ(z) = 1 / (1 + e^(-z))
    
    Key properties:
      - sigmoid(0)    = 0.5       <- exactly in the middle
      - sigmoid(+inf) → 1.0      <- very confident it's class 1
      - sigmoid(-inf) → 0.0      <- very confident it's class 0
    """
    return 1 / (1 + np.exp(-z))


# Create z values from -10 to 10 (the "input" to sigmoid)
z_values = np.linspace(-10, 10, 300)

# Pass all z values through sigmoid to get probabilities
sigmoid_values = sigmoid(z_values)

# ── Plot the S-curve ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Logistic Regression from Scratch  (numpy, no sklearn)", fontsize=13, y=1.02)

ax1 = axes[0]
ax1.plot(z_values, sigmoid_values, color='steelblue', linewidth=2.5, label='σ(z)')

# Mark z = 0 → sigmoid output is exactly 0.5
ax1.axvline(x=0, color='tomato', linestyle='--', linewidth=1.5, label='z = 0  →  σ = 0.5')
ax1.axhline(y=0.5, color='tomato', linestyle='--', linewidth=1.5)
ax1.scatter([0], [0.5], color='tomato', zorder=5, s=80)

# Horizontal guide lines at 0 and 1
ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax1.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax1.set_xlabel("z  (linear combination w·x + b)")
ax1.set_ylabel("σ(z)  =  probability")
ax1.set_title("Sigmoid Function  —  the S-Curve")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


# ============================================================
#  PART 2 — CREATE FAKE BINARY DATASET
# ============================================================

# Scenario: 50 students, each studied X hours.
# Did they pass (1) or fail (0)?
# Students who study more tend to pass — but there's some noise.

np.random.seed(42)                              # fix seed for reproducibility

n_students = 50

# Study hours: randomly between 1 and 10 hours
study_hours = np.random.uniform(1, 10, n_students)

# Pass/fail label: students with more hours are more likely to pass.
# We use a probability based on hours to simulate realistic noise.
true_probability = sigmoid(1.5 * study_hours - 7.5)   # hidden true relationship
labels = np.random.binomial(1, true_probability)       # 1 = pass, 0 = fail

print("Dataset preview (first 10 students):")
print(f"{'Study Hours':>12}  {'Pass/Fail':>10}")
print("-" * 25)
for i in range(10):
    result = "PASS ✓" if labels[i] == 1 else "FAIL ✗"
    print(f"{study_hours[i]:>12.2f}  {result:>10}")
print(f"\nTotal students: {n_students}  |  Passed: {labels.sum()}  |  Failed: {n_students - labels.sum()}")


# ============================================================
#  PART 3 — NORMALIZE FEATURES
# ============================================================

# Always normalize before gradient descent!
# Raw study hours (1–10) would make gradients unstable.
# After normalization: mean≈0, std≈1 → smooth, stable training.

X_mean = study_hours.mean()
X_std  = study_hours.std()
X_norm = (study_hours - X_mean) / X_std   # normalized input


# ============================================================
#  PART 4 — INITIALIZE WEIGHTS
# ============================================================

# Start with zero — gradient descent will find the right values
w = 0.0   # weight (controls the slope of the decision boundary)
b = 0.0   # bias   (controls the shift of the decision boundary)


# ============================================================
#  PART 5 — PREDICTION FUNCTION
# ============================================================

def predict(w, b, X):
    """
    Logistic regression prediction.
    
    Step 1: Compute the linear part z = w*x + b
    Step 2: Squash z through sigmoid to get a probability (0 to 1)
    
    Output: probability that the student passes
    """
    z     = w * X + b        # linear combination (like linear regression)
    y_hat = sigmoid(z)       # convert to probability using sigmoid
    return y_hat


# ============================================================
#  PART 6 — BINARY CROSS ENTROPY LOSS
# ============================================================

def binary_cross_entropy(y_hat, y):
    """
    Measures how wrong our predictions are for binary (0/1) labels.
    
    Formula:  Loss = -mean( y*log(ŷ) + (1-y)*log(1-ŷ) )
    
    Intuition:
      - If y=1 and ŷ≈1 → loss is small  (correct & confident)
      - If y=1 and ŷ≈0 → loss is huge   (wrong & confident)
      - If y=0 and ŷ≈0 → loss is small  (correct & confident)
      - If y=0 and ŷ≈1 → loss is huge   (wrong & confident)
    
    We clip ŷ slightly away from 0 and 1 to avoid log(0) = -infinity
    """
    epsilon = 1e-8                                          # tiny number for numerical safety
    y_hat   = np.clip(y_hat, epsilon, 1 - epsilon)         # keep ŷ in (0, 1)
    loss    = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss


# ============================================================
#  PART 7 — GRADIENTS
# ============================================================

def compute_gradients(X, y, y_hat):
    """
    Compute how much to nudge w and b to reduce the loss.
    
    These come from calculus (chain rule on BCE loss):
    
      dL/dw = mean( (ŷ - y) * x )   ← how loss changes with w
      dL/db = mean( (ŷ - y) )        ← how loss changes with b
    
    Beautiful property: sigmoid's derivative simplifies the math
    to just (ŷ - y), making logistic regression very clean.
    """
    error = y_hat - y                         # prediction error per student
    dw    = np.mean(error * X)                # gradient w.r.t. weight
    db    = np.mean(error)                    # gradient w.r.t. bias
    return dw, db


# ============================================================
#  PART 8 — GRADIENT DESCENT TRAINING LOOP
# ============================================================

learning_rate = 0.1     # how big each step is (tuned for normalized data)
iterations    = 1000    # number of times we update w and b
loss_history  = []      # track loss at each step to see it decrease

print(f"\n{'Iteration':>10}  {'BCE Loss':>12}  {'w':>10}  {'b':>10}")
print("-" * 48)

for i in range(1, iterations + 1):

    # Step 1: Forward pass — make predictions with current w, b
    y_hat = predict(w, b, X_norm)

    # Step 2: Compute how bad our predictions are
    loss = binary_cross_entropy(y_hat, labels)

    # Step 3: Compute gradients (direction to improve)
    dw, db = compute_gradients(X_norm, labels, y_hat)

    # Step 4: Update weights in the opposite direction of gradient
    #         (gradient points uphill, we want to go downhill)
    w -= learning_rate * dw
    b -= learning_rate * db

    # Save loss for plotting
    loss_history.append(loss)

    # Print every 100 iterations so we can watch loss decrease
    if i % 100 == 0:
        print(f"{i:>10}  {loss:>12.6f}  {w:>10.4f}  {b:>10.4f}")

print("-" * 48)
print(f"\nFinal  w = {w:.6f}   b = {b:.6f}")
print(f"Final BCE loss = {loss_history[-1]:.6f}")


# ============================================================
#  PART 9 — ACCURACY CHECK
# ============================================================

# Convert probabilities → class labels (threshold at 0.5)
# If model says P(pass) >= 0.5 → predict PASS, else FAIL
final_probs = predict(w, b, X_norm)
predictions = (final_probs >= 0.5).astype(int)
accuracy    = np.mean(predictions == labels) * 100
print(f"Training Accuracy = {accuracy:.1f}%")


# ============================================================
#  PART 10 — PLOT DATA + DECISION BOUNDARY
# ============================================================

ax2 = axes[1]

# Separate students by pass/fail for coloring
passed = study_hours[labels == 1]
failed = study_hours[labels == 0]

# Plot scatter: green = passed, red = failed
ax2.scatter(passed, np.ones(len(passed)),  color='seagreen', s=60,
            zorder=5, label='Pass (y=1)', alpha=0.7)
ax2.scatter(failed, np.zeros(len(failed)), color='tomato',   s=60,
            zorder=5, label='Fail (y=0)', alpha=0.7)

# Plot the sigmoid curve (the learned decision function)
x_line      = np.linspace(0, 11, 300)
x_line_norm = (x_line - X_mean) / X_std
prob_line   = predict(w, b, x_line_norm)

ax2.plot(x_line, prob_line, color='steelblue', linewidth=2.5,
         label='P(pass | hours)')

# Decision boundary = where probability = 0.5
# Solve: w * x_norm + b = 0  →  x_norm = -b/w  →  x = x_norm*std + mean
boundary_norm = -b / w
boundary_x    = boundary_norm * X_std + X_mean
ax2.axvline(x=boundary_x, color='purple', linestyle='--', linewidth=1.8,
            label=f'Decision boundary ≈ {boundary_x:.1f} hrs')

ax2.set_xlabel("Study Hours")
ax2.set_ylabel("Probability of Passing")
ax2.set_title(f"Logistic Regression  —  Accuracy: {accuracy:.1f}%")
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlim(0, 11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)


# ── Loss Curve ───────────────────────────────────────────────
ax3 = axes[2]
ax3.plot(range(1, iterations + 1), loss_history, color='darkorange', linewidth=1.8)
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Binary Cross Entropy Loss")
ax3.set_title("Loss Decreasing over 1000 Iterations")
ax3.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig("logistic_regression_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved -> logistic_regression_plot.png")