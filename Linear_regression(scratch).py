import numpy as np
import matplotlib.pyplot as plt

# ── DATA ──────────────────────────────────────────────────────────────────
X = np.array([500, 750, 1000, 1250, 1500], dtype=float)
Y = np.array([150, 200,  250,  300,  350], dtype=float)
n = len(X)

# ── NORMALIZE X so that lr = 0.000001 is stable ───────────────────────────
# (raw X values ~1000 make gradients huge; normalization keeps them ~O(1))
X_mean = X.mean()   # 1000.0
X_std  = X.std()    #  353.6
X_norm = (X - X_mean) / X_std

# ── INITIALIZE WEIGHTS ────────────────────────────────────────────────────
w = 0.0
b = 0.0

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────
learning_rate = 0.01       # sensible lr after normalization
iterations    = 500

# ── CORE FUNCTIONS ────────────────────────────────────────────────────────

def predict(w, b, X):
    """Forward pass: y_hat = w * x + b"""
    return w * X + b

def mse_loss(y_hat, Y):
    """Mean Squared Error = mean((y_hat - y)^2)"""
    return np.mean((y_hat - Y) ** 2)

def compute_gradients(X, Y, y_hat):
    """
    Partial derivatives of MSE w.r.t. w and b:
      dw = mean( 2 * x * (y_hat - y) )
      db = mean( 2 * (y_hat - y) )
    """
    dw = np.mean(2 * X * (y_hat - Y))
    db = np.mean(2 * (y_hat - Y))
    return dw, db

# ── GRADIENT DESCENT LOOP ─────────────────────────────────────────────────
loss_history = []

print(f"{'Iteration':>10}  {'Loss':>14}  {'w':>10}  {'b':>10}")
print("-" * 52)

for i in range(1, iterations + 1):
    y_hat        = predict(w, b, X_norm)
    loss         = mse_loss(y_hat, Y)
    dw, db       = compute_gradients(X_norm, Y, y_hat)

    # weight update
    w -= learning_rate * dw
    b -= learning_rate * db

    loss_history.append(loss)

    if i % 100 == 0:
        print(f"{i:>10}  {loss:>14.4f}  {w:>10.4f}  {b:>10.4f}")

print("-" * 52)
print(f"\nFinal  w = {w:.6f}   b = {b:.6f}")
print(f"Final MSE loss = {loss_history[-1]:.6f}")

# ── PLOT ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Linear Regression from Scratch  (numpy, no sklearn)", fontsize=13)

# --- scatter + fitted line ---
ax1.scatter(X, Y, color='steelblue', s=100, zorder=5, label='data points')
x_line      = np.linspace(400, 1600, 200)
x_line_norm = (x_line - X_mean) / X_std
ax1.plot(x_line, predict(w, b, x_line_norm), color='tomato', linewidth=2.5,
         label=f'fit: ŷ = {w:.3f}·x_norm + {b:.3f}')
ax1.set_xlabel("House Size (sqft)")
ax1.set_ylabel("Price ($k)")
ax1.set_title("Regression Line vs Data")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- loss curve ---
ax2.plot(range(1, iterations + 1), loss_history, color='seagreen', linewidth=1.8)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("MSE Loss")
ax2.set_title("Loss Decreasing over Iterations")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("linear_regression_plot.png", dpi=150)
plt.show()
print("\nPlot saved -> linear_regression_plot.png")