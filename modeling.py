# %%
# %%
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# -------------------------------
# Define the Goldstein–Price Function
# -------------------------------


def goldstein_price(X):
    """
    Compute the Goldstein–Price function.
    Input:
        X: numpy array of shape (n_samples, 2)
    Output:
        y: numpy array of shape (n_samples,)
    """
    x1, x2 = X[:, 0], X[:, 1]
    term1 = 1 + (x1 + x2 + 1) ** 2 * (
        19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
    )
    term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
        18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
    )
    return term1 * term2


# -------------------------------
# Generate Training Data Using Sobol Sequence
# -------------------------------

# Generate 32 training points using a Sobol sequence
sobol = qmc.Sobol(d=2, scramble=True)
X_train = sobol.random_base2(m=5) * 4 - 2  # Scale to [-2, 2] x [-2, 2]
y_train = goldstein_price(X_train)

# %%
# ONLY DO THIS IF YOU REPEAT USING LOG
y_train = np.log(y_train)

# %%

# -------------------------------
# Center the Data (Constant Mean)
# -------------------------------

# Estimate the constant mean
y_mean = np.mean(y_train)

# Center the data
y_centered = y_train - y_mean

# -------------------------------
# Define the Kernel
# -------------------------------

# Kernel: ConstantKernel * RBF (squared exponential)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# -------------------------------
# Create and Fit the GP Model
# -------------------------------

# Initialize GaussianProcessRegressor
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,  # Fixed noise variance (0.001^2)
    optimizer="fmin_l_bfgs_b",
    n_restarts_optimizer=10,  # Restart optimization to avoid local minima
    normalize_y=False,  # We've already centered y
)

# Fit the GP model (maximizes the marginal likelihood)
gp.fit(X_train, y_centered)

# Add back the mean to get predictions
gp_mean = y_mean
print("Optimized Kernel Parameters:")
print(gp.kernel_)

# Extract individual hyperparameters
constant_value = gp.kernel_.k1.constant_value  # Output scale (variance)
length_scale = gp.kernel_.k2.length_scale  # Length scale
print(f"Learned Output Scale (Variance): {constant_value:.3f}")
print(f"Learned Length Scale: {length_scale:.3f}")

# -------------------------------
# Generate Test Data and Predict
# -------------------------------

# Create a grid for testing
grid_size = 50
x1 = np.linspace(-2, 2, grid_size)
x2 = np.linspace(-2, 2, grid_size)
X1, X2 = np.meshgrid(x1, x2)
X_test = np.vstack([X1.ravel(), X2.ravel()]).T

# Predict the posterior mean and standard deviation
Y_pred, Y_std = gp.predict(X_test, return_std=True)

# Reshape predictions for plotting
Y_pred = Y_pred.reshape(grid_size, grid_size)
Y_std = Y_std.reshape(grid_size, grid_size)

# Compute the true function values
y_true = goldstein_price(X_test).reshape(grid_size, grid_size)

# -------------------------------
# Visualize the Results
# -------------------------------

# Calculate the mean squared error
mse = mean_squared_error(y_true.ravel(), Y_pred.ravel())
print(f"Mean Squared Error between predicted and true values: {mse:.4f}")

# Plot heatmaps for comparison
plt.figure(figsize=(14, 6))

# Heatmap of the GP Posterior Mean
plt.subplot(1, 2, 1)
plt.imshow(Y_pred, extent=[-2, 2, -2, 2], origin="lower", cmap="viridis", aspect="auto")
plt.colorbar(label="Predicted Mean")
plt.title("Gaussian Process Posterior Mean")
plt.xlabel("X1")
plt.ylabel("X2")

# Heatmap of the True Function
plt.subplot(1, 2, 2)
plt.imshow(y_true, extent=[-2, 2, -2, 2], origin="lower", cmap="viridis", aspect="auto")
plt.colorbar(label="True Function Value")
plt.title("True Goldstein–Price Function")
plt.xlabel("X1")
plt.ylabel("X2")

plt.tight_layout()
plt.show()

# Calculate and plot the difference (error)
plt.figure(figsize=(7, 6))
error = y_true - Y_pred
plt.imshow(error, extent=[-2, 2, -2, 2], origin="lower", cmap="coolwarm", aspect="auto")
plt.colorbar(label="Prediction Error (True - Predicted)")
plt.title("Prediction Error Heatmap")
plt.xlabel("X1")
plt.ylabel("X2")
plt.tight_layout()
plt.show()

# %%
# Plot the Posterior Standard Deviation Heatmap
plt.figure(figsize=(7, 6))
plt.imshow(Y_std, extent=[-2, 2, -2, 2], origin="lower", cmap="viridis", aspect="auto")
plt.colorbar(label="Posterior Standard Deviation")
plt.scatter(
    X_train[:, 0], X_train[:, 1], c="red", edgecolor="k", label="Training Points"
)
plt.title("Gaussian Process Posterior Standard Deviation")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()
plt.show()

# Analyze Posterior Standard Deviation at Training Points
std_at_training = gp.predict(X_train, return_std=True)[1]
print("Posterior Standard Deviation at Training Points:")
print(std_at_training)

# Check if standard deviation is close to zero
is_close_to_zero = np.all(std_at_training < 1e-3)
if is_close_to_zero:
    print("Standard deviation drops near zero at all training points, as expected.")
else:
    print(
        "Standard deviation is not close to zero at all training points. Investigate potential issues!"
    )


# %%
from scipy.stats import norm
import seaborn as sns

# Calculate residuals
residuals = y_true.ravel() - Y_pred.ravel()

# Calculate z-scores
z_scores = residuals / Y_std.ravel()

# Plot KDE of z-scores
plt.figure(figsize=(10, 6))
sns.kdeplot(z_scores, fill=True, color="blue", label="KDE of Z-Scores")
x = np.linspace(-4, 4, 500)
plt.plot(x, norm.pdf(x), color="red", linestyle="--", label="Standard Normal PDF")
plt.title("Kernel Density Estimate of Z-Scores of Residuals")
plt.xlabel("Z-Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# Analyze if the z-scores are approximately standard normal
mean_z = np.mean(z_scores)
std_z = np.std(z_scores)
print(f"Mean of Z-Scores: {mean_z:.3f}")
print(f"Standard Deviation of Z-Scores: {std_z:.3f}")

if np.isclose(mean_z, 0, atol=0.1) and np.isclose(std_z, 1, atol=0.1):
    print("The z-scores are approximately standard normal. The GP is well-calibrated.")
else:
    print(
        "The z-scores deviate from standard normal. Investigate potential calibration issues."
    )

# %%
# Number of data points
n = X_train.shape[0]

# Number of parameters (kernel hyperparameters)
k = len(gp.kernel_.theta)  # Hyperparameters of the kernel

# Log-marginal likelihood
log_likelihood = gp.log_marginal_likelihood_value_

# Compute BIC
bic = k * np.log(n) - 2 * log_likelihood

# Display results
print(f"Number of Data Points (n): {n}")
print(f"Number of Model Parameters (k): {k}")
print(f"Log-Marginal Likelihood: {log_likelihood:.3f}")
print(f"BIC Score: {bic:.3f}")

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ConstantKernel as C,
    WhiteKernel,
)
import numpy as np

# Define kernel grammar: simple and composite kernels
base_kernels = [
    RBF(length_scale=1.0),
    Matern(length_scale=1.0, nu=1.5),
    RationalQuadratic(length_scale=1.0, alpha=0.1),
]
composite_kernels = (
    [C(1.0) * k for k in base_kernels]
    + [C(1.0) * (k1 + k2) for k1 in base_kernels for k2 in base_kernels if k1 != k2]
    + [C(1.0) * (k1 * k2) for k1 in base_kernels for k2 in base_kernels]
)

# Track the best model
best_bic = float("inf")
best_model = None
best_kernel = None

# Exhaustive search over all kernels
for kernel in composite_kernels:
    print(f"Evaluating kernel: {kernel}")

    # Initialize and fit the GP model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False)
    gp.fit(X_train, y_train)

    # Compute BIC
    n = X_train.shape[0]
    k = len(gp.kernel_.theta)  # Number of hyperparameters
    log_likelihood = gp.log_marginal_likelihood_value_
    bic = k * np.log(n) - 2 * log_likelihood

    # Print current model and BIC
    print(f"Kernel: {kernel}, BIC: {bic:.3f}")

    # Update the best model if the current one is better
    if bic < best_bic:
        best_bic = bic
        best_model = gp
        best_kernel = kernel

# Print the best model
print("\nBest Model Found:")
print(f"Kernel: {best_kernel}")
print(f"BIC: {best_bic:.3f}")

# %%
import pandas as pd

svm_data = pd.read_csv("/Users/ericjia/Downloads/svm.csv")
lda_data = pd.read_csv("/Users/ericjia/Downloads/lda.csv")

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ConstantKernel as C,
    WhiteKernel,
)

# Extract hyperparameters (columns 1-3) and target (column 4)
svm_X = svm_data.iloc[:, :3].values
svm_y = svm_data.iloc[:, 3].values

lda_X = lda_data.iloc[:, :3].values
lda_y = lda_data.iloc[:, 3].values

# Randomly sample 32 observations from each dataset
np.random.seed(42)  # For reproducibility
svm_sample_idx = np.random.choice(len(svm_X), 32, replace=False)
lda_sample_idx = np.random.choice(len(lda_X), 32, replace=False)

svm_X_sample = svm_X[svm_sample_idx]
svm_y_sample = svm_y[svm_sample_idx]

lda_X_sample = lda_X[lda_sample_idx]
lda_y_sample = lda_y[lda_sample_idx]

# Kernel grammar: simple and composite kernels
base_kernels = [
    RBF(length_scale=1.0),
    Matern(length_scale=1.0, nu=1.5),
    RationalQuadratic(length_scale=1.0, alpha=0.1),
]
composite_kernels = (
    [C(1.0) * k for k in base_kernels]
    + [C(1.0) * (k1 + k2) for k1 in base_kernels for k2 in base_kernels if k1 != k2]
    + [C(1.0) * (k1 * k2) for k1 in base_kernels for k2 in base_kernels]
)


# Function to perform exhaustive GP search and return the best model
def exhaustive_gp_search(X, y):
    best_bic = float("inf")
    best_model = None
    best_kernel = None

    # for kernel in composite_kernels:
    #     gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False)
    #     gp.fit(X, y)

    #     # Compute BIC
    #     n = X.shape[0]
    #     k = len(gp.kernel_.theta)  # Number of hyperparameters
    #     log_likelihood = gp.log_marginal_likelihood_value_
    #     bic = k * np.log(n) - 2 * log_likelihood

    #     # Track the best model
    #     if bic < best_bic:
    #         best_bic = bic
    #         best_model = gp
    #         best_kernel = kernel

    # return best_model, best_kernel, best_bic
    # Exhaustive search over all kernels
    for kernel in composite_kernels:
        print(f"Evaluating kernel: {kernel}")

        # Initialize and fit the GP model
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False)
        gp.fit(X_train, y_train)

        # Compute BIC
        n = X_train.shape[0]  # Number of data points
        k = len(gp.kernel_.bounds)  # Number of free parameters
        log_likelihood = gp.log_marginal_likelihood_value_
        bic = k * np.log(n) - 2 * log_likelihood  # Correct BIC formula

        # Print current model and BIC
        print(f"Kernel: {kernel}, BIC: {bic:.3f}")

        # Update the best model if the current one is better
        if bic < best_bic:
            best_bic = bic
            best_model = gp
            best_kernel = kernel

    return best_model, best_kernel, best_bic


# Perform search for both datasets
svm_best_model, svm_best_kernel, svm_best_bic = exhaustive_gp_search(
    svm_X_sample, svm_y_sample
)
lda_best_model, lda_best_kernel, lda_best_bic = exhaustive_gp_search(
    lda_X_sample, lda_y_sample
)

# Output the results
svm_best_kernel, svm_best_bic, lda_best_kernel, lda_best_bic

# %%

# FOR GOLDSTEIN PRICE

# Define and Fit the Best GP Model
kernel = C(1.0) * Matern(length_scale=1, nu=1.5) + RBF(length_scale=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False)
gp.fit(X_train, y_train)


# Expected Improvement (EI) Function
def expected_improvement(X_candidates, gp, y_train):
    mu, sigma = gp.predict(X_candidates, return_std=True)

    # Current best observed value
    f_best = np.min(y_train)

    # Compute standardized improvement
    with np.errstate(divide="ignore", invalid="ignore"):  # Suppress warnings
        gamma = (f_best - mu) / sigma
        gamma[sigma == 0] = 0  # Handle division by zero

    # Compute EI
    ei = np.where(sigma > 0, sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma)), 0.0)
    return ei


# Candidate Points (Grid)
grid_size = 50
x1 = np.linspace(-2, 2, grid_size)
x2 = np.linspace(-2, 2, grid_size)
X1, X2 = np.meshgrid(x1, x2)
X_candidates = np.vstack([X1.ravel(), X2.ravel()]).T

# Compute Posterior Predictions and EI
mu, sigma = gp.predict(X_candidates, return_std=True)
ei_values = expected_improvement(X_candidates, gp, y_train)

# Reshape for Heatmaps
Posterior_Mean = mu.reshape(grid_size, grid_size)
Posterior_StdDev = sigma.reshape(grid_size, grid_size)
EI = ei_values.reshape(grid_size, grid_size)

# Identify the point with the highest EI
ei_max_idx = np.unravel_index(np.argmax(EI), EI.shape)
x_max_ei = X1[ei_max_idx], X2[ei_max_idx]

# Plot Posterior Mean Heatmap
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Posterior_Mean, levels=50, cmap="viridis")
plt.colorbar(label="Posterior Mean")
plt.scatter(X_train[:, 0], X_train[:, 1], c="red", label="Training Points")
plt.title("Posterior Mean Heatmap")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()

# Plot Posterior Standard Deviation Heatmap
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Posterior_StdDev, levels=50, cmap="viridis")
plt.colorbar(label="Posterior StdDev")
plt.scatter(X_train[:, 0], X_train[:, 1], c="red", label="Training Points")
plt.title("Posterior Standard Deviation Heatmap")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()

# Plot EI Heatmap
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, EI, levels=50, cmap="viridis")
plt.colorbar(label="Expected Improvement (EI)")
plt.scatter(X_train[:, 0], X_train[:, 1], c="red", label="Training Points")
plt.scatter(*x_max_ei, c="blue", marker="x", s=100, label="Max EI Point")
plt.title("Expected Improvement Heatmap")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()

# Output the coordinates of the maximum EI point
print(f"Point with Maximum EI: {x_max_ei}")
# %%
# Import libraries and use preprocessed data from earlier
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    RationalQuadratic,
    Matern,
    ConstantKernel as C,
)
from scipy.stats import norm

# Reuse svm_X, svm_y, lda_X, lda_y, and goldstein_price from the earlier steps


# Expected Improvement (EI) Function (if not already defined)
def expected_improvement(X_candidates, gp, y_train):
    mu, sigma = gp.predict(X_candidates, return_std=True)

    # Current best observed value
    f_best = np.min(y_train)

    # Compute standardized improvement
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = (f_best - mu) / sigma
        gamma[sigma == 0] = 0  # Handle division by zero

    # Compute EI
    ei = np.where(sigma > 0, sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma)), 0.0)
    return ei


# Bayesian Optimization Experiment
def bayesian_optimization(
    function, kernel, initial_data, X_candidates, num_iterations=30
):
    """
    Perform Bayesian Optimization with Expected Improvement.

    Args:
        function: Callable
            The objective function to minimize.
        kernel: Kernel
            The kernel for the Gaussian Process model.
        initial_data: Tuple
            Initial dataset (X_train, y_train).
        X_candidates: ndarray
            Candidate points for optimization.
        num_iterations: int
            Number of iterations for optimization.

    Returns:
        X_final, y_final: Final dataset after optimization.
    """
    X_train, y_train = initial_data
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False)

    for _ in range(num_iterations):
        # Fit the GP model
        gp.fit(X_train, y_train)

        # Compute EI for candidates
        ei_values = expected_improvement(X_candidates, gp, y_train)

        # Find the point with the maximum EI
        x_next = X_candidates[np.argmax(ei_values)]
        y_next = function(x_next.reshape(1, -1))

        # Add the new observation to the dataset
        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)

    return X_train, y_train


# -------------------------------
# Goldstein–Price Experiment
# -------------------------------

goldstein_kernel = C(1.0) * Matern(length_scale=1, nu=1.5) + RBF(length_scale=1)
from scipy.stats import qmc

sobol = qmc.Sobol(d=2, scramble=True)
X_gp_candidates = sobol.random_base2(m=10) * 4 - 2  # Dense Sobol set in [-2, 2]^2
X_gp_initial = sobol.random(5) * 4 - 2  # 5 random initial points
y_gp_initial = goldstein_price(X_gp_initial)

X_gp_final, y_gp_final = bayesian_optimization(
    function=goldstein_price,
    kernel=goldstein_kernel,
    initial_data=(X_gp_initial, y_gp_initial),
    X_candidates=X_gp_candidates,
)

# -------------------------------
# SVM Experiment
# -------------------------------

svm_kernel = C(1.0) * RationalQuadratic(alpha=0.1, length_scale=1)
X_svm_candidates = svm_X  # All unlabeled points as candidates
X_svm_initial = svm_X[np.random.choice(len(svm_X), 5, replace=False)]
y_svm_initial = svm_y[np.random.choice(len(svm_y), 5, replace=False)]

X_svm_final, y_svm_final = bayesian_optimization(
    function=lambda x: svm_y[np.argmin(np.linalg.norm(svm_X - x, axis=1))],
    kernel=svm_kernel,
    initial_data=(X_svm_initial, y_svm_initial),
    X_candidates=X_svm_candidates,
)

# -------------------------------
# LDA Experiment
# -------------------------------

lda_kernel = C(1.0) * Matern(length_scale=1, nu=1.5) + RationalQuadratic(
    alpha=0.1, length_scale=1
)
X_lda_candidates = lda_X  # All unlabeled points as candidates
X_lda_initial = lda_X[np.random.choice(len(lda_X), 5, replace=False)]
y_lda_initial = lda_y[np.random.choice(len(lda_y), 5, replace=False)]

X_lda_final, y_lda_final = bayesian_optimization(
    function=lambda x: lda_y[np.argmin(np.linalg.norm(lda_X - x, axis=1))],
    kernel=lda_kernel,
    initial_data=(X_lda_initial, y_lda_initial),
    X_candidates=X_lda_candidates,
)

# -------------------------------
# Final Outputs
# -------------------------------
print("Goldstein–Price Final Dataset:")
print(X_gp_final)
print(y_gp_final)

print("\nSVM Final Dataset:")
print(X_svm_final)
print(y_svm_final)

print("\nLDA Final Dataset:")
print(X_lda_final)
print(y_lda_final)


# %%
def calculate_gap(y_initial, y_final, f_maximum, is_minimization=True):
    """
    Calculate the 'gap' score for optimization performance.

    Args:
        y_initial: ndarray
            Observed values for the initial dataset.
        y_final: ndarray
            Observed values for the final dataset (after optimization).
        f_maximum: float
            The known maximum (or minimum for minimization) of the objective function.
        is_minimization: bool
            Whether the problem is a minimization problem (default: True).

    Returns:
        gap: float
            The gap score (normalized between 0 and 1).
    """
    # Best value from the initial observations
    f_best_initial = np.min(y_initial) if is_minimization else np.max(y_initial)

    # Best value found during the optimization
    f_best_found = np.min(y_final) if is_minimization else np.max(y_final)

    # Normalize the gap
    if is_minimization:
        gap = (f_best_initial - f_best_found) / (f_best_initial - f_maximum)
    else:
        gap = (f_best_found - f_best_initial) / (f_maximum - f_best_initial)

    return gap


# -------------------------------
# Evaluate Gap for Each Dataset
# -------------------------------

# Known maximum/minimum values for each function
f_max_gp = (
    3  # Example: Replace with the known theoretical maximum/minimum for Goldstein–Price
)
f_max_svm = np.min(svm_y)  # Theoretical minimum (smallest value in dataset)
f_max_lda = np.min(lda_y)  # Theoretical minimum (smallest value in dataset)

# Goldstein–Price
gap_gp = calculate_gap(y_gp_initial, y_gp_final, f_max_gp, is_minimization=True)

# SVM
gap_svm = calculate_gap(y_svm_initial, y_svm_final, f_max_svm, is_minimization=True)

# LDA
gap_lda = calculate_gap(y_lda_initial, y_lda_final, f_max_lda, is_minimization=True)

# -------------------------------
# Output Gap Scores
# -------------------------------
print(f"Gap for Goldstein–Price: {gap_gp:.4f}")
print(f"Gap for SVM: {gap_svm:.4f}")
print(f"Gap for LDA: {gap_lda:.4f}")


# %%
# Bayesian Optimization for Multiple Runs
def bayesian_optimization_multiple_runs(
    function, kernel, X_candidates, num_runs=20, num_iterations=30
):
    results = []
    for run in range(num_runs):
        # Initialize with 5 random points
        sobol = qmc.Sobol(d=X_candidates.shape[1], scramble=True)
        X_initial = sobol.random(5) * 4 - 2
        y_initial = function(X_initial)

        X_train, y_train = X_initial, y_initial
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False)

        for _ in range(num_iterations):
            # Fit the GP model
            gp.fit(X_train, y_train)

            # Compute EI
            ei_values = expected_improvement(X_candidates, gp, y_train)

            # Select the next point with the highest EI
            x_next = X_candidates[np.argmax(ei_values)]
            y_next = function(x_next.reshape(1, -1))

            # Add the new observation to the dataset
            X_train = np.vstack([X_train, x_next])
            y_train = np.append(y_train, y_next)

        # Store the sequence of observations
        results.append({"X": X_train, "y": y_train})
    return results


# Random Search for Multiple Runs
def random_search_multiple_runs(function, X_candidates, num_runs=20, total_budget=150):
    results = []
    for run in range(num_runs):
        # Initialize with 5 random points
        sobol = qmc.Sobol(d=X_candidates.shape[1], scramble=True)
        X_initial = sobol.random(5) * 4 - 2
        y_initial = function(X_initial)

        X_train, y_train = X_initial, y_initial

        # Randomly sample the remaining points
        random_indices = np.random.choice(
            len(X_candidates), total_budget - 5, replace=False
        )
        X_random = X_candidates[random_indices]
        y_random = function(X_random)

        # Combine the initial and random samples
        X_train = np.vstack([X_train, X_random])
        y_train = np.append(y_train, y_random)

        # Store the sequence of observations
        results.append({"X": X_train, "y": y_train})
    return results


# -------------------------------
# Goldstein–Price Experiment
# -------------------------------
goldstein_kernel = C(1.0) * Matern(length_scale=1, nu=1.5) + RBF(length_scale=1)
sobol = qmc.Sobol(d=2, scramble=True)
X_gp_candidates = sobol.random_base2(m=10) * 4 - 2  # Dense Sobol set in [-2, 2]^2

# Bayesian Optimization
bayesian_results_gp = bayesian_optimization_multiple_runs(
    function=goldstein_price,
    kernel=goldstein_kernel,
    X_candidates=X_gp_candidates,
    num_runs=20,
    num_iterations=30,
)

# Random Search
random_results_gp = random_search_multiple_runs(
    function=goldstein_price,
    X_candidates=X_gp_candidates,
    num_runs=20,
    total_budget=150,
)

# -------------------------------
# SVM Experiment
# -------------------------------
svm_kernel = C(1.0) * RationalQuadratic(alpha=0.1, length_scale=1)
X_svm_candidates = svm_X  # All unlabeled points as candidates

# Bayesian Optimization for SVM
bayesian_results_svm = bayesian_optimization_multiple_runs(
    function=lambda X: np.array(
        [svm_y[np.argmin(np.linalg.norm(svm_X - x, axis=1))] for x in X]
    ),
    kernel=svm_kernel,
    X_candidates=X_svm_candidates,
    num_runs=20,
    num_iterations=30,
)

# Random Search for SVM
random_results_svm = random_search_multiple_runs(
    function=lambda X: np.array(
        [svm_y[np.argmin(np.linalg.norm(svm_X - x, axis=1))] for x in X]
    ),
    X_candidates=X_svm_candidates,
    num_runs=20,
    total_budget=150,
)

# -------------------------------
# LDA Experiment
# -------------------------------
lda_kernel = C(1.0) * Matern(length_scale=1, nu=1.5) + RationalQuadratic(
    alpha=0.1, length_scale=1
)
X_lda_candidates = lda_X  # All unlabeled points as candidates

# Bayesian Optimization
bayesian_results_lda = bayesian_optimization_multiple_runs(
    function=lambda X: np.array(
        [lda_y[np.argmin(np.linalg.norm(lda_X - x, axis=1))] for x in X]
    ),
    kernel=lda_kernel,
    X_candidates=X_lda_candidates,
    num_runs=20,
    num_iterations=30,
)

# Random Search
random_results_lda = random_search_multiple_runs(
    function=lambda X: np.array(
        [lda_y[np.argmin(np.linalg.norm(lda_X - x, axis=1))] for x in X]
    ),
    X_candidates=X_lda_candidates,
    num_runs=20,
    total_budget=150,
)

# -------------------------------
# Output Results
# -------------------------------
print(
    f"Bayesian Optimization Results (Goldstein–Price): {len(bayesian_results_gp)} runs completed."
)
print(
    f"Random Search Results (Goldstein–Price): {len(random_results_gp)} runs completed."
)
print(
    f"Bayesian Optimization Results (SVM): {len(bayesian_results_svm)} runs completed."
)
print(f"Random Search Results (SVM): {len(random_results_svm)} runs completed.")
print(
    f"Bayesian Optimization Results (LDA): {len(bayesian_results_lda)} runs completed."
)
print(f"Random Search Results (LDA): {len(random_results_lda)} runs completed.")


# %%
import matplotlib.pyplot as plt
import numpy as np


# Helper function to calculate the gap
def calculate_gap(y_initial, y_observations, f_maximum, is_minimization=True):
    """
    Calculate the gap at each step of the optimization process.

    Args:
        y_initial: ndarray
            Observed values for the initial dataset.
        y_observations: ndarray
            Observed values for the full sequence of observations.
        f_maximum: float
            The known maximum (or minimum for minimization) of the objective function.
        is_minimization: bool
            Whether the problem is a minimization problem (default: True).

    Returns:
        gaps: list
            The gap values for each observation.
    """
    gaps = []
    f_best_initial = np.min(y_initial) if is_minimization else np.max(y_initial)
    for i in range(1, len(y_observations) + 1):
        f_best_found = (
            np.min(y_observations[:i])
            if is_minimization
            else np.max(y_observations[:i])
        )
        gap = (
            (f_best_initial - f_best_found) / (f_best_initial - f_maximum)
            if is_minimization
            else (f_best_found - f_best_initial) / (f_maximum - f_best_initial)
        )
        gaps.append(gap)
    return gaps


# Function to plot learning curves
def plot_learning_curves(
    dataset_name, bayesian_results, random_results, f_maximum, is_minimization=True
):
    """
    Plot learning curves for Bayesian optimization and random search.

    Args:
        dataset_name: str
            Name of the dataset.
        bayesian_results: list of dict
            Results of Bayesian optimization (list of runs).
        random_results: list of dict
            Results of random search (list of runs).
        f_maximum: float
            The known maximum (or minimum for minimization) of the objective function.
        is_minimization: bool
            Whether the problem is a minimization problem (default: True).

    Returns:
        None
    """
    # Calculate average gaps over runs
    bayesian_gaps = []
    random_gaps = []

    for run in bayesian_results:
        gaps = calculate_gap(run["y"][:5], run["y"], f_maximum, is_minimization)
        bayesian_gaps.append(gaps[:30])  # Use first 30 observations only
    for run in random_results:
        gaps = calculate_gap(run["y"][:5], run["y"][:30], f_maximum, is_minimization)
        random_gaps.append(gaps[:30])  # Use first 30 observations only

    bayesian_gaps_mean = np.mean(bayesian_gaps, axis=0)
    random_gaps_mean = np.mean(random_gaps, axis=0)

    # Plot the learning curves
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, 31), bayesian_gaps_mean, label="Bayesian Optimization", marker="o"
    )
    plt.plot(range(1, 31), random_gaps_mean, label="Random Search", marker="x")
    plt.title(f"Learning Curves - {dataset_name}")
    plt.xlabel("Number of Observations")
    plt.ylabel("Average Gap")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------
# Plot Learning Curves for Each Dataset
# -------------------------------

# Goldstein–Price
f_max_gp = 3  # Replace with the known theoretical minimum for Goldstein–Price
plot_learning_curves(
    "Goldstein–Price",
    bayesian_results_gp,
    random_results_gp,
    f_max_gp,
    is_minimization=True,
)

# SVM
f_max_svm = np.min(svm_y)  # Theoretical minimum for SVM
plot_learning_curves(
    "SVM", bayesian_results_svm, random_results_svm, f_max_svm, is_minimization=True
)

# LDA
f_max_lda = np.min(lda_y)  # Theoretical minimum for LDA
plot_learning_curves(
    "LDA", bayesian_results_lda, random_results_lda, f_max_lda, is_minimization=True
)

# %%
from scipy.stats import ttest_rel


# Helper function to calculate mean gap at specific observation counts
def calculate_mean_gap(results, num_observations, f_maximum, is_minimization=True):
    """
    Calculate the mean gap for a specific number of observations.

    Args:
        results: list of dict
            List of runs, where each run contains observations ("y").
        num_observations: int
            Number of observations to consider.
        f_maximum: float
            The known maximum (or minimum for minimization) of the objective function.
        is_minimization: bool
            Whether the problem is a minimization problem (default: True).

    Returns:
        mean_gap: float
            The mean gap for the specified number of observations.
    """
    gaps = []
    for run in results:
        gap = calculate_gap(
            run["y"][:5], run["y"][:num_observations], f_maximum, is_minimization
        )
        gaps.append(gap[-1])  # Take the gap at the last observation
    return np.mean(gaps), np.array(gaps)


# Perform paired t-test and find when random search matches EI performance
def compare_ei_and_random(
    bayesian_results, random_results, f_maximum, is_minimization=True
):
    """
    Compare EI and random search gaps using paired t-tests and find when the p-value exceeds 0.05.

    Args:
        bayesian_results: list of dict
            Results of Bayesian optimization.
        random_results: list of dict
            Results of random search.
        f_maximum: float
            The known maximum (or minimum for minimization) of the objective function.
        is_minimization: bool
            Whether the problem is a minimization problem (default: True).

    Returns:
        None
    """
    observation_counts = [30, 60, 90, 120, 150]
    print(
        f"{'Observations':<15}{'Mean Gap EI':<15}{'Mean Gap Random':<20}{'p-value':<10}"
    )

    for obs_count in observation_counts:
        # Calculate mean gaps
        mean_gap_ei, gaps_ei = calculate_mean_gap(
            bayesian_results, obs_count, f_maximum, is_minimization
        )
        mean_gap_random, gaps_random = calculate_mean_gap(
            random_results, obs_count, f_maximum, is_minimization
        )

        # Perform paired t-test
        p_value = ttest_rel(gaps_ei, gaps_random).pvalue

        print(
            f"{obs_count:<15}{mean_gap_ei:<15.4f}{mean_gap_random:<20.4f}{p_value:<10.6f}"
        )

        # Check when random search catches up to EI
        # if p_value > 0.05:
        #     print(f"Random search matches EI performance with {obs_count} observations (p > 0.05).")
        #     break


# -------------------------------
# Compare EI and Random Search
# -------------------------------

print("Goldstein-Price:")
compare_ei_and_random(
    bayesian_results_gp, random_results_gp, f_max_gp, is_minimization=True
)

print("\nSVM:")
compare_ei_and_random(
    bayesian_results_svm, random_results_svm, f_max_svm, is_minimization=True
)

print("\nLDA:")
compare_ei_and_random(
    bayesian_results_lda, random_results_lda, f_max_lda, is_minimization=True
)

# %%
import matplotlib.pyplot as plt
import numpy as np


# Debugging: Helper function to validate data
def debug_gaps(gaps, method, dataset_name):
    print(f"--- Debugging {method} Gaps for {dataset_name} ---")
    print(f"Number of Runs: {len(gaps)}")
    print(f"Gap Values (First Run): {gaps[0][:5] if len(gaps) > 0 else 'N/A'}")
    print(f"Mean Gap: {np.mean(gaps):.4f}")
    print("------------------------------------------------")


# Calculate the gap
def calculate_gap(y_initial, y_observations, f_maximum, is_minimization=True):
    gaps = []
    f_best_initial = np.min(y_initial) if is_minimization else np.max(y_initial)
    for i in range(1, len(y_observations) + 1):
        f_best_found = (
            np.min(y_observations[:i])
            if is_minimization
            else np.max(y_observations[:i])
        )
        gap = (
            (f_best_initial - f_best_found) / (f_best_initial - f_maximum)
            if is_minimization
            else (f_best_found - f_best_initial) / (f_maximum - f_best_initial)
        )
        gaps.append(gap)
    return gaps


# Plot learning curves
def plot_learning_curves(
    dataset_name, bayesian_results, random_results, f_maximum, is_minimization=True
):
    bayesian_gaps = []
    random_gaps = []

    for run in bayesian_results:
        gaps = calculate_gap(run["y"][:5], run["y"], f_maximum, is_minimization)
        bayesian_gaps.append(gaps[:30])  # First 30 observations only
    for run in random_results:
        gaps = calculate_gap(run["y"][:5], run["y"][:30], f_maximum, is_minimization)
        random_gaps.append(gaps[:30])  # First 30 observations only

    # Debugging
    debug_gaps(bayesian_gaps, "Bayesian Optimization", dataset_name)
    debug_gaps(random_gaps, "Random Search", dataset_name)

    bayesian_gaps_mean = np.mean(bayesian_gaps, axis=0)
    random_gaps_mean = np.mean(random_gaps, axis=0)

    # Plot learning curves
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, 31), bayesian_gaps_mean, label="Bayesian Optimization", marker="o"
    )
    plt.plot(range(1, 31), random_gaps_mean, label="Random Search", marker="x")
    plt.title(f"Learning Curves - {dataset_name}")
    plt.xlabel("Number of Observations")
    plt.ylabel("Average Gap")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------
# Plot Learning Curves for Each Dataset
# -------------------------------

# Goldstein-Price
f_max_gp = 3  # Replace with the known theoretical minimum for Goldstein-Price
plot_learning_curves(
    "Goldstein-Price",
    bayesian_results_gp,
    random_results_gp,
    f_max_gp,
    is_minimization=True,
)

# SVM
f_max_svm = np.min(svm_y)  # Theoretical minimum for SVM
plot_learning_curves(
    "SVM", bayesian_results_svm, random_results_svm, f_max_svm, is_minimization=True
)

# LDA
f_max_lda = np.min(lda_y)  # Theoretical minimum for LDA
plot_learning_curves(
    "LDA", bayesian_results_lda, random_results_lda, f_max_lda, is_minimization=True
)

# %%
