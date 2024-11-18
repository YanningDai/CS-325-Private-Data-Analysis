import numpy as np
from numpy.random import laplace
import random
import pandas as pd
import matplotlib.pyplot as plt

# Generate a binary dataset
def generate_binary_dataset(n):
    """Generates a binary dataset with n samples."""
    return np.random.choice([0, 1], size=n)

# Compute query function
def compute_query_function(D):
    """Computes the query function f(D) = (1 / n) * sum(x_i)."""
    return (1 / len(D)) * sum(D)

# Implement Randomized Response Mechanism
def randomized_response(x, epsilon):
    """Applies the Randomized Response mechanism to a binary input."""
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    if random.random() < p:
        return x
    else:
        return 1 - x
    
# Implement Laplacian Mechanism
def laplacian_mechanism(f_D, sensitivity, epsilon):
    """Applies the Laplacian mechanism to a query result."""
    noise = laplace(scale=sensitivity/epsilon)
    return f_D + noise

# Compute utility metrics
def compute_mse(estimated_values, true_value):
    """Computes the Mean Squared Error (MSE)."""
    return np.mean((estimated_values - true_value) ** 2)

def compute_mae(estimated_values, true_value):
    """Computes the Mean Absolute Error (MAE)."""
    return np.mean(np.abs(estimated_values - true_value))

def compute_standard_deviation(estimated_values):
    """Computes the standard deviation of the estimated values."""
    return np.std(estimated_values)

def results_to_dataframe(results):
    data = []
    for (n, epsilon), metrics in results.items():
        rr_mse = metrics["Randomized Response"]["MSE"]
        rr_mae = metrics["Randomized Response"]["MAE"]
        rr_std = metrics["Randomized Response"]["STD"]
        lap_mse = metrics["Laplacian Mechanism"]["MSE"]
        lap_mae = metrics["Laplacian Mechanism"]["MAE"]
        lap_std = metrics["Laplacian Mechanism"]["STD"]
        
        data.append([n, epsilon, rr_mse, rr_mae, rr_std, lap_mse, lap_mae, lap_std])

    df = pd.DataFrame(data, columns=["n", "epsilon", "RR_MSE", "RR_MAE", "RR_STD", "Lap_MSE", "Lap_MAE", "Lap_STD"])
    return df

# Run a single experiment using both DP mechanisms and calculate estimates
def run_single_experiment(D, true_query_value, epsilon, sensitivity):
    """Runs a single experiment on a dataset with both DP mechanisms."""

    # Randomized Response Mechanism
    rr_perturbed_data = [randomized_response(x, epsilon) for x in D]  # Perturbed data (size n)
    rr_estimated_value = compute_query_function(rr_perturbed_data)  # Noisy mean

    # Laplacian Mechanism
    lap_estimated_value = laplacian_mechanism(true_query_value, sensitivity, epsilon)  # Noisy mean

    return rr_estimated_value, lap_estimated_value

# Run multiple experiments on a single dataset, returns arrays of results
def run_experiments_on_dataset(D, true_query_value, epsilon, sensitivity, num_experiments):
    """Runs multiple experiments on a single dataset."""
    rr_estimates = []
    lap_estimates = []

    for _ in range(num_experiments):
        rr_value, lap_value = run_single_experiment(D, true_query_value, epsilon, sensitivity)
        rr_estimates.append(rr_value)
        lap_estimates.append(lap_value)

    return np.array(rr_estimates), np.array(lap_estimates)

# Run experiments for a specific combination of n and epsilon
def run_experiment_for_n_and_epsilon(n, epsilon, num_datasets, num_experiments):
    """Runs experiments for a specific combination of n and epsilon."""
    sensitivity = 1/n  # Sensitivity for Laplacian mechanism

    mse_rr, mae_rr, std_rr = [], [], []
    mse_lap, mae_lap, std_lap = [], [], []

    for _ in range(num_datasets):
        # Generate dataset
        D = generate_binary_dataset(n)
        true_query_value = compute_query_function(D)

        # Run multiple experiments on the dataset
        rr_estimates, lap_estimates = run_experiments_on_dataset(D, true_query_value, epsilon, sensitivity, num_experiments)

        # Compute utility metrics
        mse_rr.append(compute_mse(rr_estimates, true_query_value))
        mae_rr.append(compute_mae(rr_estimates, true_query_value))
        std_rr.append(compute_standard_deviation(rr_estimates))

        mse_lap.append(compute_mse(lap_estimates, true_query_value))
        mae_lap.append(compute_mae(lap_estimates, true_query_value))
        std_lap.append(compute_standard_deviation(lap_estimates))

    # Return utility metrics for each n and epsilon combination
    return {
        "Randomized Response": {
            "MSE": np.mean(mse_rr),
            "MAE": np.mean(mae_rr),
            "STD": np.mean(std_rr),
        },
        "Laplacian Mechanism": {
            "MSE": np.mean(mse_lap),
            "MAE": np.mean(mae_lap),
            "STD": np.mean(std_lap),
        }
    }

# Main experiment loop
def run_full_experiment(n_values, epsilon_values, num_datasets, num_experiments):
    """Runs the full experiment over all n and epsilon values."""
    results = {}

    for n in n_values:
        for epsilon in epsilon_values:
            results[(n, epsilon)] = run_experiment_for_n_and_epsilon(n, epsilon, num_datasets, num_experiments)

    return results

# Values for n and epsilon, number of datasets, and number of experiments
n_values = [10, 50, 100, 500, 1000, 5000]
epsilon_values = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
num_datasets = 20
num_experiments = 100

# Run the full experiment
results = run_full_experiment(n_values, epsilon_values, num_datasets, num_experiments)

# Output the results
for key, value in results.items():
    n, epsilon = key
    print(f"n={n}, epsilon={epsilon}:")
    print(f"  Randomized Response: MSE={value['Randomized Response']['MSE']}, MAE={value['Randomized Response']['MAE']}, STD={value['Randomized Response']['STD']}")
    print(f"  Laplacian Mechanism: MSE={value['Laplacian Mechanism']['MSE']}, MAE={value['Laplacian Mechanism']['MAE']}, STD={value['Laplacian Mechanism']['STD']}")
    print()

df_results = results_to_dataframe(results)
df_results.to_csv("experiment_results.csv", index=False)