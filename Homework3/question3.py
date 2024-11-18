import numpy as np
import random
from math import exp, sqrt
from scipy.special import erf
import matplotlib.pyplot as plt
import pandas as pd

# Mechanisms
# Theorem 5.7
def gaussian_mechanism1(v, sensitivity, delta, epsilon):
    # Calculate the variance for the Gaussian noise
    variance = (32 * (sensitivity ** 2) * np.log(2 / delta)) / (9 * (epsilon ** 2))

    # Sample noise from a Gaussian distribution with mean 0 and the computed variance
    noise = np.random.normal(0, np.sqrt(variance))

    # Add noise to the output of f(D)
    noisy_output = v + noise

    return noisy_output

# Theorem 5.9
def gaussian_mechanism2(v, sensitivity, delta, epsilon):
    # Calculate the variance for the Gaussian noise
    variance = (2 * (sensitivity ** 2) * np.log(1.25 / delta)) / (epsilon ** 2)

    # Sample noise from a Gaussian distribution with mean 0 and the computed variance
    noise = np.random.normal(0, np.sqrt(variance))

    # Add noise to the output of f(D)
    noisy_output = v + noise

    return noisy_output

def mechanism_518(epsilon, delta, GS, tol=1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)

    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def caseA(epsilon, s):
        return Phi(sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def caseB(epsilon, s):
        return Phi(-sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if delta == delta_thr:
        alpha = 1.0
    else:
        if delta > delta_thr:
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)
        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha * GS / sqrt(2.0 * epsilon)

    # Ensure sigma is non-negative and valid
    if sigma <= 0 or np.isnan(sigma):
        return None

    return sigma

# Theorem 5.18
def gaussian_mechanism3(v, sensitivity, delta, epsilon):
    # Calculate sigma using the mechanism_518 function
    sigma = mechanism_518(epsilon, delta, sensitivity)

    # If sigma is None, return the original value to avoid NoneType errors
    if sigma is None:
        return v

    # Sample noise from a Gaussian distribution with mean 0 and the computed sigma
    noise = np.random.normal(0, sigma)

    # Add noise to the output of f(D)
    noisy_output = v + noise

    return noisy_output

# Experiment Code

def run_experiment():
    # Define parameter values
    n_values = [10, 50, 100, 500, 1000]
    d_values = [1, 5, 10, 50, 100]
    epsilon_values = [0.05, 0.1, 0.5, 1, 2, 5]
    delta_values = [1e-6, 1e-5, 1e-4]
    
    results = []
    
    # Run experiments for different parameter combinations
    for n in n_values:
        for d in d_values:
            for epsilon in epsilon_values:
                for delta in delta_values:
                    # Generate 10 datasets
                    datasets = [np.random.randint(0, 2, (n, d)) for _ in range(10)]
                    
                    mse_results = {"mechanism1": [], "mechanism2": [], "mechanism3": []}
                    sd_results = {"mechanism1": [], "mechanism2": [], "mechanism3": []}
                    
                    for dataset in datasets:
                        # Calculate the true value of f(D)
                        f_D = np.mean(dataset)
                        
                        # Run 100 experiments for each mechanism
                        noisy_outputs1 = [gaussian_mechanism1(f_D, 1.0, delta, epsilon) for _ in range(100)]
                        noisy_outputs2 = [gaussian_mechanism2(f_D, 1.0, delta, epsilon) for _ in range(100)]
                        noisy_outputs3 = [gaussian_mechanism3(f_D, 1.0, delta, epsilon) for _ in range(100)]
                        
                        # Filter out None values to avoid errors in calculation
                        noisy_outputs3 = [output for output in noisy_outputs3 if output is not None]
                        
                        # Calculate MSE for each mechanism
                        mse_results["mechanism1"].append(np.mean((np.array(noisy_outputs1) - f_D) ** 2))
                        mse_results["mechanism2"].append(np.mean((np.array(noisy_outputs2) - f_D) ** 2))
                        if noisy_outputs3:
                            mse_results["mechanism3"].append(np.mean((np.array(noisy_outputs3) - f_D) ** 2))
                        
                        # Calculate SD for each mechanism
                        sd_results["mechanism1"].append(np.std(noisy_outputs1))
                        sd_results["mechanism2"].append(np.std(noisy_outputs2))
                        if noisy_outputs3:
                            sd_results["mechanism3"].append(np.std(noisy_outputs3))
                    
                    # Store the average MSE and SD for each mechanism
                    results.append({
                        "n": n,
                        "d": d,
                        "epsilon": epsilon,
                        "delta": delta,
                        "mechanism1_mse": np.mean(mse_results["mechanism1"]),
                        "mechanism2_mse": np.mean(mse_results["mechanism2"]),
                        "mechanism3_mse": np.mean(mse_results["mechanism3"]) if mse_results["mechanism3"] else None,
                        "mechanism1_sd": np.mean(sd_results["mechanism1"]),
                        "mechanism2_sd": np.mean(sd_results["mechanism2"]),
                        "mechanism3_sd": np.mean(sd_results["mechanism3"]) if sd_results["mechanism3"] else None,
                    })
    
    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results to a CSV file
    df_results.to_csv('experiment_results.csv', index=False)
    
    # Plot results
    plot_results(df_results)

def plot_results(df):
    mechanisms = ["mechanism1", "mechanism2", "mechanism3"]
    
    # 1) MSE and SD for different n
    n_values = [10, 100, 1000]
    epsilon = 0.1
    delta = 1e-5
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"MSE and SD for different n with varying d (epsilon={epsilon}, delta={delta})")
    for metric, ax in zip(["mse", "sd"], axes):
        for n in n_values:
            subset = df[(df["n"] == n) & (df["epsilon"] == epsilon) & (df["delta"] == delta)]
            for mechanism in mechanisms:
                ax.plot(subset["d"], subset[f"{mechanism}_{metric}"], label=f"{mechanism} (n={n})", marker='o')
        ax.set_xlabel("d (Dimensionality)")
        ax.set_ylabel(f"{'Mean Squared Error (MSE)' if metric == 'mse' else 'Standard Deviation (SD)'}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig(f"mse_sd_vs_d_for_n.png")
    plt.show()
    
    # 2) MSE and SD for different epsilon
    epsilon_values = [0.05, 0.5, 5]
    d = 10
    delta = 1e-5
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"MSE and SD for different epsilon with varying n (d={d}, delta={delta})")
    for metric, ax in zip(["mse", "sd"], axes):
        for epsilon in epsilon_values:
            subset = df[(df["d"] == d) & (df["epsilon"] == epsilon) & (df["delta"] == delta)]
            for mechanism in mechanisms:
                ax.plot(subset["n"], subset[f"{mechanism}_{metric}"], label=f"{mechanism} (epsilon={epsilon})", marker='o')
        ax.set_xlabel("n (Dataset Size)")
        ax.set_ylabel(f"{'Mean Squared Error (MSE)' if metric == 'mse' else 'Standard Deviation (SD)'}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig(f"mse_sd_vs_n_for_epsilon.png")
    plt.show()
    
    # 3) MSE and SD for different d
    d_values = [1, 10, 100]
    epsilon = 0.1
    delta = 1e-5
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"MSE and SD for different d with varying n (epsilon={epsilon}, delta={delta})")
    for metric, ax in zip(["mse", "sd"], axes):
        for d in d_values:
            subset = df[(df["d"] == d) & (df["epsilon"] == epsilon) & (df["delta"] == delta)]
            for mechanism in mechanisms:
                ax.plot(subset["n"], subset[f"{mechanism}_{metric}"], label=f"{mechanism} (d={d})", marker='o')
        ax.set_xlabel("n (Dataset Size)")
        ax.set_ylabel(f"{'Mean Squared Error (MSE)' if metric == 'mse' else 'Standard Deviation (SD)'}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig(f"mse_sd_vs_n_for_d.png")
    plt.show()
    
    # 4) MSE and SD for different delta
    delta_values = [1e-6, 1e-5, 1e-4]
    epsilon = 0.1
    d = 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"MSE and SD for different delta with varying n (epsilon={epsilon}, d={d})")
    for metric, ax in zip(["mse", "sd"], axes):
        for delta in delta_values:
            subset = df[(df["d"] == d) & (df["epsilon"] == epsilon) & (df["delta"] == delta)]
            for mechanism in mechanisms:
                ax.plot(subset["n"], subset[f"{mechanism}_{metric}"], label=f"{mechanism} (delta={delta})", marker='o')
        ax.set_xlabel("n (Dataset Size)")
        ax.set_ylabel(f"{'Mean Squared Error (MSE)' if metric == 'mse' else 'Standard Deviation (SD)'}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig(f"mse_sd_vs_n_for_delta.png")
    plt.show()

# Run the experiment
run_experiment()
