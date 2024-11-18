import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import laplace, norm

# Generate random dataset
def generate_dataset(n, d):
    return np.random.randint(0, 2, size=(n, d))

# Query function: calculate the average of the dataset
def query_function(dataset):
    return np.mean(dataset)

# Laplace mechanism
def laplace_mechanism(query_result, n, epsilon):
    sensitivity = 1 / n
    noise = laplace.rvs(scale=sensitivity / epsilon)
    return query_result + noise

# Gaussian mechanism
def gaussian_mechanism(query_result, n, epsilon, delta):
    sensitivity = 1 / n
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = norm.rvs(scale=sigma)
    return query_result + noise

# Utility analysis (calculates both MSE and SD)
def utility_analysis(true_value, perturbed_values):
    mse = np.mean((perturbed_values - true_value) ** 2)
    std_dev = np.std(perturbed_values)
    return mse, std_dev

# Run experiments for different values of n, d, epsilon
def run_experiments(n_values, d_values, epsilon_values, delta=0.01):
    results = []
    
    for n in n_values:
        for d in d_values:
            for epsilon in epsilon_values:
                mse_laplace_list = []
                mse_gaussian_list = []
                std_laplace_list = []
                std_gaussian_list = []
                
                for _ in range(10):  # 10 datasets
                    dataset = generate_dataset(n, d)
                    true_query = query_function(dataset)
                    
                    perturbed_laplace = []
                    perturbed_gaussian = []
                    
                    for _ in range(100):  # 100 experiments
                        # Apply Laplace Mechanism
                        laplace_result = laplace_mechanism(true_query, n, epsilon)
                        perturbed_laplace.append(laplace_result)
                        
                        # Apply Gaussian Mechanism
                        gaussian_result = gaussian_mechanism(true_query, n, epsilon, delta)
                        perturbed_gaussian.append(gaussian_result)
                    
                    mse_laplace, std_laplace = utility_analysis(true_query, np.array(perturbed_laplace))
                    mse_gaussian, std_gaussian = utility_analysis(true_query, np.array(perturbed_gaussian))
                    
                    mse_laplace_list.append(mse_laplace)
                    mse_gaussian_list.append(mse_gaussian)
                    std_laplace_list.append(std_laplace)
                    std_gaussian_list.append(std_gaussian)
                
                avg_mse_laplace = np.mean(mse_laplace_list)
                avg_mse_gaussian = np.mean(mse_gaussian_list)
                avg_sd_laplace = np.mean(std_laplace_list)
                avg_sd_gaussian = np.mean(std_gaussian_list)
                
                results.append((n, d, epsilon, avg_mse_laplace, avg_mse_gaussian, avg_sd_laplace, avg_sd_gaussian))
    
    return results

# Save results to a single CSV
def save_results_to_csv(results):
    df = pd.DataFrame(results, columns=['n', 'd', 'epsilon', 'Laplace MSE', 'Gaussian MSE', 'Laplace SD', 'Gaussian SD'])
    df.to_csv('question4_mechanism_comparison_results.csv', index=False)
    print("Results saved to 'question4_mechanism_comparison_results.csv'.")

# Plot results for different parameter combinations
def plot_combined_results(results, x_param, y_param, curves_param, fixed_curves_values, x_label, y_label, curves_label, filename):
    x_values = sorted(set([r[x_param] for r in results]))
    y_values = sorted(set([r[y_param] for r in results]))
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Create 1x2 subplot
    
    # Define empty lists to store MSE and SD for Laplace and Gaussian for each curve value
    for curves_value in fixed_curves_values:
        z_mse_laplace = np.zeros((len(y_values), len(x_values)))
        z_mse_gaussian = np.zeros((len(y_values), len(x_values)))
        z_sd_laplace = np.zeros((len(y_values), len(x_values)))
        z_sd_gaussian = np.zeros((len(y_values), len(x_values)))
        
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                # Filter results by x, y, and curve parameter
                filtered_result = [r for r in results if r[x_param] == x and r[y_param] == y and r[curves_param] == curves_value]
                if filtered_result:
                    z_mse_laplace[i, j] = filtered_result[0][3]  # Laplace MSE
                    z_mse_gaussian[i, j] = filtered_result[0][4]  # Gaussian MSE
                    z_sd_laplace[i, j] = filtered_result[0][5]  # Laplace SD
                    z_sd_gaussian[i, j] = filtered_result[0][6]  # Gaussian SD
        
        # Plot MSE (left)
        axes[0].plot(x_values, z_mse_laplace.mean(axis=0), label=f'Laplace ({curves_label}={curves_value})', marker='o')
        axes[0].plot(x_values, z_mse_gaussian.mean(axis=0), label=f'Gaussian ({curves_label}={curves_value})', marker='s')
        axes[0].set_yscale('log')  # Logarithmic y-axis for MSE
        axes[0].set_title(f'MSE for different {curves_label} with varying {x_label} and {y_label}')
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel('MSE')
        axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot SD (right)
        axes[1].plot(x_values, z_sd_laplace.mean(axis=0), label=f'Laplace ({curves_label}={curves_value})', marker='o')
        axes[1].plot(x_values, z_sd_gaussian.mean(axis=0), label=f'Gaussian ({curves_label}={curves_value})', marker='s')
        axes[1].set_yscale('log')  # Logarithmic y-axis for SD
        axes[1].set_title(f'SD for different {curves_label} with varying {x_label} and {y_label}')
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('SD')
        axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    # Group legends by Laplace and Gaussian Mechanisms
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper right')
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc='upper right')


    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300)
    plt.show()
    print(f"Plot saved as {filename}.png")

# Main execution
if __name__ == "__main__":
    n_values = [10, 50, 100, 500, 1000]
    d_values = [1, 5, 10, 50, 100]
    epsilon_values = [0.05, 0.1, 0.5, 1, 2, 5]
    
    results = run_experiments(n_values, d_values, epsilon_values, delta=0.01)
    
    # Save results to CSV
    save_results_to_csv(results)
    
    # Plot 1: x-axis = n, y-axis = d, epsilon fixed at 0.05, 0.5, 5
    plot_combined_results(results, x_param=0, y_param=1, curves_param=2, 
                          fixed_curves_values=[0.05, 0.5, 5], 
                          x_label='n', y_label='d', 
                          curves_label='epsilon', filename='plot_n_vs_d_curves_epsilon')

    # Plot 2: x-axis = n, y-axis = epsilon, d fixed at 1, 10, 100
    plot_combined_results(results, x_param=0, y_param=2, curves_param=1, 
                          fixed_curves_values=[1, 10, 100], 
                          x_label='n', y_label='epsilon', 
                          curves_label='d', filename='plot_n_vs_epsilon_curves_d')

    # Plot 3: x-axis = d, y-axis = epsilon, n fixed at 10, 100, 1000
    plot_combined_results(results, x_param=1, y_param=2, curves_param=0, 
                          fixed_curves_values=[10, 100, 1000], 
                          x_label='d', y_label='epsilon', 
                          curves_label='n', filename='plot_d_vs_epsilon_curves_n')
