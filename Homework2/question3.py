import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import laplace

# Generate a random dataset
def generate_dataset(size=100, value_range=(0, 100)):
    return np.random.uniform(value_range[0], value_range[1], size)

# Noisy-Max Mechanism: Add Laplace noise to each element and select the top 3
def noisy_max(X, epsilon):
    noisy_scores = X + laplace.rvs(scale=1/epsilon, size=len(X))
    top3_indices = np.argsort(noisy_scores)[-3:][::-1]  # Get indices of top 3 noisy scores
    return top3_indices

# Exponential Mechanism: Assign probabilities based on the exponential function and select top 3
def exponential_mechanism(X, epsilon):
    # Prevent overflow by normalizing values before calculating exp
    probabilities = np.exp(epsilon * (X - np.max(X)) / 2)  # Normalize to avoid overflow
    probabilities /= np.sum(probabilities)  # Normalize to make it a valid probability distribution
    selected_indices = np.random.choice(len(X), size=3, replace=False, p=probabilities)
    return selected_indices

# Utility function to calculate overlap (how many of the selected top 3 match the true top 3)
def calculate_overlap(selected_indices, true_top3):
    overlap = len(set(selected_indices) & set(true_top3))
    return overlap

# Experiment design
def run_experiments(epsilon_values, num_datasets=20, num_experiments=100):
    results = {epsilon: {'noisy_max': [], 'exponential': []} for epsilon in epsilon_values}
    
    for epsilon in epsilon_values:
        for _ in range(num_datasets):
            # Generate a random dataset
            dataset = generate_dataset()
            
            # Get true top 3 indices based on actual values
            true_top3_indices = np.argsort(dataset)[-3:][::-1]
            
            # Run experiments for each mechanism
            for _ in range(num_experiments):
                # Noisy-Max mechanism
                noisy_max_top3 = noisy_max(dataset, epsilon)
                noisy_max_overlap = calculate_overlap(noisy_max_top3, true_top3_indices)
                results[epsilon]['noisy_max'].append(noisy_max_overlap)
                
                # Exponential mechanism
                exp_mechanism_top3 = exponential_mechanism(dataset, epsilon)
                exp_mechanism_overlap = calculate_overlap(exp_mechanism_top3, true_top3_indices)
                results[epsilon]['exponential'].append(exp_mechanism_overlap)
    
    return results

# Analyze the results: calculate average accuracy and variance
def analyze_results(results):
    epsilons = []
    noisy_max_averages = []
    noisy_max_variances = []
    exp_mechanism_averages = []
    exp_mechanism_variances = []
    
    for epsilon, res in results.items():
        noisy_max_avg = np.mean(res['noisy_max'])
        noisy_max_var = np.var(res['noisy_max'])
        exp_mechanism_avg = np.mean(res['exponential'])
        exp_mechanism_var = np.var(res['exponential'])
        
        epsilons.append(epsilon)
        noisy_max_averages.append(noisy_max_avg)
        noisy_max_variances.append(noisy_max_var)
        exp_mechanism_averages.append(exp_mechanism_avg)
        exp_mechanism_variances.append(exp_mechanism_var)
        
        print(f"Epsilon: {epsilon}")
        print(f"Noisy-Max Mechanism - Avg. Overlap: {noisy_max_avg}, Variance: {noisy_max_var}")
        print(f"Exponential Mechanism - Avg. Overlap: {exp_mechanism_avg}, Variance: {exp_mechanism_var}")
        print("")
    
    return epsilons, noisy_max_averages, noisy_max_variances, exp_mechanism_averages, exp_mechanism_variances

# Function to plot and save results
def plot_and_save_results(epsilons, noisy_max_averages, noisy_max_variances, exp_mechanism_averages, exp_mechanism_variances):
    plt.figure(figsize=(10, 6))
    
    # Plot with transparent error bars
    plt.errorbar(epsilons, noisy_max_averages, yerr=noisy_max_variances, label='Noisy-Max Mechanism', marker='o', capsize=5, alpha=0.7)
    plt.errorbar(epsilons, exp_mechanism_averages, yerr=exp_mechanism_variances, label='Exponential Mechanism', marker='s', capsize=5, alpha=0.7)
    
    plt.title('Mechanism Comparison: Noisy-Max vs Exponential')
    plt.xlabel('Epsilon (Privacy Budget)')
    plt.ylabel('Average Overlap with True Top-3')
    plt.xscale('log')  # Logarithmic scale for epsilon
    plt.legend()
    plt.grid(True)
    
    # Save the plot with the specified name
    plt.savefig('question3.png', dpi=300)
    plt.show()

# Function to save results to CSV
def save_results_to_csv(epsilons, noisy_max_averages, noisy_max_variances, exp_mechanism_averages, exp_mechanism_variances):
    df = pd.DataFrame({
        'Epsilon': epsilons,
        'Noisy-Max Avg': noisy_max_averages,
        'Noisy-Max Variance': noisy_max_variances,
        'Exponential Avg': exp_mechanism_averages,
        'Exponential Variance': exp_mechanism_variances
    })
    
    df.to_csv('question3_mechanism_comparison_results.csv', index=False)
    print("Results saved to question3_mechanism_comparison_results.csv")

# Main execution
if __name__ == "__main__":
    np.random.seed(42)  # Set random seed for reproducibility
    epsilon_values = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    results = run_experiments(epsilon_values, num_datasets=20, num_experiments=100)
    
    # Analyze results
    epsilons, noisy_max_averages, noisy_max_variances, exp_mechanism_averages, exp_mechanism_variances = analyze_results(results)
    
    # Plot and save the results
    plot_and_save_results(epsilons, noisy_max_averages, noisy_max_variances, exp_mechanism_averages, exp_mechanism_variances)
    
    # Save results to CSV
    save_results_to_csv(epsilons, noisy_max_averages, noisy_max_variances, exp_mechanism_averages, exp_mechanism_variances)
