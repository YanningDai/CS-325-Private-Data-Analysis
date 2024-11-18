import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV file
df = pd.read_csv('experiment_results.csv')  # Ensure the file path is correct

# Set Times New Roman font for all plots
plt.rcParams['font.family'] = 'Times New Roman'

# Function to plot both mechanisms on the same plot for comparison, and combine them into subplots
def plot_comparison_subplots(df, metric, fixed_n_values, fixed_epsilon_values):
    """Plots both comparisons (RR and Lap mechanisms for a given metric) as subplots with log scaling."""
    # Increase font size for better readability
    plt.rcParams.update({'font.size': 18})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Adjusted figure size for compact layout

    # Subplot 1: RR and Lap vs epsilon for fixed n
    ax1 = axes[0]
    for n in fixed_n_values:
        subset = df[df['n'] == n]
        ax1.plot(subset['epsilon'], subset[f'RR_{metric}'], label=f"RR {metric} (n={n})", marker='o', markersize=8)
        ax1.plot(subset['epsilon'], subset[f'Lap_{metric}'], label=f"Lap {metric} (n={n})", linestyle='--', marker='o', markersize=8)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Privacy Budget (epsilon)')
    ax1.set_ylabel(f'{metric}')
    ax1.set_title(f'Comparison of {metric} (RR vs Lap) for Fixed n')
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    # Subplot 2: RR and Lap vs n for fixed epsilon
    ax2 = axes[1]
    for epsilon in fixed_epsilon_values:
        subset = df[df['epsilon'] == epsilon]
        ax2.plot(subset['n'], subset[f'RR_{metric}'], label=f"RR {metric} (epsilon={epsilon})", marker='o', markersize=8)
        ax2.plot(subset['n'], subset[f'Lap_{metric}'], label=f"Lap {metric} (epsilon={epsilon})", linestyle='--', marker='o', markersize=8)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size (n)')
    ax2.set_ylabel(f'{metric}')
    ax2.set_title(f'Comparison of {metric} (RR vs Lap) for Fixed epsilon')
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    plt.tight_layout()
    plt.savefig(f'{metric}_comparison_combined_log.png', dpi=300)  # Save with high resolution
    plt.close()

# Allow user to specify which n and epsilon values to use
def select_values_from_list(values, name):
    print(f"Available {name} values: {values}")
    selected_values = input(f"Enter the {name} values you want to use, separated by commas: ")
    return [int(v.strip()) if name == 'n' else float(v.strip()) for v in selected_values.split(',')]

# Use all unique n and epsilon values from the dataframe
all_n_values = sorted(df['n'].unique())
all_epsilon_values = sorted(df['epsilon'].unique())

# Select n and epsilon values
selected_n_values = select_values_from_list(all_n_values, 'n')
selected_epsilon_values = select_values_from_list(all_epsilon_values, 'epsilon')

# Metrics to be plotted (MSE, MAE, STD)
metrics = ['MSE', 'MAE', 'STD']

# Plot and save the combined subplots for each metric with log scale
for metric in metrics:
    plot_comparison_subplots(df, metric, selected_n_values, selected_epsilon_values)
