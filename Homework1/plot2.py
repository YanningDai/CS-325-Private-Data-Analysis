import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV file (adjust the file path accordingly)
df = pd.read_csv('experiment_results.csv')

# Set font to Times New Roman and font size for the entire plot
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# Metrics and mechanisms
metrics = ['MSE', 'MAE', 'STD']
mechanisms = ['RR', 'Lap']

# Function to plot heatmaps for combined effect of n and epsilon
def plot_heatmaps(df, metric):
    """Plots heatmap for the combined effect of n and epsilon for a given metric and both mechanisms."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Adjusted figure size for compact layout

    for i, mechanism in enumerate(mechanisms):
        pivot_table = df.pivot('n', 'epsilon', f'{mechanism}_{metric}')
        
        sns.heatmap(pivot_table, annot=True, fmt=".2g", cmap='coolwarm', cbar_kws={'label': f'{metric}'},
                    ax=axes[i], annot_kws={"size": 12})  # Set smaller font size for the text inside the cells
        
        axes[i].set_title(f'{mechanism} {metric}')
        axes[i].set_xlabel('Privacy Budget (epsilon)')
        axes[i].set_ylabel('Sample Size (n)')
    
    plt.tight_layout()
    plt.savefig(f'{metric}_heatmap_combined.png', dpi=300)  # Save without showing the plot
    plt.close()  # Close the figure to free up memory

# Plot heatmaps for each metric
for metric in metrics:
    plot_heatmaps(df, metric)
