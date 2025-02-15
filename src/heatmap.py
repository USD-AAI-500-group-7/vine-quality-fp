import seaborn as sns
import matplotlib.pyplot as plt

# Creating a function to plot correlation heatmap
def plot_correlation_heatmap(data, title):
    plt.figure(figsize = (12, 8))
    sns.heatmap(data.corr(), annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
    plt.title(title, fontsize = 17)
    plt.show()

def draw_heatmap(red_wine, white_wine):
    # Plot correlation heatmap for both datasets(Red and White Wine)
    plot_correlation_heatmap(red_wine, "Correlation Heatmap - Red Wine")
    plot_correlation_heatmap(white_wine, "Correlation Heatmap - White Wine")