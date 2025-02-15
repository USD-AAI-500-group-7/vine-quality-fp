import seaborn as sns
import matplotlib.pyplot as plt

# Creating a function to plot boxplots for each feature
def plot_boxplots(data, title):
    plt.figure(figsize = (12, 8))
    sns.boxplot(data = data)
    plt.xticks(rotation = 45)
    plt.title(title, fontsize = 17)
    plt.show()

def draw_boxplots(red_wine, white_wine):
    # Boxplots for red and white wine datasets
    plot_boxplots(red_wine, "Boxplot of Feature - Red Wine")
    plot_boxplots(white_wine, "Boxplot of Feature - White Wine")