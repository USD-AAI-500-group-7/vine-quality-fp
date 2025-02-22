import matplotlib.pyplot as plt

# Creating a function to plot histogram for dataset both white and red wine

def plot_feature_distributions(data, title):
    fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize = (15, 12)) # Creating 4x3 grid
    fig.subplots_adjust(hspace = 0.5, wspace = 0.3) # adjusting the spacing

    axes = axes.flatten() # flatten axes array for easy looping

    for i, column in enumerate(data.columns):
        data[column].hist(ax = axes[i], bins = 20, alpha = 0.7, color = "purple", edgecolor='black')
        axes[i].set_title(column)

    plt.suptitle(title, fontsize = 17)
    plt.show()

def draw_histogram(red_wine,  white_wine):
    # Plotting histograms for both datasets(Red and White wine)
    print("\n")
    plot_feature_distributions(red_wine, "Red Wine Feature Distributions")
    print("\n")
    plot_feature_distributions(white_wine, "White Wine Feature Distributions")