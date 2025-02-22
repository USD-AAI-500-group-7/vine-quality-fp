import pandas as pd               # panda is used for data manipulation and alalysis
import matplotlib.pyplot as plt   # matplotlib is used for visualization 
import seaborn as sns             # seaborn is used for statistical data visulization
from sklearn.feature_selection import mutual_info_classif  # this is used to calculate feature importance based on mutual information.

def feature_selection(red_wine, white_wine):

    # Defining features (x) and target variable (y)
    # Here, features (x) are all columns except "quality" which is our target variable.
    x_red = red_wine.drop(columns = ["quality"])  # here, we drop the quality column from the red wine data
    y_red = red_wine["quality"]  # setting quality as the target variable for red wine

    x_white = white_wine.drop(columns = ["quality"])  # here, we dropped the quality column from the white_wine_cleand dataset
    y_white = white_wine["quality"]  # setting quality as the target variable for white wine_cleanded dataset

    # Computing the correlation matrix to understand relationships between features
    correlation_matrix_red = red_wine.corr()      # calculating correlation matrix for red wine dataset
    correlation_matrix_white = white_wine.corr()  # calculating correlation matrix for white wine dataset

    # Plotting the heatmaps to visualize feature correlation for red and white cleaned data set
    plt.figure(figsize = (12, 8))  # setting the figure size for better readability
    sns.heatmap(correlation_matrix_red, annot = True, cmap = "coolwarm", fmt = ".2f") # creating heatmap with values annotated
    plt.title("Feature Correlation Heatmap - Red Wine (cleaned dataset)")
    plt.show()

    plt.figure(figsize = (12, 8))
    sns.heatmap(correlation_matrix_white, annot = True, cmap = "coolwarm", fmt = ".2f") # ceating heatmap for white wine
    plt.title("Feature Correlation Heatmap - White Wine (cleaned dataset)")

    # Identifying feature importance by using mutual information.
    # Mutual information measures how much information a feature contributes to predicting the target variable
    mi_red = mutual_info_classif(x_red, y_red)  # compute mutual information for red wine dataset
    mi_white = mutual_info_classif(x_white, y_white) # compute mutual information for white wine dataset.

    # Converting mutual information results into a DataFrame for better visualization
    mi_red_df = pd.DataFrame({"Feature": x_red.columns, "Importance": mi_red}).sort_values(by = "Importance", ascending = False)
    # here in the above line, we sfoted features by importance for red wine

    mi_white_df = pd.DataFrame({"Feature": x_white.columns, "Importance": mi_white}).sort_values(by = "Importance", ascending = False)
    # sorting features by importance for white wine

    # plotting Mutual Information scores to visualize feature importance for red wine.
    plt.figure(figsize = (10, 6))
    sns.barplot(x = mi_red_df["Importance"], y = mi_red_df["Feature"])  # creatiang a horizontal bar plot
    plt.title("Feature Importance (Mutual Information) - Red Wine")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    # plotting Mutual Information scores to visualize feature importance for white wine.
    plt.figure(figsize = (10, 6))
    sns.barplot(x = mi_white_df["Importance"], y = mi_white_df["Feature"])
    plt.title("Feature Importance (Mutual Information) - White Wine")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    # selecting the top features based on importance
    # we choose a threshold of 0.02 to select only the most relevant features
    selected_features_red = mi_red_df[mi_red_df["Importance"] > 0.02]["Feature"].tolist()  # Selecting important features for red wine
    selected_features_white = mi_white_df[mi_white_df["Importance"] > 0.02]["Feature"].tolist() # selecting important features for white wine

    # Creating a new datasets with only selected features and keeping quality as the target value.
    red_wine_selected_f = red_wine[selected_features_red + ["quality"]]  # keeping selected features for red wine
    white_wine_selected_f = white_wine[selected_features_white + ["quality"]]

    # printing the selected features for each dataset.
    print(f"\nSelected features for Red wine: {selected_features_red}")
    print(f"Selected features for Whtie wine: {selected_features_white}\n")

    return red_wine_selected_f, white_wine_selected_f
