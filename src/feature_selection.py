# step 7: Feature selction

# Importing the necessary liberaries
import pandas as pd               # panda is used for data manipulation and alalysis
import matplotlib.pyplot as plt   # matplotlib is used for visualization 
import seaborn as sns             # seaborn is used for statistical data visulization
from sklearn.feature_selection import mutual_info_classif  # this is used to calculate feature importance based on mutual information.
from sklearn.ensemble import RandomForestClassifier  # is has been used for feature selection

# Loading the cleaned dataset
# we put header = 0 to make sure that the first row of the CSV file is used as column names.
red_wine_cleaned = pd.read_csv("D:/AI/semester__1__/AAI_500_Probability_and_Statistics/final_project_code/red_wine_cleaned.csv", sep = r",", header = 0)
white_wine_cleaned = pd.read_csv("D:/AI/semester__1__/AAI_500_Probability_and_Statistics/final_project_code/white_wine_cleaned.csv", sep = r",", header = 0)

# Defining features (x) and target variable (y)
# Here, features (x) are all columns except "quality" which is our target variable.
x_red = red_wine_cleaned.drop(columns = ["quality"])  # here, we drop the quality column from the red wine data
y_red = red_wine_cleaned["quality"]  # setting quality as the target variable for red wine

x_white = white_wine_cleaned.drop(columns = ["quality"])  # here, we dropped the quality column from the white_wine_cleand dataset
y_white = white_wine_cleaned["quality"]  # setting quality as the target variable for white wine_cleanded dataset

# Computing the correlation matrix to understand relationships between features
correlation_matrix_red = red_wine_cleaned.corr()      # calculating correlation matrix for red wine dataset
correlation_matrix_white = white_wine_cleaned.corr()  # calculating correlation matrix for white wine dataset

# Plotting the heatmaps to visualize feature correlation for red and white cleaned data set
plt.figure(figsize = (12, 8))  # setting the figure size for better readability
sns.heatmap(correlation_matrix_red, annot = True, cmap = "coolwarm", fmt = ".2f") # creating heatmap with values annotated
plt.title("Feature Correlation Heatmap - Red Wine (cleaned dataset)")
plt.show

plt.figure(figsize = (12, 8))
sns.heatmap(correlation_matrix_white, annot = True, cmap = "coolwarm", fmt = ".2f") # ceating heatmap for white wine
plt.title("Feature Correlation Heatmap - White Wine (cleaned dataset)")

# Idnetifying feature importance by using mutual information
# mutual information measures how much information a feature contributes to predicting the target variable
mi_red = mutual_info_classif(x_red, y_red)  # compute mutual information for red wine dataset
mi_white = mutual_info_classif(x_white, y_white) # compute mutual information for white wine dataset. 

# Converting mutual information results into a DataFrame for better visualization
mi_red_df = pd.DataFrame({"Feature": x_red.columns, "Importance": mi_red}).sort_values(by = "Importance", ascending = False)
# here in the above line, we sfoted features by importance for red wine

mi_white_df =pd.DataFrame({"Feature": x_white.columns, "Importance": mi_white}).sort_values(by = "Importance", ascending = False)
# sotring features by importance for white wine

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
red_wine_selected_f = red_wine_cleaned[selected_features_red + ["quality"]]  # keeping selected features for red wine
white_wine_selected_f = white_wine_cleaned[selected_features_white + ["quality"]]

# Saving the updated datasets with selected features to csv files for later use. 
red_wine_selected_f.to_csv("D:/AI/semester__1__/AAI_500_Probability_and_Statistics/final_project_code/red_wine_selected_f.csv", index=False)
white_wine_selected_f.to_csv("D:/AI/semester__1__/AAI_500_Probability_and_Statistics/final_project_code/white_wine_selected_f.csv", index=False)


# printing the selected features for each dataset. 
print(f"\nSelected features for Red wine: {selected_features_red}")
print(f"Selected features for Whtie wine: {selected_features_white}\n")
