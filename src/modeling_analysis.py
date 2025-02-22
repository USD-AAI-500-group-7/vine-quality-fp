# step8: Statistical Modeling & Analysis

# Importing necessary libraries
import pandas as pd  # fro data manipulation
import numpy as np   # for numerical operations
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns    # for statistical plots
from sklearn.model_selection import train_test_split # splitting data into fitting & testing sets
from sklearn.linear_model import LinearRegression  # liear regression model
from sklearn.metrics import mean_squared_error, r2_score   # RMSE and R-squared for evaluation

file_path = "D:/AI/semester__1__/AAI_500_Probability_and_Statistics/final_project_code/"

# lading the datasets with selected features after Feature selection
file_path = "D:/AI/semester__1__/AAI_500_Probability_and_Statistics/final_project_code/"
red_wine_selected_f =pd.read_csv(file_path + "red_wine_selected_f.csv")
white_wine_selected_f = pd.read_csv(file_path + "white_wine_selected_f.csv")


# defining features (x) and target variable (y) for both datasets
x_red = red_wine_selected_f.drop(columns = ["quality"])   # independent variables for red wine
y_red = red_wine_selected_f["quality"]  # target variable (wine "quality") for white wine

x_white = white_wine_selected_f.drop(columns = ["quality"])  # independent variables for white wine
y_white = white_wine_selected_f["quality"]  # target variable "quality" for white wine


# why do we split the dataset into fitting and testing sets?
# we split the dataset to evaluate the model's performance on unseen data, we are splitting the data into fitting (80%) and testing (20%) 
# sets to evaluate the model's performance. Here we fit the model on one part of the dataset and then we test the another part to evaluate 
# its performance. It will help us to make sure our model doesn't memorize the fitting data, but instead, it learns the patterns that can 
# generalize well to new and unseen data. 
# we have splitted both red and white dataset in order to analyze them separately, it will ensure us that we have evaluated each model independently
x_fit_red, x_test_red, y_fit_red, y_test_red = train_test_split(x_red, y_red, test_size = 0.2, random_state = 42)
x_fit_white, x_test_white, y_fit_white, y_test_white = train_test_split(x_white, y_white, test_size = 0.2, random_state = 42)

# here, train_test_split() divide the data to 80% fitting and 20% testing, fitting part fit the model, and testing part evaluate the model's accuracy
# tes_size = 0.2 means we have assigned 20% of the dataset to testing and the remaining which is 80% will be sued for fitting.
# random_state = 42 makes sure that the data split is reproducible, and if we run the code many times, the fit-test split will be the same every time.


# initializing the linear regression model
red_wine_model = LinearRegression()
white_wine_model = LinearRegression()

# what will happen when we fit the model?
# the fit() function estimates the relationshiop between input features (x) and output (y)
# It finds the best_fitting line that minimizes the error in predicting wine quality.

# fitting the model using the fitting dataset
red_wine_model.fit(x_fit_red, y_fit_red)
white_wine_model.fit(x_fit_white, y_fit_white)



# here, why we have used predict() function?
# Atere we fitted the model, we use predict() function to generate whine quality predictions on unseen data.

#  Predicitng the wine quality on the test dataset
y_prediction_red = red_wine_model.predict(x_test_red)
y_prediction_white = white_wine_model.predict(x_test_white)


# What does RMSE and R-Squared will tell us? and why we use them?
#     RMSE:   Root Mean Squared Error measuares how much the model's predictions deviate from actual values. therefore, 
# a lower RMSE value means the model is making smaller errors, which it tells us the model has a better accuracy.

#     R-Squared:  what does R-squared tell us?
# It tells us how well the model explains the variance in wine quality.
# If R2 is close to 1, the model can explan most of the variation, but close to 0, we say the model doesn't explalin the variation well.

#Evaluating the model performance using RMSE and R-Squared
rmse_red = np.sqrt(mean_squared_error(y_test_red, y_prediction_red))  # RMSE for red wine
r2_red = r2_score(y_test_red, y_prediction_red)  # r-squared score for the red wine

rmse_white = np.sqrt(mean_squared_error(y_test_white, y_prediction_white)) # RMSE for white wine
r2_white = r2_score(y_test_white, y_prediction_white)  # r-squared for white wine.


# printing the resluts of model performance
print(f"\nRed Wine Model Performance:")
print(f"R-Sauared Score: {r2_red: .2f}")  # here, r-saured will tell us how wll the model can explian the variance in quality
print(f"RMSE: {rmse_red: .2f}")  # RMSE will tell us the average error we have in quality prediction

print(f"\nWhite Wine Model Performance: ")
print(f"R-Squared Score: {r2_white: .2f}")
print(f"RMSE: {rmse_red: .2f}")



# Here, in this part our goal from plotting actual with predicted values are:
# A perfect model will have all points on the black dashed line( y = x)
# This plot can help us see how well predictions match the actual wine quality. 

# visualization: plotting actual with predicted values for both models
plt.figure(figsize = (12, 5))

# scatter plot for red wine predictions
plt.subplot(1, 2, 1)
sns.scatterplot(x = y_test_red, y = y_prediction_red, color = "red") # scatter plot of actual vs predicted
plt.plot(y_test_red, y_test_red, color = "black", linestyle = "--")  # perfect prediction line (y = x)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Red Wine: Actual VS Predicted Quality")
plt.show()

# scatter plot for white wine predictions
plt.subplot(1, 2, 2)
sns.scatterplot(x = y_test_white, y = y_prediction_white, color = "blue")
plt.plot(y_test_white, y_test_white, color = "black", linestyle = "--")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("White Wine Actutal VS Predicted Quality")
plt.show()

print("\n")
print(
"""
********************************************   Interpretation of the Model Performance   ********************************************
Red Wine Performance:
    R-Squared Score:  0.40
    RMSE:  0.64

White Wine Performance: 
    R-squared Score: 0.25
    RMSE: 0.64

                                                            R-Squared:                                                         
    R-squared measures how well the model explains the variability in wine quality. 
    Therefore, R-squared 0.40 for red wine explains that 40% of the variation in red wine quality is explained by the model. We can 
    say that this is a moderate performance, and it almost shows that the other factors that affect the wine quality are missing from 
    our model.
    For Withe wine, our R-squared score is 0.25 which means 25% of the variation in white wine quality is explained by the model. and it 
    shows that the model struggles more with white wine in comparing with red wine. 
    Higher R-squared (closer to 1.0) tells us that we have a better model, but as it decreases from 1.0 and became lower, we say the model 
    doesn't capturing enough of the variation.



                                                              RMSE:                                         
    RMSE tell us the average difference between actual and predicted wine quality scores. 
    As we see in the results, both model have an RMSE of 0.64, and it means on average, our predicted wine quality score is off by 0.64 units.
    Lower RMSE is better since it shows the model's predictions are closet to the actual values. 



                                                     Analysis of performance:
    We have moderate performance for Red Wine, but we have lower performance for White Wine

Red Wine:
    The model almost has done a good job in predicting red wine quality. It explains 40% of variation in quality score. but the RMSE shows
    almost moderate errors in predictions but we can say it is reasonable for a dataset like th is. 

White Wine:
    The model here struggles since it only explains 25% of the variation. Because of that, we can say that we don't have some of the important
    variables that affect the white wine quality more, and we don't have them in the dataset. The result for RMSE is the same with red wine. 

Scatter plots:
    The scatter plot of actual vs predicted quality shows the predictions are spread out, and it tells us that the model doesn't perfectly align 
    with actual values. In a perfect model, all points will be on the diagonal black line (y = x), while in our model is not like that and many points 
    deviate from this line and it means the model is not perfect at predicting quality.

Refining the model:
    if we want to improve the model, we can apply data scaling and transformation. We can normalize or standardize the features to improve the model's
    performance, or we can use generalized linear model (GLM).

Conclusion:
    We can say the model is almost useful for predicting the wine quality, especially for red wine, even though it doesn't explain the wine quality
    since we don't have other factors that can affect the quality. They are not in the dataset.  
"""
)