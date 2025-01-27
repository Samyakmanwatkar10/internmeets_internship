**Insurance Dataset Analysis and Modeling**

This Python notebook performs a comprehensive data science workflow on the 'insurance.csv' dataset, which contains data on over 1500 customers. The dataset includes information such as age, sex, region, smoker status, and more. The aim is to explore the data, preprocess it, visualize various aspects, and build predictive models to analyze insurance costs.

1. Exploring the Dataset
In this section, we load the dataset and perform initial exploration:
- Display the first few rows of the dataset.
- Generate summary statistics to understand the distribution of numerical features.
- Check data types and non-null counts of each column.

2. Converting Categorical Values to Numerical
To prepare the dataset for machine learning models, we need to convert categorical values into numerical format:
- Identify categorical columns and apply Label Encoding.
- Create new columns with encoded values and drop the original categorical columns.

3. Plotting Heatmap to See Dependency of Dependent Value on Independent Features
We analyze the correlations between features and the target variable:
- Compute the correlation matrix of the dataset.
- Plot a heatmap to visualize the dependencies and correlations.

4. Data Visualization (Plots of Feature vs Feature)
In this section, we create visualizations to explore relationships between different features:
- Use pair plots to visualize pairwise relationships between features.

5. Plotting Skew and Kurtosis
We analyze the distribution of the features:
- Calculate skewness and kurtosis for each feature.
- Plot bar graphs to visualize the skewness and kurtosis values.

6. Data Preparation
We prepare the dataset for modeling:
- Define the target variable and features.
- Split the data into training and testing sets.

7. Prediction using Linear Regression
We build and evaluate a Linear Regression model:
- Train the model on the training set.
- Make predictions on the test set.
- Evaluate the model using Mean Squared Error (MSE) and R² score.

8. Prediction using SVR (Support Vector Regression)
We build and evaluate a Support Vector Regression (SVR) model:
- Train the model on the training set.
- Make predictions on the test set.
- Evaluate the model using MSE and R² score.

9. Prediction using Ridge Regressor
We build and evaluate a Ridge Regression model:
- Train the model on the training set.
- Make predictions on the test set.
- Evaluate the model using MSE and R² score.

10. Prediction using Random Forest Regressor
We build and evaluate a Random Forest Regressor model:
- Train the model on the training set.
- Make predictions on the test set.
- Evaluate the model using MSE and R² score.
  
11. Hyperparameter Tuning for Linear Regression, SVR, Ridge Regressor, Random Forest Regressor
We perform hyperparameter tuning to optimize the models:
- Use Grid Search to find the best hyperparameters for each model.
- Evaluate the tuned models on the test set and compare their performance.
  
12. Plotting Graph for All Models to Compare Performance
We compare the performance of all models:
- Plot bar graphs to visualize the Mean Squared Error (MSE) and R² score for each model.
- Compare the performance of original and tuned models.

This notebook provides a structured approach to analyzing and modeling the insurance dataset, covering data exploration, preprocessing, visualization, modeling, and evaluation. It serves as a comprehensive guide for performing data science tasks on similar datasets.
