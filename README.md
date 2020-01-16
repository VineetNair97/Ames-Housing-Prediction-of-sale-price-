# Ames-Housing-Prediction-of-sale-price-


## Introduction

The Ames Housing dataset, which describes the sale of individual residential property in Ames,
Iowa from 2006 to 2010, It includes different features of houses sold in Ames along with their corresponding Sales Price.
The dataset has various variables describing attributes of each house like its size, quality, area,
age, and other miscellaneous characteristics. The values of the target variable, Sale Price, is also
given so that the dataset can be used to apply different supervised machine learning algorithm.
The main objective of our project is to build regression models which can predict the Sale Price
of a house with high accuracy given its different other attributes.


## Data Understanding

We performed some Exploratory data analysis to study all the variables in depth. the dataset contains 2930 rows and 82 different
variables, out of which 39 are numerical and 43 are categorical.The target variable is the “SalePrice” which represents the selling price of each house in the
area. The average sale value of the properties in the dataset is $ 180,796.06 with the standard deviation of $79, 886.69. The minimum sale price is $12,789 while maximum is $75,5000.


## Data Preparation

We prepared our data in multiple steps:

1. Duplicated entries:
In the first step, we tried to see if there are any duplicate rows or columns in the dataset and we found none.
2. Missing values:
We found out that there were total 13997 entries listed as missing values and that was distributed among 27 variables out of 82.
for this dataset, there are surprisingly few nulls in the numeric data. Almost all the nulls are in the categorical variables.
We also found out that some of the variables like ‘Pool QC’, ‘Alley’, ‘Misc. Feature’, ‘Fence’ and ‘Fireplace Qu’ had the highest percentage of missing values. Upon further inspection, we
realized that all these variables had class ‘NA ‘which means the feature isn’t available in the house but could have been inferred as missing values. So, we checked each variable
closely to see if they are really a missing value or a class.

a) For variable ‘Pool QC’, we looked at corresponding values in ‘Poor Area’ and found that Pool Area was also zero for values which were null in ‘Pool QC’ which signifies that there
was no pool in those houses. So, we imputed all missing values with text ‘Not Available’.

b) We checked the corresponding value of ‘Fireplaces’ for each missing value in ‘Fireplace Qu’ and found out that the number of fireplaces were also zero for those rows with missing
values. We again imputed those missing values with text ‘Not Available’. Similarly, we also imputed other variables ‘Alley’, ‘Fence’ and ‘Miscellaneous Features’ with text ‘Not
available as they also clearly had class showing that feature is not available.

c) For variable ‘Lot frontage’, we plotted the histogram and realized that it is somehow normally distributed. So, missing values were replaced with mean.

d) For all the variables related to Garage like ‘Garage Yr Blt’, ‘Garage Cond’, ‘Garage Qual’,‘Garage Finish’, ‘Garage Type’ , ‘Garage Cars ‘ and ‘Garage Area’, we checked if the missing
values are missing in all these variables or not as they can again signify that there was no garage in those houses sold. We found that 157 rows in ‘Garage Cars’ and ‘Garage Area’ had value 0 while all other corresponding cells in other variables had null values which told us that these 157 houses had no garage in them. So, we imputed missing values in these rows
with text ‘Not available’ except for ‘Garage Yr Blt’ which we imputed with 0. We still had 1 or 2 missing values in these variables which we imputed with the mode.

e) We repeated the same process for all the variables related to Basement ‘Bsmt Exposure’,‘BsmtFin Type 1’, ‘BsmtFin Type 2’, ‘Bsmt Qual’, ‘Bsmt Cond’, ‘Total Bsmt SF’ and ‘Total Bsmt
Full Bath’. We found 79 rows with all missing values in these variables and imputed them with text ‘Not available’ or ‘0’ and for remaining 2 or 3 missing values in each variable, we
imputed them with mode.

f) We imputed 23 null values in each variable ‘Mass Vnr Area’ and ‘Mass Vnr Type’ with text‘Not available’ as well since we found that they were also class.

g) Finally, there was one last missing value in variable ‘Electrical’ which we imputed with mode.

## Outliers


After imputing all missing values, we next checked for the outliers in our dataset. For this, we took help of the scatterplot. We
found some outliers in variables like ‘GrLiving Area’, ‘Lot Frontage’, ‘Lot Area’ ,‘Wood Deck SF’ and ‘Enclosed Porch’ . We
couldn’t find a justifiable reason for these outliers to exist in the dataset. For example,we found out that there were some data
points with significantly lower Sale Price for higher Gr Living Area in the right, which shouldn’t be the case. So,we removed the rows with those outliers after which we had 2918 rows remaining out of 2930
rows we started with.we also checked values of other variables individually and found that for the variable ‘GarageYr Blt’, one value was 2207 which doesn’t make sense for the year. We looked at corresponding
value for ‘Year Built’ and found that it was 2006. So, we replaced 2207 


## Feature engineering

a. We removed variables ‘PID’ and ‘Order’ representing Property Identification number and observation numbers in the dataset as they didn’t provide any significant information about
sale price of the house for the analysis.

b. We then checked correlation between each two variables in the dataset to see if there is multicollinearity problem using heatmap.Upon inspecting the heatmap, we realized ‘Garage Area’ and ‘Garage Cars’ , ‘Gr Liv Area’
and ‘TotRms AbvGd’, ‘First Floor SF’ and ‘Total Bsmt SF’ have high collinearity with 6 correlation coefficient of 0.89, 0.80 and 0.78 respectively. We decided that 0.80 will be our
cut off point to decide in removing variables with high collinearity. So, we removed ‘GarageCars’ variable from the dataset and kept ‘Garage Area’ only.

c. We also checked the correlation between predictor variables and Sale Price and found that top seven variables with high positive collinearity were Overall Qual, Gr Liv Area, Total Bsmt
SF, Garage Area , 1st Flr SF and one with negative collinearity were Enclosed Porch , Kitchen AbvGr ,Overall Cond , MS SubClass, Bsmt Half Bath, Low Qual Fin SF and Yr Sold.

d. We plotted histogram of our target variable ‘SalePrice’ and used two numerical measures of shape skewness and kurtosis to test for normality. Kurtosis tells us the height and sharpness
of the central peak relative to that of a standard bell curve while skewness tells us how skewed our variable is from the center. Since ‘SalePrice’ had skewness of 1.59>1 therefore it
could be said that distribution is highly skewed to the right. We changed the scale to logarithmic scale, and it was found to be effective as the skewness value is between -0.5
and 0.5 which ensures that curve is more symmetric and the errors in predicting expensive houses and cheap houses will affect the result equally.
The distribution of target variable before and after log transformation
The residual plot between ‘SalePrice’ and one of the highly correlated variables with it ‘Gr Liv Area’ also got better after the transformation.

e. Our final step was to change all categorical variables to numerical as the models that we plan to use require only numerical values. We had 43 categorical variables in dataset among which 22 were ordinal and 21 were nominal. For Ordinal variables, we simply replaced the categories with ordinal numbers. For example: In ‘Alley’ variable, there are three categories:Paved, Graveled and not available. So, if there is no alley, it is one major drawback of the house, so we replaced it with zero. If it is gravel which is better than not having alley, we replaced it with 1 and if it is paved which is best option among all, we replaced it with 2. Weused a similar strategy to replace all other ordinal variables with ordinal numbers. For nominal variables, we used one-hot encoding in sci-kit learn library so that we don’t weight different values improperly.

## Model Induction


After our data was processed, we split it into training and testing set using test_train_split() function in sci-kit learn library. We also standardized all the variables except our response
variable so that all variables will have a similar effect on the model irrespective of their ranges.We created 3 different models using our processed data.
1. Multiple Linear Regression Using Least Square Estimation
2. Lasso Regression
3. Ridge Regression


From Lasso Regression, we found out that the variables with more influence on Sale Price were ‘Gr Liv Area’, ‘Overall Quality’,‘Year Built’, ‘Overall Cond’, ‘Total Bsmt SF’,
‘Garage Area’, ‘Lot Area’, ‘Functional’ and ‘Neighbourhood’ which aligns with our general knowledge about important features of a house for deciding its price.

## Model Evaluation


1. Metrics of Performance Measurement
We used 3 metrics to measure the performance of our model:

a) Root Mean squared error: It measures the square root of the average of the squared difference between the predictionsand real values. RMSE is more sensitive to outliers. So, if
outliers are undesirable, RMSE better evaluates how well our model is performing.

b) Mean absolute error (MAE): It measures the average of the absolute difference between each real value and the prediction. It is an absolute measure of fit and signifies standard deviation of the unexplained variance.

c) Coefficient of Determination(R-squared): It measures howwell the independent variables explain the variance in the dependent variable.

<img width="317" alt="AMES 1" src="https://user-images.githubusercontent.com/53135657/72563081-e6f1e700-387a-11ea-9820-d688ea7bec2d.PNG">


From the graph above, we can see that Multiple Linear Regression, Lasso Regression and Ridge Regression have higher R-square values than other three. For RMSE and MAE, Lasso Regression has the lowest value.


2. Cross Validation:

Cross Validation is one of the techniques used to evaluate the performance of machine learningmodels. It helps us to know if our model is under-fitting, over-fitting or well-generalized. It is a re-sampling procedure which is more effective when we have limited number of observations,but large number of variables like in our dataset.

![image](https://user-images.githubusercontent.com/53135657/72563676-308f0180-387c-11ea-8b2d-f15297ffce2e.png)

The result that we get from each model is summarized in the figure above. As we can see in the picture, we got lowest mean squared error from Ridge Regression followed by Multiple Linear Regression and Lasso Regression.


3. Learning Curve: 
While evaluating a model, we should also make sure that our model generalizes. A good model is one that not only performs well on data seen during training but, also on unseen data.Learning curve helps us to measure generalization performance of a model. In a learning curve,
the performance of a model both on the training and validation set is plotted as a function of the training set size.
from the plot the generalized performance for Lasso Regression is best among all. The test performance becomes similar with performance of training data as training size increases. For Ridge Regression, the performance on test set never gets closer to the performance in training set.

## Result


Looking at the combination of error measures, cross-validation scores and learning curves, Lasso Regression gave us the best result among all. It had one of the lowest Mean Absolute error and Root-Mean squared error and highest R-squared value. The generalization
performance measured by cross-validation and learning curve also favored this model.
