
# Customer Churn Prediction

### Problem Statement

Decreasing the customer churn is a key goal for any business. Predicting customer churn (also known as Customer Attrition) represent an additional potenetial revenue source for any business. Customer churn impacts the cost to the business. Higher Customer churn leads to loss in revenue and the additional marketing costs involved with replacing those customers with new ones.

Our objective is to build a machine learning models to predict whether the customer will churn or not in the next six months.

### Feature Description

|variable|Description|
|-|-|
|ID|Unique Identifier of a row|
|Age|Age of the customer|
|Gender|Gender of the customer(Male or Female)|
|Income|Yearly income of the customer|
|Balance|Average quarterly balance of the customer|
|Vintage|No. of years the customer is associated with bank|
|Transaction_status|Whether the customer has done any transaction in the past 3 months or not|
|Product_Holdings|No. of Product holdings with the bank|
|Credit_Card|Whether the customer has a credit card or not|
|Credit_Category|Catogory of a customer based on the credit score|
|Is_Churn|Whether the customer will churn in next 6 months or not|


## Approach
### 1. Exploratory Data Analysis for Feature Selection

First, I did some statistical data analysis and Data Visualization.


### 2. Hypothesis test for Feature Selection

Performed a hypothesis test for the statistical relationship between two attributes. The `Chi-square test` is performed on data to find the relation between `Is_churn` and `Other_column_feature`. 

* H0: feature1 and feature2 are dependent.
* H1: feature1 and feature2 are not dependent.

The result shows that, 'Age', 'Gender', 'Transaction_Status', 'Product_Holdings' and 'Credit_Category' are dependent variables.

### 3. Data Preprocessing

***Age*** - This is highly correlated with label. But some outlier present in data so, for better analysis, Binning or Discretization is perform on Age feature which smooths out data with normal distribution. 

***Balance*** - The Balance feature has a skewed distribution with outliers. After treating the outlier, log transformation is performed to reduce the dynamic range of a feature.

***Normalization*** - After feature selection, Normalization is perfromed on data using `MinMaxScaler`.

### 4. Imbalanced Data

The data is highly imbalanced i.e. the number of customers who churned are significantly smaller than the number of customers who did not. I tried different approaches to handled imbalanced data. 

*  Using Class Weight 

Tunning Class weight of sklearn modules gives a best result on training data but does not generalized well on test data which results in Overfitting the model.

* Resampling data using IMBLearn library 

Tried different resampling methods of `imblearn` library such as RandomOverSampler, SMOTE, SVMSMOTE, RandomUnderSampler, combination of RandomOverSampler and RandomUnderSampler to balance the data.

### 5. Model Training
I tried serveral base model such LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, Naive Bayes, GradientBoostingClassifier, XGBoost on balanced data. Amonge these models, `GradientBoostingClassifier` with `SVMSMOTE` oversampling technique gives best macro f1 score and therefore, this model is used for final submission.


