# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The project is a package to predict customer churn. It provides

  1. An comprehensive EDA;

  2. A pipeline to train and save two ML models (Random Forest + Logistic Regression) for customer churn;

The package includes a comprehensive test suite.

## Files and data description

Input data:
	
 - Customer data available in `data/bank_data.csv`

Generated files from EDA:

 - `images/eda/Churn.png`: a histogram of how many customers have churned;

 - `images/eda/Customer_Age.png`: a histogram of the age of the churning customers;

 - `images/eda/heatmap.png`: a heatmap of the correlation between features:

 - `images/eda/marital_status.png`: a histogram of the marital status of the churning customers

 - `images/eda/Total_Trans_Ct.png`: a histogram of the total transactions of the churning customers;

Generate files from execution:

 - `/images/results/classification_report_rf.png`: the classification report of the random forest model;

 - `/images/results/classification_report_lr.png`: the classification report of the logistic regression model;

 - `/images/results/feature_importances.png`: a plot of feature importances for the random forest model;

 - `/models/rfc_model.pkl`: the trained random forest model;

 - `/models/logistic_model.pkl`: the trained logistic regression model;

## Running Files

How do you run your files? What should happen when you run your files?

To create baseline for images:

`pytest churn_script_logging_and_tests.py --mpl-generate-path=baseline`

the images created are then compared on subsequent runs of `pytest` as 

`pytest -v churn_script_logging_and_tests.py`

To generate the data, rune

`python churn_library.py`

The default is executed without retraining the two models. To retrain the two models, set the value `RETRAIN = True`

