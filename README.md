[![CI](https://github.com/fparisio/mlops_churn/actions/workflows/python-app.yml/badge.svg)](https://github.com/fparisio/mlops_churn/actions/workflows/python-app.yml)

# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

<img src="images/results/roc_plot.png" width="400">

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

- `/images/results/roc_plot.png`: a plot of the ROC curves for RF and LR models;

- `/models/rfc_model.pkl`: the trained random forest model;

- `/models/logistic_model.pkl`: the trained logistic regression model;

## Running Files

To install the package:

```bash
poetry install
```

or with pip:

```bash
pip install .
```

To generate the data:

```bash
poetry run python mlops_churn/churn_library.py
```

To execute the tests:

```bash
python run pytest -v tests/churn_script_logging_and_tests.py
```

To create baseline for images:

```bash
pytest tests/churn_script_logging_and_tests.py --mpl-generate-path=baseline
```

the images created are then compared on subsequent test.

The default is executed without retraining the two models. To retrain the two models, set the value `RETRAIN = True`.
