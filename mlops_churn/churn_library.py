"""
churn_library: A script to predict customer churn
=====
It provides
  1. An comprehensive EDA
  2. A pipeline to train and save two ML models for customer churn
----------------------------

Author: Francesco Parisio
Date: August 2022
Contact: francesco.parisio@protonmail.com
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()


# import libraries
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """returns dataframe for the csv found at pth

    Args:
      pth(str): path to the input data

    Returns:


    """
    # Read csv
    data = pd.read_csv(pth)

    # Assign churn variables
    data["Churn"] = data["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    return data


class PerformEDA:
    """A class to perform the EDA"""

    def __init__(self, df):
        """Summary

        Args:
            df (pandas.DataFrame): input for the EDA
        """
        self.df = df

    def plot_histogram(self, variable):
        """plot histogram of churn

        Args:
          variable(str): name of the column to plot
          variable(str): name of the column to plot
        the histogram

        Returns:


        """
        fig, ax = plt.subplots()
        self.df[variable].hist(ax=ax)

        fig.tight_layout()

        plt.savefig("./images/eda/" + variable + ".png", dpi=500)

        return fig

    def plot_count_marital_status(self):
        """plot value count of marital status"""
        fig, ax = plt.subplots()
        self.df["Marital_Status"].value_counts("normalize").plot(kind="bar", ax=ax)

        fig.tight_layout()

        plt.savefig("./images/eda/marital_status.png", dpi=500)
        return fig

    def plot_heatmap(self):
        """plot value count of marital status"""
        fig, _ = plt.subplots()
        sns.heatmap(self.df.corr(), annot=False, cmap="Dark2_r", linewidths=2)

        fig.tight_layout()

        plt.savefig("./images/eda/heatmap.png", dpi=500)
        return fig

    def run_sequential_eda(self):
        """A function to sequentially execute all plots"""
        self.plot_histogram("Churn")
        self.plot_histogram("Customer_Age")
        self.plot_histogram("Total_Trans_Ct")
        self.plot_count_marital_status()
        self.plot_heatmap()

        return True


class Pipeline:
    """A class to perform execute the ML training steps sequentially"""

    def __init__(self, df, category_lst, col_to_keep):
        """
        Args:
            df (pandas.DataFrame): input for the EDA
        """
        self.df = df
        self.category_lst = category_lst
        self.col_to_keep = col_to_keep

    def encoder_helper(self, response="Churn"):
        """helper function to turn each categorical column into a new column with
        propotion of churn for each category

        Args:
          df(pandas.DataFrame): pandas dataframe
          category_lst(list): columns that contain categorical features
          response(str, optional): string of response name [optional argument that
        could be used for naming variables
        or index y column] (Default value = "Churn")

        Returns:


        """
        for i in self.category_lst:
            # encoded column
            tmp_lst = []
            tmp_groups = self.df.groupby(i).mean()[response]

            for val in self.df[i]:
                tmp_lst.append(tmp_groups.loc[val])

            self.df[i + "_" + response] = tmp_lst

        return self.df[self.col_to_keep]

    def perform_feature_engineering(self, split_ratio):
        """Function to pefrom feature engineering on the data

        Args:
          split_ratio(float): split ration between training and testings
        could be used for naming variables or index y column] (Default value = "")

        Returns:
          X_train(pandas.DataFrame): X training data
          X_test(pandas.DataFrame): X testing data
          y_train(pandas.DataFrame): y training data
          y_test(pandas.DataFrame): y testing data

        """

        self.df_encoded = self.encoder_helper()

        self.y_labels = self.df_encoded.pop("Churn")
        self.X_features = self.df_encoded

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_features, self.y_labels, test_size=split_ratio, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def classification_report_image(
        self,
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    ):
        """produces classification report for training
         and testing results and stores report as image
        in images folder

        Args:
          y_train(pandas.Series): training response values
          y_test(pandas.Series): test response values
          y_train_preds_lr(numpy.ndarray): training predictions from logistic regression
          y_train_preds_rf(numpy.ndarray): training predictions from random forest
          y_test_preds_lr(numpy.ndarray): test predictions from logistic regression
          y_test_preds_rf(numpy.ndarray): test predictions from random forest

        Returns:


        """

        # Classification report for random forest
        fig, ax = plt.subplots()
        plt.rc("figure", figsize=(5, 5))
        ax.text(
            0.01,
            1.25,
            str("Random Forest Train"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        ax.text(
            0.01,
            0.05,
            str(classification_report(y_test, y_test_preds_rf)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        # approach improved by OP -> monospace!
        ax.text(
            0.01,
            0.6,
            str("Random Forest Test"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        ax.text(
            0.01,
            0.7,
            str(classification_report(y_train, y_train_preds_rf)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        # approach improved by OP -> monospace!
        ax.axis("off")
        fig.tight_layout()
        plt.savefig("./images/results/classification_report_rf.png", dpi=500)

        # Classification report for logistic regression
        fig, ax = plt.subplots()
        plt.rc("figure", figsize=(5, 5))
        ax.text(
            0.01,
            1.25,
            str("Logistic Regression Train"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        ax.text(
            0.01,
            0.05,
            str(classification_report(y_train, y_train_preds_lr)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        # approach improved by OP -> monospace!
        ax.text(
            0.01,
            0.6,
            str("Logistic Regression Test"),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        ax.text(
            0.01,
            0.7,
            str(classification_report(y_test, y_test_preds_lr)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        # approach improved by OP -> monospace!
        ax.axis("off")
        fig.tight_layout()
        plt.savefig("./images/results/classification_report_lr.png", dpi=500)

    def feature_importance_plot(self, model, X_data, output_pth):
        """creates and stores the feature importances in pth

        Args:
          model(sklearn.ensemble._forest.RandomForestClassifier): model object containing feature_importances_
          X_data(pandas.DataFrame): pandas dataframe of X values
          output_pth(str): path to store the figure

        Returns:


        """

        # Calculate feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        fig, ax = plt.subplots()

        # Create plot title
        ax.set_title("Feature Importance")
        ax.set_ylabel("Importance")

        # Add bars
        ax.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        ax.set_xticks(range(X_data.shape[1]))
        ax.set_xticklabels(names, rotation=90)

        # Layout configure
        fig.tight_layout()

        # Save output
        plt.savefig(output_pth, dpi=500)

        return fig

    def roc_plot(self, models, test_data, output_pth):
        """creates the ROC plot for the two models

        Args:
          models(tuple): LR and RF models
          test_data(tuple): x and y test data
          output_pth(str): path to store the figure

        Returns:


        """
        # unpack input
        lr_model, rfc_model = models
        X_test, y_test = test_data

        # Create plot
        fig, ax = plt.subplots()

        # plot linear regression
        plot_roc_curve(lr_model, X_test, y_test, ax=ax, alpha=0.8)

        # plot random forest
        plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)

        # Layout configure
        fig.tight_layout()

        # Save output
        plt.savefig(output_pth, dpi=500)

        return fig

    def train_models(self, X_train, X_test, y_train, retrain=False):
        """train, store model results: images + scores, and store models

        Args:
          X_train(pandas.DataFrame): X training data
          X_test(pandas.DataFrame): X testing data
          y_train(pandas.DataFrame): y training data
          y_test(pandas.DataFrame): y testing data
          retrain: (Default value = False)

        Returns:


        """
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

        if retrain:
            # fit and dump
            cv_rfc.fit(X_train, y_train)
            lrc.fit(X_train, y_train)
            joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
            joblib.dump(lrc, "./models/logistic_model.pkl")

        # load and make predictions
        rfc_model = joblib.load("./models/rfc_model.pkl")
        lr_model = joblib.load("./models/logistic_model.pkl")

        self.y_train_preds_rf = rfc_model.predict(X_train)
        self.y_test_preds_rf = rfc_model.predict(X_test)

        self.y_train_preds_lr = lr_model.predict(X_train)
        self.y_test_preds_lr = lr_model.predict(X_test)


def main():
    """main execution function
    TBD: use a parser for cmd line arguments

    Args:

    Returns:


    """

    # parameters
    # features representing a category
    CATEGORIES = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    # features to keep in the final pipeline
    FINAL_FEATURES = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
        "Churn",
    ]

    # bool variable to whether or not retrain the model
    RETRAIN = False

    # load data
    df = import_data("./data/bank_data.csv")

    # execute EDA
    eda = PerformEDA(df)
    eda.run_sequential_eda()

    # build pipeline
    model_pipeline = Pipeline(df, CATEGORIES, FINAL_FEATURES)

    # feature engineering
    X_train, X_test, y_train, y_test = model_pipeline.perform_feature_engineering(0.3)

    # train models
    model_pipeline.train_models(X_train, X_test, y_train, y_test, RETRAIN)

    # load models
    lr_model_ = joblib.load("./models/logistic_model.pkl")
    rfc_model_ = joblib.load("./models/rfc_model.pkl")

    # build feature importance plot
    model_pipeline.feature_importance_plot(
        rfc_model_,
        model_pipeline.X_features,
        "./images/results/feature_importances.png",
    )

    # build roc plot
    model_pipeline.roc_plot(
        (lr_model_, rfc_model_),
        (X_test, y_test),
        "./images/results/roc_plot.png",
    )

    # build classification report plot
    model_pipeline.classification_report_image(
        y_train,
        y_test,
        model_pipeline.y_train_preds_lr,
        model_pipeline.y_train_preds_rf,
        model_pipeline.y_test_preds_lr,
        model_pipeline.y_test_preds_rf,
    )


if __name__ == "__main__":

    main()
