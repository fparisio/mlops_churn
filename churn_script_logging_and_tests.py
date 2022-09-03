# pylint: disable=redefined-outer-name
# ^^^ this

"""
A test suite for the customer_churn script
----------------------------

Author: Francesco Parisio
Date: August 2022
Contact: francesco.parisio@protonmail.com
"""

import sys
import math
import logging
import pytest
import joblib
import numpy as np

import churn_library as cls

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Define logging config
logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


@pytest.fixture(scope="module")
def input_path():
    """input data"""
    yield "./data/bank_data.csv"


@pytest.fixture(scope="module")
def input_df(input_path):
    """load input dataframe

    Args:
      input_path(str): path to the input csv data

    Returns:

    """
    input_df = cls.import_data(input_path)
    yield input_df


@pytest.fixture(scope="module")
def perform_eda():
    """instantiate perform_eda class"""
    func_perform_eda = cls.perform_eda
    yield func_perform_eda


@pytest.fixture(scope="module")
def categorical_features():
    """returns categorical features"""
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    yield cat_columns


@pytest.fixture(scope="module")
def features_to_keep():
    """reaturns features after feat eng"""
    x_columns_to_keep = [
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
    yield x_columns_to_keep


@pytest.fixture(scope="module")
def split_ratio():
    """returns split ratio"""
    yield 0.3


@pytest.fixture(scope="module")
def pipeline(input_df, categorical_features, features_to_keep):
    """

    Args:
      input_df(TYPE): Description
      categorical_features(TYPE): Description
      features_to_keep(TYPE): Description

    Returns:

    """
    pipeline = cls.Pipeline(input_df, categorical_features, features_to_keep)
    yield pipeline


@pytest.fixture(scope="module")
def feature_engineering(input_df, categorical_features, features_to_keep, split_ratio):
    """

    Args:
      input_df(TYPE): Description
      categorical_features(TYPE): Description
      features_to_keep(TYPE): Description
      split_ratio(TYPE): Description

    Returns:

    """
    pipeline = cls.Pipeline(input_df, categorical_features, features_to_keep)
    x_train, x_test, y_train, y_test = pipeline.perform_feature_engineering(
        split_ratio)
    yield x_train, x_test, y_train, y_test


class TestImport(object):
    """A class to group all importing tests"""

    def test_import(self, input_path):
        """test data import

        Args:
          input_path(TYPE): Description
        No Longer Returned:

        Returns:

        Raises:
          err: FileNotFoundError to check correct import
Deleted Parameters:
          input_df: pytest fixture for the input file path

        """
        try:
            _ = cls.import_data(input_path)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_eda: The file wasn't found")
            raise err

    def test_non_null_dataframe(self, input_df):
        """test non-empty dataframe

        Args:
          input_df(pandas.DataFrame): pytest fixture for the input dataframe

        Returns:

        Raises:
          err: Assertion error to check non-empty dataframe

        """
        try:
            assert input_df.shape[0] > 0
            assert input_df.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns"
            )
            raise err


@pytest.mark.mpl_image_compare
class TestEDA(object):
    """A class to group all EDA tests"""

    def test_plot_histogram_churn(self, input_df):
        """test the histogram of churn plot

        Args:
          input_df(pandas.DataFrame): pytest fixture for the input dataframe

        Returns:

        Raises:
          err: Description

        """

        try:
            eda = cls.PerformEDA(input_df)
        except TypeError as err:
            logging.error("Testing plot_heatmap: missing input to function")
            raise err

        return eda.plot_histogram("Churn")

    def test_plot_histogram_age(self, input_df):
        """test the histogram of customer age plot

        Args:
          input_df(pandas.DataFrame): pytest fixture for the input dataframe

        Returns:

        Raises:
          err: Description

        """

        try:
            eda = cls.PerformEDA(input_df)
        except TypeError as err:
            logging.error("Testing plot_heatmap: missing input to function")
            raise err

        return eda.plot_histogram("Customer_Age")

    def test_plot_histogram_total_transactions(self, input_df):
        """test the plot of histogram of total transactions

        Args:
          input_df(pandas.DataFrame): pytest fixture for the input dataframe

        Returns:

        Raises:
          err: Description

        """

        try:
            eda = cls.PerformEDA(input_df)
        except TypeError as err:
            logging.error("Testing plot_heatmap: missing input to function")
            raise err

        return eda.plot_histogram("Total_Trans_Ct")

    def test_plot_count_marital_status(self, input_df):
        """test the plot of value count of marital status

        Args:
          input_df(pandas.DataFrame): pytest fixture for the input dataframe

        Returns:

        Raises:
          err: Description

        """

        try:
            eda = cls.PerformEDA(input_df)
        except TypeError as err:
            logging.error("Testing plot_heatmap: missing input to function")
            raise err

        return eda.plot_count_marital_status()

    def test_plot_heatmap(self, input_df):
        """test the plot of the data frame heatmap

        Args:
          input_df(pandas.DataFrame): pytest fixture for the input dataframe

        Returns:

        Raises:
          err: Description

        """

        try:
            eda = cls.PerformEDA(input_df)
        except TypeError as err:
            logging.error("Testing plot_heatmap: missing input to function")
            raise err

        return eda.plot_heatmap()

    def test_run_sequential_eda(self, input_df):
        """test the plot of the data frame heatmap

        Args:
          input_df(pandas.DataFrame): pytest fixture for the input dataframe

        Returns:

        Raises:
          err: Description

        """

        try:
            eda = cls.PerformEDA(input_df)
        except TypeError as err:
            logging.error("Run eda: missing input to function")
            raise err

        try:
            assert eda.run_sequential_eda()
            logging.info("Testing run_eda: SUCCESS")
        except AssertionError as err:
            logging.error("Testing run_eda: failed to execute")
            raise err


class TestEncoder(object):
    """A class to group all encoder tests"""

    def test_category_list(self, pipeline, features_to_keep):
        """test encoder helper

        Args:
          pipeline(TYPE): Description
          features_to_keep(TYPE): Description

        Returns:

        Raises:
          err: Description
Deleted Parameters:
          input_df: Description
          categorical_features: Description

        """
        encoded_df = pipeline.encoder_helper()
        try:
            assert encoded_df.columns.tolist() == features_to_keep
            logging.info("Testing category_list: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing category_list: the list of features is wrong.")
            raise err

    def test_columns_feature_engineering(self, pipeline, features_to_keep, split_ratio):
        """test perform_feature_engineering

        Args:
          pipeline(TYPE): Description
          features_to_keep(TYPE): Description
          split_ratio(TYPE): Description
          split_ratio(TYPE): Description
        perform_feature_engineering
          No Longer Returned: Deleted Parameters:
          perform_feature_engineering(TYPE): Description

        Returns:

        Raises:
          err: Description

        """
        xtrain, _, _, _ = pipeline.perform_feature_engineering(split_ratio)

        features_to_keep = features_to_keep.copy()
        features_to_keep.remove("Churn")

        try:
            assert xtrain.columns.tolist() == features_to_keep
        except AssertionError as err:
            logging.error(
                "Testing perform_feature_engineering: columns do not coincide."
            )
            logging.info(f"Current features: {xtrain.columns.tolist()} \n")
            logging.info(f"What  features should be: {features_to_keep} \n")
            raise err

    def test_data_split(self, pipeline, split_ratio):
        """test perform_feature_engineering

        Args:
          pipeline(TYPE): Description
          split_ratio(TYPE): Description
          split_ratio(TYPE): Description
        perform_feature_engineering
          No Longer Returned: Deleted Parameters:
          perform_feature_engineering(TYPE): Description

        Returns:

        Raises:
          err: Description

        """
        xtrain, xtest, ytrain, ytest = pipeline.perform_feature_engineering(
            split_ratio)

        try:
            assert (
                xtrain.shape[0]
                == pytest.approx(
                    math.floor((1 - split_ratio) *
                               (xtrain.shape[0] + xtest.shape[0]))
                )
            ) and (
                xtest.shape[0]
                == pytest.approx(
                    math.ceil(split_ratio * (xtrain.shape[0] + xtest.shape[0]))
                )
            )
            logging.info("Testing data_split: SUCCESS")
        except AssertionError as err:
            logging.error("Testing data_split: wrong size of vectors.")
            logging.info("X_train size: {}".format(xtrain.shape[0]))
            logging.info("X_test size: {}".format(xtest.shape[0]))
            raise err

        try:
            assert (
                ytrain.shape[0]
                == pytest.approx(
                    math.floor((1 - split_ratio) *
                               (ytrain.shape[0] + ytest.shape[0]))
                )
            ) and (
                ytest.shape[0]
                == pytest.approx(
                    math.ceil(split_ratio * (ytrain.shape[0] + ytest.shape[0]))
                )
            )
            logging.info("Testing data_split: SUCCESS")
        except AssertionError as err:
            logging.error("Testing data_split: wrong size of vectors.")
            logging.info("y_train size: {}".format(ytrain.shape[0]))
            logging.info("y_test size: {}".format(ytest.shape[0]))
            raise err


class TestModels(object):
    """A class to group all encoder tests"""

    def test_train_models(self, feature_engineering):
        """test train_models

        Args:
          feature_engineering(TYPE): Description
          feature_engineering(TYPE): Description
        train_models
        No Longer Returned:
        Deleted Parameters:
          train_models(TYPE): Description

        Returns:

        Raises:
          err: Description

        """
        xtrain, xtest, ytrain, ytest = feature_engineering

        try:
            assert (len(xtrain) == len(ytrain)) and (len(xtest) == len(ytest))
        except AssertionError as err:
            logging.error("Testing train_models: wrong input shape.")
            raise err

    @pytest.mark.slow
    def test_create_models_large(self, pipeline, feature_engineering):
        """Summary

        Args:
          pipeline(TYPE): Description
          feature_engineering(TYPE): Description

        Returns:

        Raises:
          err: Description

        """

        xtrain, xtest, ytrain, ytest = feature_engineering
        pipeline.train_models(xtrain, xtest, ytrain, ytest)

        # check that models were created
        try:
            rfc_model = joblib.load("./models/rfc_model.pkl")
            lr_model = joblib.load("./models/logistic_model.pkl")
            logging.info("Testing create_models: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing load_models: The file wasn't found")
            raise err

        # load reference models
        rfc_model_reference = joblib.load("./models/rfc_model_reference.pkl")
        lr_model_reference = joblib.load(
            "./models/logistic_model_reference.pkl")

        # compare with reference predictions rfc
        try:
            np.testing.assert_almost_equal(
                rfc_model.predict(xtest), rfc_model_reference.predict(xtest)
            )
            logging.info("Testing rfc creation: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing load_models: rfc model does not coincide with reference"
            )
            raise err

        # compare with reference predictions lr
        try:
            np.testing.assert_almost_equal(
                lr_model.predict(xtest), lr_model_reference.predict(xtest)
            )
            logging.info("Testing lr creation: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing load_models: lr model does not coincide with reference"
            )
            raise err


class MyPlugin:
    """plugin to execute pytests"""

    def pytest_sessionfinish(self):
        """print finishing session"""
        print("*** test run reporting finishing")


if __name__ == "__main__":
    sys.exit(pytest.main(
        ["-v", "churn_script_logging_and_tests.py"], plugins=[MyPlugin()]))
