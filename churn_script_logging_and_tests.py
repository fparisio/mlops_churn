"""
A test suite for the customer_churn script
----------------------------

Author: Francesco Parisio
Date: August 2022
Contact: francesco.parisio@protonmail.com
"""


import os
import logging
import churn_library as cls
import pytest
import math
import joblib
import numpy as np

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Define logging config
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope='module')
def input_path():
    """
    Yields:
        str: path to the input csv data
    """
    yield "./data/bank_data.csv"


@pytest.fixture(scope='module')
def input_df(input_path):
    """
    Args:
        input_path (str): path to the input csv data

    Yields:
        pandas.DataFrame: input dataframe
    """
    input_df = cls.import_data(input_path)
    yield input_df


@pytest.fixture(scope='module')
def perform_eda():
    """
    Yields:
        TYPE: Description
    """
    func_perform_eda = cls.perform_eda
    yield func_perform_eda


@pytest.fixture(scope='module')
def categorical_features():
    """
    Yields:
        TYPE: Description
    """
    cat_columns = ['Gender', 'Education_Level',
                   'Marital_Status', 'Income_Category', 'Card_Category']
    yield cat_columns


@pytest.fixture(scope='module')
def features_to_keep():
    """
    Yields:
        TYPE: Description
    """
    X_columns_to_keep = ['Customer_Age',
                         'Dependent_count',
                         'Months_on_book',
                         'Total_Relationship_Count',
                         'Months_Inactive_12_mon',
                         'Contacts_Count_12_mon',
                         'Credit_Limit',
                         'Total_Revolving_Bal',
                         'Avg_Open_To_Buy',
                         'Total_Amt_Chng_Q4_Q1',
                         'Total_Trans_Amt',
                         'Total_Trans_Ct',
                         'Total_Ct_Chng_Q4_Q1',
                         'Avg_Utilization_Ratio',
                         'Gender_Churn',
                         'Education_Level_Churn',
                         'Marital_Status_Churn',
                         'Income_Category_Churn',
                         'Card_Category_Churn',
                         'Churn']
    yield X_columns_to_keep


@pytest.fixture(scope='module')
def split_ratio():
    """
    Yields:
        TYPE: Description
    """
    SPLIT_RATIO = 0.3
    yield SPLIT_RATIO


@pytest.fixture(scope='module')
def pipeline(input_df, categorical_features, features_to_keep):
    """
    Args:
        input_df (TYPE): Description
        categorical_features (TYPE): Description
        features_to_keep (TYPE): Description

    Yields:
        TYPE: Description
    """
    pipeline = cls.Pipeline(input_df, categorical_features, features_to_keep)
    yield pipeline


@pytest.fixture(scope='module')
def feature_engineering(input_df, categorical_features, features_to_keep, split_ratio):
    """
    Args:
        input_df (TYPE): Description
        categorical_features (TYPE): Description
        features_to_keep (TYPE): Description
        split_ratio (TYPE): Description

    Yields:
        TYPE: Description
    """
    pipeline = cls.Pipeline(input_df, categorical_features, features_to_keep)
    X_train, X_test, y_train, y_test = pipeline.perform_feature_engineering(
        split_ratio)
    yield X_train, X_test, y_train, y_test


class TestImport(object):
    """A class to group all importing tests
    """

    def test_import(self, input_path):
        """test data import

        Args:
            input_path (TYPE): Description

        No Longer Returned:

        Raises:
            err: FileNotFoundError to check correct import

        Deleted Parameters:
            input_df (str): pytest fixture for the input file path
        """
        try:
            df = cls.import_data(input_path)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_eda: The file wasn't found")
            raise err

    def test_non_null_dataframe(self, input_df):
        """test non-empty dataframe

        Args:
            input_df (pandas.DataFrame): pytest fixture for the input dataframe

        Raises:
            err: Assertion error to check non-empty dataframe
        """
        try:
            assert input_df.shape[0] > 0
            assert input_df.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err


@pytest.mark.mpl_image_compare
class TestEDA(object):
    """A class to group all EDA tests
    """

    def test_plot_histogram_churn(self, input_df):
        """test the histogram of churn plot

        Args:
            input_df (pandas.DataFrame): pytest fixture for the input dataframe

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
            input_df (pandas.DataFrame): pytest fixture for the input dataframe

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
            input_df (pandas.DataFrame): pytest fixture for the input dataframe

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
            input_df (pandas.DataFrame): pytest fixture for the input dataframe

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
            input_df (pandas.DataFrame): pytest fixture for the input dataframe

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


class TestSequentialEDA(object):
    def test_run_sequential_eda(self, input_df):
        """test the plot of the data frame heatmap

        Args:
            input_df (pandas.DataFrame): pytest fixture for the input dataframe

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
    """A class to group all encoder tests
    """

    def test_category_list(self, pipeline, features_to_keep):
        """test encoder helper

        Args:
            pipeline (TYPE): Description
            features_to_keep (TYPE): Description

        Raises:
            err: Description

        Deleted Parameters:
            input_df (TYPE): Description
            categorical_features (TYPE): Description
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
            pipeline (TYPE): Description
            features_to_keep (TYPE): Description
            split_ratio (TYPE): Description
            perform_feature_engineering
            No Longer Returned:

        Deleted Parameters:
            perform_feature_engineering (TYPE): Description

        Raises:
            err: Description

        """
        X_train, X_test, y_train, y_test = pipeline.perform_feature_engineering(
            split_ratio)

        features_to_keep_X = features_to_keep.copy()
        features_to_keep_X.remove("Churn")

        try:
            assert X_train.columns.tolist() == features_to_keep_X
        except AssertionError as err:
            logging.error(
                "Testing perform_feature_engineering: columns do not coincide.")
            logging.info(
                "Current features: {} \n".format(X_train.columns.tolist()))
            logging.info(
                "What  features should be: {} \n".format(features_to_keep_X))
            raise err

    def test_data_split(self, pipeline, features_to_keep, split_ratio):
        """test perform_feature_engineering

        Args:
            pipeline (TYPE): Description
            features_to_keep (TYPE): Description
            split_ratio (TYPE): Description
            perform_feature_engineering
            No Longer Returned:

        Deleted Parameters:
            perform_feature_engineering (TYPE): Description

        Raises:
            err: Description

        """
        X_train, X_test, y_train, y_test = pipeline.perform_feature_engineering(
            split_ratio)

        try:
            assert (X_train.shape[0] == pytest.approx(
                math.floor((1 - split_ratio) * (X_train.shape[0] + X_test.shape[0])))) and (X_test.shape[0] == pytest.approx(
                    math.ceil(split_ratio * (X_train.shape[0] + X_test.shape[0]))))
            logging.info("Testing data_split: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing data_split: wrong size of vectors.")
            logging.info("X_train size: {}".format(X_train.shape[0]))
            logging.info("X_test size: {}".format(X_test.shape[0]))
            raise err

        try:
            assert (y_train.shape[0] == pytest.approx(
                math.floor((1 - split_ratio) * (y_train.shape[0] + y_test.shape[0])))) and (y_test.shape[0] == pytest.approx(
                    math.ceil(split_ratio * (y_train.shape[0] + y_test.shape[0]))))
            logging.info("Testing data_split: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing data_split: wrong size of vectors.")
            logging.info("y_train size: {}".format(y_train.shape[0]))
            logging.info("y_test size: {}".format(y_test.shape[0]))
            raise err


class TestModels(object):
    """A class to group all encoder tests
    """

    def test_train_models(self, pipeline, feature_engineering):
        """test train_models

        Args:
            pipeline (TYPE): Description
            feature_engineering (TYPE): Description
            train_models

        No Longer Returned:

        Deleted Parameters:
            train_models (TYPE): Description

        Raises:
            err: Description


        """
        X_train, X_test, y_train, y_test = feature_engineering

        try:
            assert (len(X_train) == len(y_train)) and (
                len(X_test) == len(y_test))
        except AssertionError as err:
            logging.error(
                "Testing train_models: wrong input shape.")
            raise err

    @pytest.mark.slow
    def test_create_models_large(self, pipeline, feature_engineering):
        """Summary

        Raises:
            err: Description

        Args:
            pipeline (TYPE): Description
            feature_engineering (TYPE): Description
        """

        X_train, X_test, y_train, y_test = feature_engineering
        pipeline.train_models(X_train, X_test, y_train, y_test)

        # check that models were created
        try:
            rfc_model = joblib.load('./models/rfc_model.pkl')
            lr_model = joblib.load('./models/logistic_model.pkl')
            logging.info("Testing create_models: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing load_models: The file wasn't found")
            raise err

        # load reference models
        rfc_model_reference = joblib.load('./models/rfc_model_reference.pkl')
        lr_model_reference = joblib.load(
            './models/logistic_model_reference.pkl')

        # compare with reference predictions rfc
        try:
            np.testing.assert_almost_equal(rfc_model.predict(
                X_test), rfc_model_reference.predict(X_test))
            logging.info("Testing rfc creation: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing load_models: rfc model does not coincide with reference")
            raise err

        # compare with reference predictions lr
        try:
            np.testing.assert_almost_equal(lr_model.predict(
                X_test), lr_model_reference.predict(X_test))
            logging.info("Testing lr creation: SUCCESS")
        except AssertionError as err:
            logging.error(
                "Testing load_models: lr model does not coincide with reference")
            raise err


if __name__ == "__main__":
    pass
