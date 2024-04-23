"""
Unit test of model using pytest
author: Vivek Sheth
Date: April 22, 2024
"""

import pytest
import os
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from ml.model import inference, compute_model_metrics, compute_confusion_matrix
from ml.data import process_data


"""
Fixture - The test functions will 
use the return of data() as an argument
"""


@pytest.fixture(scope="module")
def data():
    # code to load in the data.
    datapath = "./data/census.csv"
    return pd.read_csv(datapath)


@pytest.fixture(scope="module")
def path():
    return "./data/census.csv"


@pytest.fixture(scope="module")
def features():
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"]
    return cat_features


@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Fixture - returns cleaned train dataset to be used for model testing
    """
    train, test = train_test_split(data,
                                   test_size=0.20,
                                   random_state=10,
                                   stratify=data['salary']
                                   )
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=features,
        label="salary",
        training=True
    )
    return X_train, y_train


def test_import_data(path):
    """
    Test dataset files for presence and shape
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as err:
        logging.error("Test file not found")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "The file doesn't appear to have rows and columns")
        raise err


def test_features(data, features):
    """
    Test that categorical features are in dataset or not
    """
    try:
        assert sorted(set(data.columns).intersection(
            features)) == sorted(features)
    except AssertionError as err:
        logging.error(
            "Testing dataset: Features are missing in the data columns")
        raise err


def test_is_model():
    """
    Test saved model is present or not
    """
    savepath = "../model/trained_model.pkl"
    if os.path.isfile(savepath):
        try:
            _ = pickle.load(open(savepath, 'rb'))
        except Exception as err:
            logging.error(
                "Testing saved model: Saved model does not appear to be valid")
            raise err
    else:
        pass


def test_is_fitted_model(train_dataset):
    """
    Test if saved model is fitted or not
    """

    X_train, y_train = train_dataset
    savepath = "./model/trained_model.pkl"
    model = pickle.load(open(savepath, 'rb'))

    try:
        model.predict(X_train)
    except NotFittedError as err:
        logging.error(
            f"Model is not fit, error {err}")
        raise err


def test_inference(train_dataset):
    """
    Test inference function from model
    """
    X_train, y_train = train_dataset

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))

        try:
            preds = inference(model, X_train)
        except Exception as err:
            logging.error(
                "Inference cannot be performed on saved model and train data")
            raise err
    else:
        pass


def test_compute_confusion_matrix(train_dataset):
    """
    Test calculation of confusion matrix function from model 
    """
    X_train, y_train = train_dataset

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))
        preds = inference(model, X_train)

        try:
            cm = compute_confusion_matrix(y_train, preds)
        except Exception as err:
            logging.error(
                "Confusion matrix cannot be calculated on train data")
            raise err
    else:
        pass


def test_compute_model_metrics(train_dataset):
    """
    Test calculation of performance metrics function from model
    """
    X_train, y_train = train_dataset

    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        model = pickle.load(open(savepath, 'rb'))
        preds = inference(model, X_train)

        try:
            precision, recall, fbeta = compute_model_metrics(y_train, preds)
        except Exception as err:
            logging.error(
                "Performance metrics cannot be calculated on train data")
            raise err
    else:
        pass
