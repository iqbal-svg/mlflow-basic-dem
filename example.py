import os  # Provides functions to interact with the operating system
import warnings  # Allows managing warnings (e.g., suppressing them)
import sys  # Used for accessing command-line arguments

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # For model evaluation
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.linear_model import ElasticNet  # Linear regression with L1 + L2 regularization
from urllib.parse import urlparse  # For parsing URLs
import mlflow  # ML lifecycle tool to log parameters, metrics, and models
from mlflow.models.signature import infer_signature  # To capture input/output structure of the model
import mlflow.sklearn  # MLflow module for logging scikit-learn models

import logging  # Standard logging module to report messages

logging.basicConfig(level=logging.WARN)  # Set logging level to WARNING
logger = logging.getLogger(__name__)  # Create logger for this script/module


def eval_metrics(actual, pred):  # Function to evaluate model performance
    rmse = np.sqrt(mean_squared_error(actual, pred))  # Root Mean Squared Error
    mae = mean_absolute_error(actual, pred)  # Mean Absolute Error
    r2 = r2_score(actual, pred)  # R-squared (coefficient of determination)
    return rmse, mae, r2  # Return all metrics


if __name__ == "__main__":  # Ensures the code below only runs if script is executed directly
    warnings.filterwarnings("ignore")  # Suppress warnings
    np.random.seed(40)  # Set random seed for reproducibility

    # URL to the wine quality dataset (semicolon-separated CSV)
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")  # Read dataset from URL using ';' as separator
    except Exception as e:  # If error occurs while reading
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )  # Log the error and exit

    train, test = train_test_split(data)  # Split data into training and test sets

    train_x = train.drop(["quality"], axis=1)  # Training features (excluding target)
    test_x = test.drop(["quality"], axis=1)  # Test features (excluding target)
    train_y = train[["quality"]]  # Training labels (target)
    test_y = test[["quality"]]  # Test labels (target)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5  # Read alpha from command line or default to 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5  # Read l1_ratio from command line or default to 0.5

    with mlflow.start_run():  # Start an MLflow run for tracking
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)  # Initialize ElasticNet model
        lr.fit(train_x, train_y)  # Train model

        predicted_qualities = lr.predict(test_x)  # Predict on test data

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)  # Calculate evaluation metrics

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))  # Display hyperparameters
        print("  RMSE: %s" % rmse)  # Print RMSE
        print("  MAE: %s" % mae)  # Print MAE
        print("  R2: %s" % r2)  # Print R² score

        mlflow.log_param("alpha", alpha)  # Log alpha parameter to MLflow
        mlflow.log_param("l1_ratio", l1_ratio)  # Log l1_ratio parameter
        mlflow.log_metric("rmse", rmse)  # Log RMSE metric
        mlflow.log_metric("r2", r2)  # Log R² score
        mlflow.log_metric("mae", mae)  # Log MAE

        predictions = lr.predict(train_x)  # Predict on training data to infer model signature
        signature = infer_signature(train_x, predictions)  # Capture the model input-output schema

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  # Check how MLflow is tracking runs

        if tracking_url_type_store != "file":  # If using a remote MLflow tracking server
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )  # Log and register model in MLflow Model Registry
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)  # Just log model without registry if using local tracking