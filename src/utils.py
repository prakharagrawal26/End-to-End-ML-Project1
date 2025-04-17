import os
import sys
import pickle
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_yaml_config(file_path="src/components/config.yaml"):
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise CustomException(e, sys)

def get_model_class(model_name):
    """
    Returns the class of the given model name.
    """
    model_mapping = {
        "RandomForestRegressor": RandomForestRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "XGBRegressor": XGBRegressor,
        "CatBoostRegressor": CatBoostRegressor,
        "AdaBoostRegressor": AdaBoostRegressor,
        "LinearRegression": LinearRegression,
    }
    if model_name not in model_mapping:
        raise CustomException(f"Model {model_name} is not recognized")
    return model_mapping[model_name]

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model_class in models.items():
            parameters = param.get(model_name, {})  # Get model parameters
            model = model_class()  # Instantiate the model
            best_params = {}  # Store best hyperparameters

            if parameters:  # Perform GridSearchCV if hyperparameters exist
                gs = GridSearchCV(model, parameters, cv=3, n_jobs=-1,error_score="raise")
                gs.fit(X_train, y_train)
                model = gs.best_estimator_  # Best model
                best_params = gs.best_params_  # Best hyperparameters

            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R² Scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "r2_score": test_score,
                "train_score": train_score,
                "best_params": best_params
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)