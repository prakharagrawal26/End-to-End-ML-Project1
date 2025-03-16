import os
import sys
from dataclasses import dataclass


from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, load_yaml_config, get_model_class

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.config = load_yaml_config()  # Load YAML config

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Loading model parameters from YAML")
            
            # Split training and test input data
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Load models and hyperparameters from YAML
            model_names = self.config["models"]
            params = self.config["hyperparameters"]

            # Dynamically get model classes
            model_instances = {name: get_model_class(name) for name in model_names}

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=model_instances, param=params
            )

            # Log details of each model
            for model_name, details in model_report.items():
                logging.info(f"Model: {model_name}")
                logging.info(f"Train R² Score: {details['train_score']:.4f}")
                logging.info(f"Test R² Score: {details['r2_score']:.4f}")
                logging.info(f"Best Hyperparameters: {details['best_params']}")

            # Get the best model
            best_model_name = max(model_report, key=lambda x: model_report[x]["r2_score"])
            best_model_score = model_report[best_model_name]["r2_score"]
            best_model_class = get_model_class(best_model_name)

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with R² score: {best_model_score}")

            # Instantiate and train the best model
            best_model = best_model_class(**model_report[best_model_name]["best_params"])
            best_model.fit(X_train, y_train)  # 🔹 Fixed: Model is trained before prediction

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using best model
            predicted = best_model.predict(X_test)
            final_r2_score = r2_score(y_test, predicted)
            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys)
