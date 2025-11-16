import os
import sys
import numpy as np
from numpy import argmax

from src.exception.exception import DementiaException 
from src.logging.logger import logging

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.config_entity import ModelTrainerConfig

from src.utils.ml_utils.model.estimator import DementiaModel
from src.utils.main_utils.utils import save_object,load_object
from src.utils.main_utils.utils import load_numpy_array_data,evaluate_models
# **REMOVED**: get_classification_score, as we will calculate manually
# from src.utils.ml_utils.metric.classification_metric import get_classification_score 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# **ADDED**: Imports for manual metric calculation and threshold finding
from sklearn.metrics import (
    f1_score, 
    average_precision_score, 
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve
)

import mlflow
from urllib.parse import urlparse

import dagshub
# dagshub.init(repo_owner='sliitguy', repo_name='SecurityNetwork', mlflow=True)


from dotenv import load_dotenv
load_dotenv()


os.environ["MLFLOW_TRACKING_URI"]=os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]=os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]=os.getenv("MLFLOW_TRACKING_PASSWORD")



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise DementiaException(e,sys)
        
    ## <-- REMOVED: The 'track_mlflow' method was here.
    # It is now integrated into 'train_model'.
        
    def train_model(self,X_train,y_train,x_test,y_test):
        # Models and params are unchanged from your file
        models = {
                "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
                "CatBoost": CatBoostClassifier(random_state=42, verbose=1, auto_class_weights='Balanced'),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42, verbose=1),
                "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', verbose=1),
                "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
                "XGBClassifier": XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=1, scale_pos_weight=2.39),
            }
        params={
            "LightGBM": {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 63],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0, 0.1]
            },
            "CatBoost":{
                'learning_rate':[0.1, .01,.05,.001],
                'n_estimators': [8,16,32,128,256],
                'iterations': [100, 200],
                'depth': [4, 6],
                'l2_leaf_reg': [1, 3]
            },
            "Gradient Boosting":{
                'learning_rate':[.1,.01,.05,.001],
                'max_depth':[3,5],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{
                'penalty':['l1','l2'],
                'C':[0.01, 0.1, 1, 10],
            },
            "RandomForest":{
                'n_estimators':[100, 200, 300],
                'max_depth': [10, 20, 40],
                'min_samples_leaf':[1,2,4],
            },
            "XGBClassifier":{
                'n_estimators':[100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.01, 0.2],
                'max_depth':[3, 4, 5, 6]
            }
            
        }

        # **MODIFIED**: evaluate_models now returns ROC-AUC scores
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        
        logging.info(f"Best model found: {best_model_name} with ROC-AUC: {best_model_score}")

        # --- **START: NEW LOGIC FROM NOTEBOOK** ---

        # 1. Find Optimal Threshold on Training Data
        logging.info("Finding optimal F1 threshold on training data...")
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_train, y_train_proba)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
        f1_scores = f1_scores[:-1] # Align with thresholds
        
        best_threshold = thresholds[argmax(f1_scores)]
        logging.info(f"Optimal F1 threshold found: {best_threshold:.4f}")

        # 2. Calculate Train Metrics using threshold and probabilities
        y_train_pred_label = (y_train_proba >= best_threshold).astype(int)
        
        train_f1 = f1_score(y_train, y_train_pred_label, average='binary')
        train_recall = recall_score(y_train, y_train_pred_label)
        train_precision = precision_score(y_train, y_train_pred_label)
        train_pr_auc = average_precision_score(y_train, y_train_proba)
        train_roc_auc = roc_auc_score(y_train, y_train_proba)

        classification_train_metric = ClassificationMetricArtifact(
            f1_score=train_f1, 
            precision_score=train_precision, 
            recall_score=train_recall, 
            pr_auc_score=train_pr_auc, 
            roc_auc_score=train_roc_auc
        )
        
        ## <-- REMOVED: self.track_mlflow(best_model,classification_train_metric)

        # 3. Calculate Test Metrics using threshold and probabilities
        logging.info("Evaluating model on test set with optimal threshold...")
        y_test_proba = best_model.predict_proba(x_test)[:, 1]
        y_test_pred_label = (y_test_proba >= best_threshold).astype(int)

        test_f1 = f1_score(y_test, y_test_pred_label, average='binary')
        test_recall = recall_score(y_test, y_test_pred_label)
        test_precision = precision_score(y_test, y_test_pred_label)
        test_pr_auc = average_precision_score(y_test, y_test_proba)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)

        classification_test_metric = ClassificationMetricArtifact(
            f1_score=test_f1, 
            precision_score=test_precision, 
            recall_score=test_recall, 
            pr_auc_score=test_pr_auc, 
            roc_auc_score=test_roc_auc
        )

        ## <-- REMOVED: self.track_mlflow(best_model, classification_test_metric)

        # --- **END: NEW LOGIC** ---

        ## --- **START: NEW MLFLOW TRACKING BLOCK** ---
        mlflow.set_registry_uri("https://dagshub.com/sliitguy/SecurityNetwork.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        logging.info("Starting MLflow run...")
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_param("optimal_f1_threshold", f"{best_threshold:.4f}")
            
            # Log Train Metrics
            mlflow.log_metric("train_f1_score", classification_train_metric.f1_score)
            mlflow.log_metric("train_precision", classification_train_metric.precision_score)
            mlflow.log_metric("train_recall", classification_train_metric.recall_score)
            mlflow.log_metric("train_pr_auc", classification_train_metric.pr_auc_score)
            mlflow.log_metric("train_roc_auc", classification_train_metric.roc_auc_score)
            
            # Log Test Metrics
            mlflow.log_metric("test_f1_score", classification_test_metric.f1_score)
            mlflow.log_metric("test_precision", classification_test_metric.precision_score)
            mlflow.log_metric("test_recall", classification_test_metric.recall_score)
            mlflow.log_metric("test_pr_auc", classification_test_metric.pr_auc_score)
            mlflow.log_metric("test_roc_auc", classification_test_metric.roc_auc_score)
            
            # Log Model
            mlflow.sklearn.log_model(best_model, "model")
        
        logging.info("MLflow run completed.")
        ## --- **END: NEW MLFLOW TRACKING BLOCK** ---


        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Dementia_Model_Object=DementiaModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Dementia_Model_Object)
        
        #model pusher
        save_object("final_model/model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise DementiaException(e,sys)