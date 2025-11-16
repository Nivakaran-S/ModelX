import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.constants.training_pipeline import TARGET_COLUMN
# from src.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS # No longer needed

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import DementiaException 
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise DementiaException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise DementiaException(e, sys)
        
    def get_data_transformer_object(self, numeric_features, object_categorical_features, manual_categorical_features) -> ColumnTransformer:
        """
        This function creates and returns a ColumnTransformer based on the
        feature lists provided, replicating the notebook's logic.
        """
        logging.info("Building preprocessing pipelines for 3 feature types.")
        try:
            # --- Pipeline 1: NUMERIC features ---
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # --- Pipeline 2: STRING Categorical features ---
            string_categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int8))
            ])

            # --- Pipeline 3: NUMERIC-CODED Categorical features ---
            numeric_categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int8))
            ])

            # --- Create the master preprocessor ---
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat_str', string_categorical_transformer, object_categorical_features),
                    ('cat_num', numeric_categorical_transformer, manual_categorical_features)
                ],
                remainder='passthrough',
                n_jobs=-1
            )
            
            logging.info("ColumnTransformer built successfully.")
            return preprocessor
        
        except Exception as e:
            raise DementiaException(e,sys)

        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            # --- Replicating Notebook Feature Identification Logic ---
            logging.info("Identifying feature types (numeric, string-cat, numeric-cat)...")
            all_cols = input_feature_train_df.columns.tolist()

            # 1. Identify true object/category columns
            object_categorical_features = input_feature_train_df.select_dtypes(include=['object', 'category']).columns.tolist()

            # 2. Identify numeric-coded categorical columns (as defined in your notebook)
            #    NOTE: 'INDEPEND' was removed in your notebook, so I am omitting it.
            potential_manual_categorical = ['SEX', 'MARISTAT', 'LIVSIT', 'RESIDENC'] 
            manual_categorical_features = [
                col for col in potential_manual_categorical
                if col in all_cols and col not in object_categorical_features
            ]

            # 3. All remaining columns are numeric
            numeric_features = [
                col for col in all_cols
                if col not in object_categorical_features and col not in manual_categorical_features
            ]
            
            logging.info(f"Found {len(numeric_features)} numeric features.")
            logging.info(f"Found {len(object_categorical_features)} string categorical features.")
            logging.info(f"Found {len(manual_categorical_features)} numeric categorical features.")
            # --- End of Feature Identification ---


            preprocessor = self.get_data_transformer_object(
                numeric_features=numeric_features,
                object_categorical_features=object_categorical_features,
                manual_categorical_features=manual_categorical_features
            )

            preprocessor_object=preprocessor.fit(input_feature_train_df)
            
            logging.info("Transforming training and testing dataframes...")
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)
             
            # Combine transformed features with target
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            logging.info("Saving processed arrays and preprocessor object.")
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)


            #preparing artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info(f"Data transformation artifact created: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise DementiaException(e,sys)