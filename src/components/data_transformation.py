import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler  # <-- CHANGED from StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import DementiaException
from src.logging.logger import logging
from src.utils.main_utils.utils import save_object, read_yaml_file
from src.constants.training_pipeline import SCHEMA_FILE_PATH

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise DementiaException(e, sys)

    def _get_feature_engineering_columns(self):
        """
        Helper function to define column lists for feature engineering
        based on the notebook's logic.
        """
        comorbidity_cols = [
            'CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACDEF', 'CVPACE', 'CVCHF',
            'CVANGINA', 'CVHVALVE', 'CBSTROKE', 'CBTIA', 'PD', 'SEIZURES', 'TBI',
            'DIABETES', 'HYPERTEN', 'HYPERCHO', 'B12DEF', 'THYROID', 'ARTHRIT',
            'INCONTU', 'INCONTF', 'APNEA', 'RBD', 'INSOMN', 'ALCOHOL', 'ABUSOTHR',
            'PTSD', 'BIPOLAR', 'SCHIZ', 'DEP2YRS', 'ANXIETY', 'OCD', 'NPSYDEV', 'PSYCDIS'
        ]

        memory_cols = [
            'EMPTY', 'BORED', 'SPIRITS', 'AFRAID', 'HAPPY', 'HELPLESS', 'STAYHOME',
            'MEMPROB', 'WONDRFUL', 'WRTHLESS', 'ENERGY', 'HOPELESS', 'BETTER'
        ]

        functional_cols = [
            'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP',
            'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL'
        ]
        
        return comorbidity_cols, memory_cols, functional_cols

    def _get_columns_to_drop(self):
        """
        Helper function to define columns to drop based on the notebook.
        """
        # 1. Columns used to create new features
        comorbidity_cols, memory_cols, functional_cols = self._get_feature_engineering_columns()
        
        # 2. Object type columns from schema
        object_cols_to_drop = [
            list(col_dict.keys())[0] for col_dict in self._schema_config['columns']
            if list(col_dict.values())[0] == 'object'
        ]
        
        # 3. Drug columns from schema
        drug_cols_to_drop = [f'DRUG{i}' for i in range(1, 41)]
        
        # 4. Other columns identified in notebook (e.g., high correlation)
        other_cols_to_drop = ['NACCNIHR']

        # Combine all columns to be dropped
        all_cols_to_drop = list(set(
            comorbidity_cols + memory_cols + functional_cols + 
            object_cols_to_drop + drug_cols_to_drop + other_cols_to_drop
        ))
        
        return all_cols_to_drop

    def _perform_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the new features from the notebook and adds them to the dataframe.
        """
        logging.info("Starting feature engineering...")
        comorbidity_cols, memory_cols, functional_cols = self._get_feature_engineering_columns()
        
        # Ensure all columns exist, fill missing with 0 for summation
        for col in comorbidity_cols + memory_cols + functional_cols:
            if col not in df.columns:
                logging.warning(f"Column '{col}' not found for feature engineering. Assuming 0.")
                df[col] = 0
            else:
                # Fill NaNs with 0 for summation logic
                df[col] = df[col].fillna(0)
                
        df['FE_COMORBIDITY_SCORE'] = df[comorbidity_cols].sum(axis=1)
        df['FE_MEMORY_COMPLAINT'] = df[memory_cols].sum(axis=1)
        df['FE_FUNCTIONAL_DECLINE'] = df[functional_cols].sum(axis=1)
        
        logging.info("Feature engineering complete.")
        return df

    def get_data_transformer_object(self, numerical_columns: list) -> ColumnTransformer:
        """
        Creates the ColumnTransformer object using RobustScaler as per the notebook.
        """
        try:
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())  # <-- Use RobustScaler
            ])

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            raise DementiaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation initiated.")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            logging.info("Reading train and test data completed.")

            # --- Notebook Logic ---
            # 1. Perform Feature Engineering
            train_df = self._perform_feature_engineering(train_df)
            test_df = self._perform_feature_engineering(test_df)
            
            # 2. Drop original columns
            cols_to_drop = self._get_columns_to_drop()
            
            # Ensure we only drop columns that actually exist
            existing_cols_to_drop_train = [col for col in cols_to_drop if col in train_df.columns]
            existing_cols_to_drop_test = [col for col in cols_to_drop if col in test_df.columns]
            
            train_df = train_df.drop(columns=existing_cols_to_drop_train)
            test_df = test_df.drop(columns=existing_cols_to_drop_test)
            logging.info(f"Dropped {len(existing_cols_to_drop_train)} columns.")
            # --- End Notebook Logic ---

            # 3. Separate Target and Features
            target_column_name = self._schema_config['target_column']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # 4. Get Preprocessing Object
            # All remaining columns are numerical and will be processed
            numerical_columns = list(input_feature_train_df.columns)
            preprocessor = self.get_data_transformer_object(numerical_columns)
            
            logging.info("Applying preprocessing object on training and testing dataframes.")

            # 5. Apply Preprocessor
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # 6. Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 7. Save Artifacts
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor
            )
            logging.info("Saved preprocessing object.")
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info("Data Transformation completed.")
            return data_transformation_artifact

        except Exception as e:
            raise DementiaException(e, sys)