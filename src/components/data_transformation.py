import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class DataTransformationConfig:
    artifact_dir: str = os.path.join(artifact_folder)
    transformed_train_file_path: str = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path: str = os.path.join(artifact_dir, 'test.npy')
    transformed_object_file_path: str = os.path.join(artifact_dir, 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, feature_store_file_path: str):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        """
        Load and preprocess data from the feature store file.
        """
        try:
            data = pd.read_csv(feature_store_file_path)
            logging.info(f"Data loaded successfully with shape: {data.shape}")

            # Validate the existence of the target column
            if "Good/Bad" not in data.columns:
                raise ValueError("Target column 'Good/Bad' not found in dataset.")
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)

            logging.info(f"Columns in dataset: {list(data.columns)}")
            logging.info(f"First few rows of the dataset:\n{data.head()}")

            return data
        except Exception as e:
            logging.error("Error loading data: %s", str(e))
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        Create and return a preprocessing pipeline for data transformation.
        """
        try:
            # Steps for preprocessing: imputing missing values and scaling
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(steps=[imputer_step, scaler_step])
            logging.info("Preprocessing pipeline created successfully.")
            return preprocessor
        except Exception as e:
            logging.error("Error creating data transformer pipeline: %s", str(e))
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """
        Execute the data transformation pipeline: load, preprocess, split, and save data.
        """
        logging.info("Starting data transformation process.")
        try:
            # Load and validate the dataset
            dataframe = self.get_data(self.feature_store_file_path)
            logging.info(f"Loaded data with shape: {dataframe.shape}")

            if TARGET_COLUMN not in dataframe.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

            # Feature matrix (X) and target array (y)
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)
            logging.info(f"Feature matrix shape: {X.shape}, Target array shape: {y.shape}")

            # Handle non-numeric columns
            non_numeric_columns = X.select_dtypes(include=['object', 'string']).columns
            if not non_numeric_columns.empty:
                logging.warning(f"Non-numeric columns detected: {list(non_numeric_columns)}")
                for col in non_numeric_columns:
                    if X[col].str.isnumeric().all():
                        X[col] = pd.to_numeric(X[col])
                    else:
                        X.drop(columns=col, inplace=True)
                        logging.warning(f"Dropped non-numeric column: {col}")

            # Split the data into training and testing sets
            if len(X) < 2:
                raise ValueError("Insufficient data for splitting into training and testing sets.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data successfully split into training and testing sets.")

            # Apply preprocessing
            preprocessor = self.get_data_transformer_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            logging.info("Data preprocessing completed.")

            # Save the preprocessor object
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            # Combine features and labels into arrays and save them
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]
            logging.info("Transformed data arrays created successfully.")

            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            logging.error("Error in data transformation: %s", str(e))
            raise CustomException(e, sys) from e


# import sys
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler, FunctionTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# from src.constant import *
# from src.exception import CustomException
# from src.logger import logging
# from src.utils.main_utils import MainUtils
# from dataclasses import dataclass


# @dataclass
# class DataTransformationConfig:
#     artifact_dir = os.path.join(artifact_folder)
#     transformed_train_file_path = os.path.join(artifact_dir,'train.npy')
#     transformed_test_file_path = os.path.join(artifact_dir,'test.npy')
#     transformed_object_file_path = os.path.join(artifact_dir,'preprocessor.pkl')


# class DataTransformation:
#     def __init__(self,feature_store_file_path):
#         self.feature_store_file_path = feature_store_file_path

#         self.data_transformation_config = DataTransformationConfig()

#         self.utils = MainUtils()

#     @staticmethod
#     def get_data(feature_store_file_path: str) ->pd.DataFrame:

#         try:

#             data = pd.read_csv(feature_store_file_path)

#             data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)

#             return data
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     def get_data_transformer_object(self):

#         try:

#             imputer_step = ('imputer',SimpleImputer(strategy='constant', fill_value=0))
#             #scaler_step = ('scaler',StandardScaler())
#             scaler_step = ('scaler',RobustScaler())


#             preprocessor = Pipeline(
#                 steps=[
#                     imputer_step,
#                     scaler_step
#                 ]
#             )

#             return preprocessor
#         except Exception as e:
#             raise CustomException(e,sys)
    
#     def initiate_data_transformation(self):

#         logging.info("Entered initiate data transformation method of data transfomration class")

#         try:
#             dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)
#             logging.info(f"Loaded data with shape: {dataframe.shape}")

#             if TARGET_COLUMN not in dataframe.columns:
#                 raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")
   
#             X=dataframe.drop(columns= TARGET_COLUMN)
#             y= np.where(dataframe[TARGET_COLUMN]==-1,0,1)

#             logging.info(f"Feature matrix shape: {X.shape}, Target array shape: {y.shape}")

#             if len(X) < 2:
#                 raise ValueError("Insufficient data for splitting into training and testing sets.")
#             X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#             logging.info("Split data into training and testing sets.")

#             preprocessor = self.get_data_transformer_object()

#             X_train_scaled = preprocessor.fit_transform(X_train)
#             X_test_scaled = preprocessor.transform(X_test)

#             preprocessor_path = self.data_transformation_config.transformed_object_file_path
#             os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

#             self.utils.save_object(file_path= preprocessor_path, obj= preprocessor)
#             logging.info(f"Preprocessor saved at: {preprocessor_path}")

#             train_arr = np.c_[X_train_scaled, np.array(y_train)]
#             test_arr = np.c_[X_test_scaled, np.array(y_test)]
#             logging.info("Transformed data arrays created.")
#             return (train_arr,test_arr,preprocessor_path)
        
#         except Exception as e:
#             logging.error("Error in initiate_data_transformation: %s", str(e))
#             raise CustomException(e,sys) from e
