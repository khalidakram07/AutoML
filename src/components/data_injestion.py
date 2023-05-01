"""
Module that performs data ingestion from a CSV file and prepares it for model training. The module uses the pandas
library to read the CSV file, split the data into train and test sets, and store the resulting data sets in CSV files.

Attributes:
    train_data_path (str): The path to the CSV file that contains the training data.
    test_data_path (str): The path to the CSV file that contains the test data.
    raw_data_path (str): The path to the CSV file that contains the raw data.

Classes:
    DataIngestion: Class that performs data ingestion from a CSV file and prepares it for model training.

Methods:
    __init__(self): Initializes the DataIngestion object with default configuration values.
    initiate_data_ingestion(self): Reads the CSV file, splits the data into train and test sets, and stores the resulting
        data sets in CSV files.
"""
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass

class DataIngestionConfig:
    """
    This code defines a dataclass DataIngestionConfig with three attributes: train_data_path, test_data_path, 
    and raw_data_path. The os.path.join() function is used to join the specified path components using 
    the appropriate separator for the current operating system.
    """
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This code defines a class DataIngestion with a method initiate_data_ingestion() which performs the data ingestion process. 
        First, it reads the CSV file specified by the path 'notebook\data\stud.csv' as a pandas dataframe. 
        Then, it creates a directory where the train data will be stored using os.makedirs() 
        and saves the dataframe as raw data. It then splits the data into train and test sets using train_test_split() 
        from sklearn.model_selection. The train and test sets are saved as CSV files using to_csv() from pandas. 
        Finally, the method returns the paths of the train and test data. 
        If any exception is raised during the process, a CustomException is raised.
        """
        logging.info("Entered the data ingestion method or component")
        
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))