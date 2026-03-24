import pandas as pd
import os # Used for file & directory operations
import logging # Used to track program execution (logs)
import yaml # Used to read configuration file (params.yaml) , Helps avoid hardcoding values (like test_size)
from sklearn.model_selection import train_test_split
import sys 
import yaml

def load_params(params_path:str)->dict:
    """load parameters from params.yaml file """
    try:
        with open(params_path,"r") as file:
            params = yaml.safe_load(file)
        logger.debug("parameters load from %s",params_path)
        return params
    except FileNotFoundError:
        logger.error("file not found :%s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error : %s",e)
        raise

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# def load_params(params_path : str) -> dict:
#     """Load parameters from yaml file"""
#     try:
#         with open(params_path,"r") as file:

def load_data(data_url:str)-> pd.DataFrame:
    """load data from csv fiel"""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file :%s",e)
        raise
    except Exception as e:
        logger.error("unexcepted error occured during loading the data: %s", e)

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """preprocess the data """
    try:
        df.drop(columns=["Unnamed: 2",'Unnamed: 3',"Unnamed: 4"], inplace=True)
        df.rename(columns={"v1":"target", "v2":"text"}, inplace= True)
        logger.debug("data preprocessing completed")
        return df
    except KeyError as e:
        logger.error("missing column in dataframe %s",e)
        raise
    except Exception as e:
        logger.error("unexcepted error during preprocessing %s",e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)-> None:
    """save the train test datasets"""
    try:
        raw_data_path = os.path.join(data_path,"raw")
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug("train and test data save to %s",raw_data_path)
    except Exception as e:
        logger.error("unexpected error occured while saving the data :%s",e)


def main():
    try:
        params = load_params(params_path="params.yaml")
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.20 before not create params.yaml
        data_path = "https://raw.githubusercontent.com/virendrathakre03/mlops_end_to_end_pipeline/refs/heads/main/experiments/spam.csv"
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=0.20,random_state=2)
        save_data(train_data,test_data,data_path="./data")
    except Exception as e:
        logger.error("failed to complete the data ingestion step :%s",e)
        print(f"Error : {e}")

if __name__ == "__main__":
    main()



