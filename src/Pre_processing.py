import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "log" directory exists
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# setting up logger
logger = logging.getLogger("data pre-processing")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,"data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formattor = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formattor)
file_handler.setFormatter(formattor)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """transform the text by lowercasing, tokenizing, removing stopwords and punctuation and stemming"""
    ps = PorterStemmer()
    # convert to lowercase
    text = text.lower()

    # tokenize the text
    text = nltk.word_tokenize(text)

    # remove non alpha numeric tokens 
    text = [word for word in text if word.isalnum()]

    # remove stopwords and punctuation
    # text = [word for word in text if word not in stopwords.words('english') AND word not in string.punctuation ]
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
   
    # stem the words 
    text = [ps.stem(word) for word in text]

    # join the tokens back into the single string
    return " ".join(text)

# def preprocess_df(df,text_column = 'text',target_column ='target'):
#     """ 
#     preprocess the dataframe by encoding the target column , removing duplicates and transforming the text column.
#     """
#     try:
#         logger.debug('starting  preprocessing the dataframe')
#         # Encode the DataFrame
#         encoder = LabelEncoder()
#         df[target_column]= encoder.fit_transform(df['target'])
#         logger.debug('target column encoded ')

#         # remove duplicate columns 
#         df = df.drop_duplicates(keep = 'first')
#         logger.debug("Duplicates removed")

#         # apply text transformation to the specified text columns 
#         df[text_column]= df[text_column].apply(transform_text)
#         logger.debug('text column transformed ')
#         return df
#     except KeyError as e :
#         logger.error('column not found : %s',e)
#         raise
#     except Exception as e:
#         logger.error('error during the text normalization :%s',e)
#         raise
def preprocess_df(df, text_column='text', target_column='target'):
    try:
        logger.debug('starting preprocessing the dataframe')

        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('target column encoded')

        df = df.drop_duplicates()
        logger.debug("Duplicates removed")

        df[text_column] = df[text_column].apply(transform_text)
        logger.debug('text column transformed')

        logger.debug("Final shape: %s", df.shape)

        return df

    except KeyError as e:
        logger.error('column not found: %s', e)
        raise

    except Exception as e:
        logger.error('error during preprocessing: %s', e)
        raise
def main(text_column= 'text',target_column = 'target'):
    """ Main function to load raw data ,preprocess it and save the processed data
    """
    try:
        # fetch the data from data/raw
        # train_data = pd.read_csv(r'.\data\raw\train.csv')
        # test_data = pd.read_csv(r'.\data\raw\test.csv')
        train_data = pd.read_csv('.\\data\\raw\\train.csv')
        test_data = pd.read_csv('.\\data\\raw\\test.csv')
        logger.debug('data loaded sucessfully')

        # transform the data 
        train_processed_data = preprocess_df(train_data,text_column,target_column)
        test_processed_data = preprocess_df(test_data,text_column,target_column)

        # store the data into data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"), index = False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"), index = False)

        logger.debug('processed data saved to %s',data_path)
    
    except FileNotFoundError as e:
        logger.error('file not found $s',e)
    except pd.errors.EmptyDataError as e :
        logger.error('No data %s',e)
    except Exception as e:
        logger.error('failed to complete the data transformation process %s',e)
        print(f"Error : {e}")

if __name__ == '__main__':
    main()