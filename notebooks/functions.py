import pandas as pd
import matplotlib.pyplot as plt  # Import the plotting library
import seaborn as sns

import numpy as np


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string


import re

import json
# download nltk corpus (first time only)
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')




def missing_values_table(df, table=True):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Types
    types = df.dtypes
    types.replace({'object': 'str'}, inplace=True)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, types], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Type'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values('% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n")
    
    # Add plot
    plt.figure(figsize=(13, 6))
    sns.set(style="whitegrid", color_codes=True)
    sns.barplot(x=mis_val_table_ren_columns.index, y=mis_val_table_ren_columns["% of Total Values"], data=mis_val_table_ren_columns)
    plt.xticks(rotation=90)
    plt.title("Missing values count")
    plt.show()
    
    if table:
        return mis_val_table_ren_columns


def drop_func(df, prc, lst_col_keep):
    columns_to_drop = []  
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    for col in df.columns:
        if mis_val_percent[col] >= prc and col not in lst_col_keep:
            columns_to_drop.append(col)
    return columns_to_drop


def check_sentence_for_word(sentence, lst):
    result_list = []  
    float_values = []  

    words = sentence.split()  

    for i, word in enumerate(words):
        if word in lst:
            if i > 0:  
                before = words[i - 1]
                if is_float(before):
                    float_values.append(float(before))
            if i < len(words) - 1:  
                after = words[i + 1]
                if is_float(after):
                    float_values.append(float(after))

    if float_values:
        max_float = max(float_values)
        result_list.append(max_float)

    return first_element_or_nan(result_list)

def is_float(word):
    try:
        float(word)
        return True
    except ValueError:
        return False
    
    


def first_element_or_nan(lst):
    if lst:
        return lst[0]
    else:
        return np.nan 


def split_measurement_strings(text):
    
    pattern = r'(\d+)\s*([m²])'

    result = re.sub(pattern, r'\1 \2', text)

    return result



def preprocess_text_nltk(text):

    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('french')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    
    processed_text = split_measurement_strings(processed_text)
    # remove ponct
    exclude = set(string.punctuation)
    processed_text = ''.join(ch for ch in processed_text if ch not in exclude)
    return processed_text


