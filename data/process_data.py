"""
Process Data
Project: Disaster Response Pipeline

Sample Script Execution:
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

Arguments:
    1) Path disaster messages csv (data/disaster_messages.csv)
    2) Path to disaster categories csv (data/categories.csv)
    3) Path to SQLite destination database (data/DisasterResponse.db)
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories datasets.

    Args:
        messages_filepath (str): Filepath of the messages dataset.
        categories_filepath (str): Filepath of the categories dataset.

    Returns:
        pandas.DataFrame: Merged dataset.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged dataset.

    Args:
        df (pandas.DataFrame): Merged dataset.

    Returns:
        pandas.DataFrame: Cleaned dataset.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of 'categories'
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    #changing values equal to 2 to 1
    categories['related'] = categories['related'].map(lambda x: 1 if x==2 else x)      

    # drop the original categories column from 'df'
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataset to an SQLite database.

    Args:
        df (pandas.DataFrame): Cleaned dataset.
        database_filename (str): Filepath of the SQLite database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)


def main():
    """
    Main function to run the ETL pipeline.

    Reads the filepaths of the messages and categories datasets,
    cleans and merges the datasets, and saves the cleaned dataset to a database.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories datasets as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument.\n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')


if __name__ == '__main__':
    main()
