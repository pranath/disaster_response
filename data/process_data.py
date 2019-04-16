import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy as db
import os


def load_data(messages_filepath, categories_filepath):
    """
    Description: Loads csv files of messages and categories into a merged pandas dataframe

    Args:
        - messages_filepath: Name & path of csv file of message data
        - categories_filepath: Name & path of csv file of categories data

    Returns:
        - df: Dataframe of merged messages & categories
    """
    # Load messages
    messages = pd.read_csv(messages_filepath)
    # Load categories
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Description: Cleans data in dataframe & transforms category fields

    Args:
        - df: Dataframe of messages and categories

    Returns:
        - df: Cleaned dataframe
    """

    # Split categories into separate columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[[0]]
    # use this row to extract a list of new column names for categories.
    category_colnames = [category.split('-')[0] for category in row.values[0]]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Replace all 2 with 1
    categories = categories.replace(2, 1)
    # Drop column with no examples
    categories = categories.drop(['child_alone'], axis=1)
    # Replace categories column in df with new category columns
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    # drop duplicates
    df = df.drop_duplicates()
    # Return cleaned df
    return df

def save_data(df, database_filename):
    """
    Description: Save the current data to an sqlite database file on disk

    Args:
        - df: Dataframe of message data
        - database_filename: Filename to save data to

    Returns:
        Nothing
    """

    # Delete previous db
    os.remove(database_filename)
    # Save current version of database
    engine = db.create_engine('sqlite:///' + str(database_filename))
    # Get db name minus .db
    name = database_filename[:-3]
    df.to_sql('DisasterResponse', engine, index=False)


def main():
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
