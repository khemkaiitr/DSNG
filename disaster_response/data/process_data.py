import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to read message and category csv data
    :param messages_filepath: File path for message data
    :param categories_filepath: Csv file path for categories data
    :return: Data frame for messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Function to perform cleaning operations on the dataframe. This function:
        Expands category column and get the category name
        Use this category name to hot encode category column
        removes the duplicates
    :param df: pandas data frame
    :return: df: cleaned data frame
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates.
    df.drop_duplicates(keep='first', inplace=True)
    return df


def save_data(df, database_filename):
    """
    Function to save data in sql data base
    :param df: Cleaned pandas data frame
    :param database_filename: SQL database filename
    :return: None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)
  


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