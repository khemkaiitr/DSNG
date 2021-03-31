# import libraries
import sys
import sqlite3

from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import joblib

import pickle

def load_data(database_filepath):
    """
    Function to load data from sql database
    :param database_filepath: Database path
    :return:
        X: training data
        y: target
        categories: list of unique categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)

    X = df['message']
    y = df.iloc[:, 4:]
    categories = y.columns.values
    return X, y, categories


def tokenize(text):
    """
    Function to preprocess text data and convert them to tokens
    :param text: Input text
    :return: Clean tokens
    """
    # Remove pucutation, and normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Split the text into tokens
    word_tokens = word_tokenize(text)
    # Remove stop words
    words = [w for w in word_tokens if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Function to build model and apply grid serach to get the best parameters
    :param X: Training data
    :param y: Target data
    :return: model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # grid search parameters
    parameters = {
#         'vect__max_df': (0.75, 1.0),
#         'tfidf__norm':('l1', 'l2'),
#         'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__learning_rate': [0.1, .2]
    }
    # create grid search object
    grid_model = GridSearchCV(pipeline, parameters, verbose=1)
    return grid_model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate model performance
    :param model: training model
    :param X_test: Test data
    :param Y_test: test target
    :param category_names: unique categories
    :return: None
    """
    y_pred = model.predict(X_test)
    # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    function to save trained model
    :param model: model object
    :param model_filepath: path to store model file
    :return: None
    """
#     joblib.dump(model, model_filepath)
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
