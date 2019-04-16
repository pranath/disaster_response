import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import datetime
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import resample
import sqlite3
from sqlalchemy import create_engine
import pickle
from sklearn.externals import joblib

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """
        Description: Boolean function returns true if string contains first word as a kind of verb, false otherwise.

        Args:
            - text: text string

        Returns:
            - Boolean
        """

        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
                return False
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Description: Applies starting verb function to X and returns boolean dataframe

        Args:
            - X: Dataframe

        Returns:
            - Dataframe of boolean values
        """

        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """
    Description: Loads data from sqlite database file

    Args:
        - database_filepath: Filename & path to database file

    Returns:
        - Predictor features X dataframe, Target classes dataframe Y, target class names list target_names
    """

    # Load the db
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    target_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    X = df['message']
    Y = df[target_names]
    # Return processed values
    return X, Y, target_names


def tokenize(text):
    """
    Description: Takes a string, tokenises, lemmatises & strips white space to return a list of cleaned tokens

    Args:
        - text: Text string

    Returns:
        - clean_tokens: tokenised version fo string
    """

    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenise
    tokens = word_tokenize(text)
    # Create lemmatiser
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    # Lemmatize, standardise (lower case) & clean tokens
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Description: Creates a ML model

    Args:
        None

    Returns:
        - Model built with pipeline, specific parameters & grid search
    """

    # Define pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(GradientBoostingClassifier()))
    ])
    # Define params for grid search
    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__n_estimators': [50, 100, 200, 400],
        #'clf__min_samples_split': [2, 3, 4],
        #'clf__subsample': [0.2, 0.5, 1.0],
        #'clf__max_depth': [2, 3, 5],
        #'features__transformer_weights': (
        #    {'text_pipeline': 1, 'starting_verb': 0.5},
        #    {'text_pipeline': 0.5, 'starting_verb': 1},
        #    {'text_pipeline': 0.8, 'starting_verb': 1},
        #)
    }
    # Execute grid search on pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    # Return model
    return cv


def performance_report(actual, predicted, target_names):
    """
    Description: Takes predicted & actual class values & calculates a range of performance metrics

    Args:
        - actual: Actual target class values for data
        - predicted: Predicted target class values for data
        - target_names: List of names of target classes

    Returns:
        - Dataframe of metrics
    """

    # Init list for metrics
    metrics = []
    # For each of the target features
    for i in target_names:
        # Calculate metrics
        accuracy = accuracy_score(actual[i], predicted[i])
        precision = precision_score(actual[i], predicted[i])
        recall = recall_score(actual[i], predicted[i])
        f1 = f1_score(actual[i], predicted[i])
        # Save metrics
        metrics.append([accuracy, precision, recall, f1])

    # Convert to numpy array
    metrics = np.array(metrics)
    # Convert to dataframe
    metrics_df = pd.DataFrame(data = metrics, index = target_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
    # Add mean
    mean_acc = metrics_df['Accuracy'].mean()
    mean_pre = metrics_df['Precision'].mean()
    mean_rec = metrics_df['Recall'].mean()
    mean_f1 = metrics_df['F1'].mean()
    row = pd.Series({'Accuracy':mean_acc,'Precision':mean_pre,'Recall':mean_rec,'F1':mean_f1},name='MEAN SCORE')
    metrics_df = metrics_df.append(row)
    # Return metrics df
    return metrics_df


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description: Take a trained model, predict on test data and print performance report

    Args:
        - model: The ML model
        - X_test: Dataframe of test features
        - Y_test: Dataframe of test classes
        - category_names: List of names of test classes

    Returns:
        Nothing
    """

    # predict on test data
    Y_pred = model.predict(X_test)
    # Re-structure returned numpy array as dataframe with same column names
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred.columns = category_names
    # Evaluate the model
    print(performance_report(Y_test, Y_pred, category_names))
    # Print best params
    print("\nBest Parameters:", model.best_params_)

def save_model(model, model_filepath):
    """
    Description: Same ML model to file

    Args:
        - model: The ML model
        - model_filepath: Path and name to file to store trained ML model

    Returns:
        Nothing
    """

    # save the model to disk
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        print('Start time:')
        print(datetime.datetime.now())
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
        print('End time:')
        print(datetime.datetime.now())

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
