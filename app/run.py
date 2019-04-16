import json
import plotly
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sqlalchemy import create_engine

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



app = Flask(__name__)

def tokenize(text):
    """
    Description: Takes a string, tokenises, lemmatises & strips white space to return a list of cleaned tokens

    Args:
        - text: Text string

    Returns:
        - clean_tokens: tokenised version fo string
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Extract data for genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Extract data for category counts
    categories_df = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = list(categories_df.columns.values)
    category_totals = categories_df.apply(pd.Series.value_counts)
    category_totals = category_totals.iloc[1]

    # Extract data for heatmap
    coocc_matrix = categories_df.T.dot(categories_df)
    coocc_matrix = coocc_matrix.values

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_totals
                )
            ],

            'layout': {
                'title': 'Message category usage counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category name"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names,
                    z=coocc_matrix
                )
            ],

            'layout': {
                'title': 'Message category co-occurance heatmap',
                'yaxis': {
                    'title': "Category name"
                },
                'xaxis': {
                    'title': "Category name"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
