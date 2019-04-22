import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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
