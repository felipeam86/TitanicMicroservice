#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.pipeline import TransformerMixin, BaseEstimator, make_pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, Imputer


class ColumnLabelEncoder:
    """
    http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
    """
    def __init__(self, columns=None):
        # array of column names to encode
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class FeatureExtractor(TransformerMixin, BaseEstimator):
    """
    sklearn custom transformer that extracts a list of columns from a pd.DF.
    Its purpose is to be used inside pipelines. Otherwise, normal pandas
    column selection is more than enough.
    """
    def __init__(self, features):
        self.features = features

    def transform(self, data):
        return data[self.features]

    def fit(self, *_):
        return self


class PassengerTransformer(TransformerMixin, BaseEstimator):
    """
    Pandas friendly FeatureUnion
    """
    def __init__(self):
        self.transformer = FeatureUnion(
            transformer_list=[
                ('age', make_pipeline(FeatureExtractor(['age']), Imputer())),
                ('sex', make_pipeline(FeatureExtractor(['sex']), ColumnLabelEncoder(['sex']))),
                ('numerical', FeatureExtractor(['pclass', 'sibsp', 'parch', 'fare'])),
            ]
        )
        self.columns = ['age', 'sex', 'pclass', 'sibsp', 'parch', 'fare']

    def transform(self, X):
        df = pd.DataFrame(
            data=self.transformer.transform(X),
            columns=self.columns
        )
        return df

    def fit(self, X, y=None):
        return self.transformer.fit(X)
