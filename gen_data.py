#! /usr/bin/env python
"""
Produces classifications of MNIST so we have something to develop calibration
tools against.  The classifications are saved as a pandas dataframe.
"""
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def get_mnist(predicate=None, shuffle=True):
    """Load, split, and shuffle the data.

    Returns:
        A 4-tuple of (X_train, X_test, y_train, y_test)
    """
    mnist = fetch_mldata('MNIST original', data_home='.')
    X, y = mnist['data'], mnist['target']
    X_train, X_test = X[:60_000], X[60_000:]
    y_train, y_test = y[:60_000], y[60_000:]
    if predicate is not None:
        y_train = predicate(y_train)
        y_test = predicate(y_test)
    if shuffle:
        # Some models need shuffling
        np.random.seed(42)
        indices = np.random.permutation(60_000)
        X_train = X_train[indices]
        y_train = y_train[indices]
    return X_train, X_test, y_train, y_test


def predict_proba(clf, X_test):
    """Produce the probability vector for clf on X_test"""
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob = clf.decision_function(X_test)
        prob = (prob - prob.min()) / (prob.max() - prob.min())
    return prob


if __name__ == '__main__':
    # We create an even detector so that we have a binary classifier to work
    # with that has a decent number of true actuals vs the total
    def evens(vec):
        return vec % 2 == 0

    X_train, X_test, y_train, y_test = get_mnist(predicate=evens)

    # note: this last one (KNeighborsClassifier) takes a while
    classifiers = (LogisticRegression(C=1., solver='lbfgs'),
                   GaussianNB(),
                   LinearSVC(),
                   RandomForestClassifier(),
                   SGDClassifier(tol=None, max_iter=5),
                   KNeighborsClassifier())

    headers = []
    results = []

    # Train and run all the classifiers
    for clf in classifiers:
        print(f'Fitting {clf.__class__.__name__}')
        clf.fit(X_train, y_train)
        headers.append(clf.__class__.__name__)
        print('Generating predictions')
        pred = clf.predict(X_test)
        print('Generating probabilities')
        prob = predict_proba(clf, X_test)
        results.extend([pred, prob])

    # Read the data into a dataframe and serialize it to disk
    rows = np.array(results).T
    columns = pd.MultiIndex.from_product(
        [headers, ['Prediction', 'Probability']],
        names=['Classifier', 'Method'],
    )
    df = pd.DataFrame(rows, columns=columns)
    df['actual'] = y_test
    df.to_pickle('results.pkl')
