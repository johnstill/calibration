import pandas as pd


def load_probs(fname='results.pkl'):
    """Load up the pickled data and return just the calculated probabilities"""
    raw = pd.read_pickle(fname)
    actual = raw.actual.copy()
    probs = raw.xs('Probability', level='Method', axis=1).copy()
    probs.loc[:, 'actual'] = actual
    return probs
