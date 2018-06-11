import pandas as pd
from sklearn.metrics import brier_score_loss


def load_results(fname='results.pkl'):
    """Load up the pickled data and sorts it by brier score loss.  Returns a
    3-tuple of (predictions, probabilities, actual values).
    """
    raw = pd.read_pickle(fname)
    actual = raw.actual.copy()
    preds = raw.xs('Probability', level='Method', axis=1)
    probs = raw.xs('Probability', level='Method', axis=1)
    indices = (probs
               .apply(lambda v: brier_score_loss(actual, v))
               .sort_values()
               .index)
    preds = preds.reindex(indices, axis=1)
    probs = probs.reindex(indices, axis=1)
    return preds, probs, actual
