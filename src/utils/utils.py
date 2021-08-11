import os
import pandas as pd
import numpy as np

# Given error metric to calculate RMSPE
def metric(preds, actuals):
    """
    Calculate Root Mean Squared Percentage Error
    input: yhat, ytest
    """
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])