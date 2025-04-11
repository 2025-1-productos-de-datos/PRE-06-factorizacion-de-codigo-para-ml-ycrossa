"""Load the model from disk"""

import os
import pickle


def load_estimator_from_disk():

    if not os.path.exists("models"):
        return None

    with open("models/estimator.pkl", "rb") as file:
        estimator = pickle.load(file)

    return estimator
