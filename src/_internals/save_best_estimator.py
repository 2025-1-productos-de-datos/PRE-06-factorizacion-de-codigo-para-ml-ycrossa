"""Save the estimator with the best score."""

import os
import pickle


def save_best_estimator(estimator):

    if not os.path.exists("models"):
        os.makedirs("models")

    with open("models/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)
