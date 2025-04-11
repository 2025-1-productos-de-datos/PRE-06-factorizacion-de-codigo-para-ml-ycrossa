"""Train a KNeighborsRegressor model"""

import argparse

from sklearn.neighbors import KNeighborsRegressor

from _internals import train_estimator


def train_knn_model(n_neighbors):

    estimator = KNeighborsRegressor(n_neighbors=n_neighbors)
    train_estimator(estimator)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an ElasticNet model.")
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=5,
        help="Number of neighbors (default:5)",
    )

    args = parser.parse_args()

    train_knn_model(n_neighbors=args.n_neighbors)
