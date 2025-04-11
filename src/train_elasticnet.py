import argparse

from sklearn.linear_model import ElasticNet

from _internals import train_estimator


def train_elasticnet_model(alpha, l1_ratio):

    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
    train_estimator(estimator)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an ElasticNet model.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Regularization strength (default: 0.5)",
    )
    parser.add_argument(
        "--l1_ratio", type=float, default=0.5, help="L1 ratio (default: 0.5)"
    )

    args = parser.parse_args()

    train_elasticnet_model(alpha=args.alpha, l1_ratio=args.l1_ratio)
