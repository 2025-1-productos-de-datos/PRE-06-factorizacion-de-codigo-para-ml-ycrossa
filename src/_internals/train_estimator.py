"""Train a preconfigurated estimator"""

from .eval_metrics import eval_metrics
from .load_data import load_data
from .load_estimator_from_disk import load_estimator_from_disk
from .make_train_test_split import make_train_test_split
from .report import report
from .save_best_estimator import save_best_estimator


def train_estimator(estimator):
    """Train the estimator and return the trained estimator."""

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)

    estimator.fit(x_train, y_train)
    mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))
    report(estimator, mse, mae, r2)
    best_estimator = load_estimator_from_disk()

    if best_estimator is None or estimator.score(x_test, y_test) > best_estimator.score(
        x_test, y_test
    ):
        best_estimator = estimator

    save_best_estimator(best_estimator)
