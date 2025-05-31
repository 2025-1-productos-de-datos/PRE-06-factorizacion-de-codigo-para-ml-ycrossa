# descarga de datos
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(file_path, test_size, random_state):

    df = pd.read_csv(file_path)

    # preparacion de datos
    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    # dividir los datos en entrenamiento y testing
    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return x_train, x_test, y_train, y_test
