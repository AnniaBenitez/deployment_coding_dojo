from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
import numpy as np

def data_splitter(data: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets, balances the classes by downsampling the majority class, 
    and returns the balanced train and test datasets.

    Args:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series:
            - X_train: Features for the training set.
            - X_test: Features for the testing set.
            - y_train: Target for the training set.
            - y_test: Target for the testing set.
    """
    # Separar X (características) e y (target)
    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]

    # Combinar X e y en un solo DataFrame para balancear las clases
    combined_data = pd.concat([X, y], axis=1)

    # Separar clases mayoritaria y minoritaria
    class_majority = combined_data[combined_data[target_column] == 0]
    class_minority = combined_data[combined_data[target_column] == 1]

    # Submuestreo de la clase mayoritaria para igualar a la clase minoritaria
    class_majority_downsampled = resample(
        class_majority,
        replace=False,
        n_samples=len(class_minority),
        random_state=random_state
    )

    # Combinar clases balanceadas
    balanced_data = pd.concat([class_majority_downsampled, class_minority])

    # Mezclar aleatoriamente los datos
    balanced_data = balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Actualizar X e y con los datos balanceados
    X_balanced = balanced_data.drop(columns=[target_column])
    y_balanced = balanced_data[target_column]

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
