from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def process_data(data: pd.DataFrame, columns_to_impute: list, target_column: str = None, apply_pca: bool = False, pca_variance_threshold: float = 0.90) -> tuple[pd.DataFrame, pd.Series]:
    """
    Processes the dataset by imputing missing values, cleaning categorical variables,
    removing unnecessary columns, scaling numeric variables, applying PCA (if enabled), and handling outliers.

    Args:
        data (pd.DataFrame): The input dataset.
        columns_to_impute (list): List of numeric columns to impute missing values for.
        target_column (str, optional): The name of the target column. Defaults to None.
        apply_pca (bool): Whether to apply PCA for dimensionality reduction. Defaults to False.
        pca_variance_threshold (float): Cumulative variance threshold for PCA. Defaults to 0.90.

    Returns:
        tuple: 
            - pd.DataFrame: Processed features (X).
            - pd.Series or None: Target column (y), if specified.
    """
    # Imputar valores faltantes
    imputer = SimpleImputer(strategy='mean')
    data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

    # Eliminar columnas innecesarias
    columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # Sustituir valores no válidos en estado civil
    INVALID_MARITAL_STATUSES = ['Absurd', 'YOLO']
    REPLACEMENT_STATUS = 'Other'
    data['Marital_Status'] = data['Marital_Status'].replace(INVALID_MARITAL_STATUSES, REPLACEMENT_STATUS)

    # Separar target si está especificado
    target = data[target_column] if target_column else None
    if target_column:
        data = data.drop(columns=[target_column])

    # Identificar columnas numéricas y categóricas
    categorical_columns = data.select_dtypes(include=['category', 'object']).columns
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ]
    )

    # Aplicar el preprocesador
    X_processed = preprocessor.fit_transform(data)

    # Aplicar PCA si está habilitado
    if apply_pca:
        pca = PCA()
        X_pca = pca.fit_transform(X_processed)

        # Calcular la varianza explicada acumulada
        explained_variance = pca.explained_variance_ratio_.cumsum()

        # Determinar el número de componentes necesarios
        n_components = next(i for i, var in enumerate(explained_variance) if var >= pca_variance_threshold) + 1
        print(f"Se necesitan {n_components} componentes para capturar al menos el {pca_variance_threshold * 100}% de la variabilidad.")

        # Aplicar PCA con el número óptimo de componentes
        pca = PCA(n_components=n_components)
        X_processed = pca.fit_transform(X_processed)

    return X_processed, target
