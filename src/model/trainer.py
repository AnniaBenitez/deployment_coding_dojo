from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def train_model(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2):
    """
    Train a neural network model on the provided training dataset.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        epochs (int): Number of epochs for training. Defaults to 30.
        batch_size (int): Batch size for training. Defaults to 32.
        validation_split (float): Fraction of training data to use for validation. Defaults to 0.2.

    Returns:
        Sequential: Trained Keras model.
        History: Training history containing loss and accuracy metrics.
    """
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")

    # Crear el modelo
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compilar el modelo
    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Mostrar resumen del modelo
    model.summary()

    # Entrenar el modelo
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    return model, history
