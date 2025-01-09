from tensorflow.keras.models import Model

def evaluate_model(model: Model, X_test, y_test):
    """
    Evalúa el modelo entrenado en el conjunto de prueba.
    
    Parameters:
        model (Model): Modelo Keras entrenado.
        X_test (numpy.ndarray): Datos de entrada de prueba.
        y_test (numpy.ndarray): Etiquetas de prueba.
    
    Returns:
        dict: Resultados de evaluación que incluyen pérdida y precisión.
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Pérdida en el conjunto de prueba: {loss}")
    print(f"Precisión en el conjunto de prueba: {accuracy}")
    return {"loss": loss, "accuracy": accuracy}