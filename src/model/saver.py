# src/model/saver.py
import joblib
from tensorflow.keras.models import Model

def save_model(model: tensorflow.keras.models, file_path: str)-> None:
    """
    Guarda el modelo en el disco en el formato TensorFlow SavedModel.
    
    Parameters:
        model (Model): Modelo Keras entrenado.
        file_path (str): Ruta donde se guardará el modelo.
    """
    joblib.dump(model, file_path)
    print(f"Modelo guardado en {file_path}")
