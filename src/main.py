import sys
import os

# Agrega la raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar funciones de data
from src.data.data_loader import load_data
from src.data.data_processor import process_data
from src.data.data_splitter import data_splitter

#importar funciones de model
from src.model.evaluator import evaluate_model
from src.model.trainer import train_model
from src.model.saver import save_model

def main():
    print("Programa en ejecución")
    
    # Cargar los datos
    data = load_data(file_path="data/raw/marketing_campaign.csv")
    data.columns = data.columns.str.strip()
    
    # Procesar los datos, aplicando PCA
    X, y = process_data(
        data=data, 
        columns_to_impute=['Income'], 
        target_column='Response', 
        apply_pca=True, 
        pca_variance_threshold=0.90
    )

    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = data_splitter(X, y)
    print("Datos divididos con éxito.")
    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}, {y_train.shape}")
    print(f"Tamaño del conjunto de prueba: {X_test.shape}, {y_test.shape}")
    
    #Entrenar el modelo
    model, history = train_model(X_train, y_train)
    
    # Evaluar el modelo
    evaluation_results = evaluate_model(model, X_test, y_test)
    print("Resultados de evaluación:", evaluation_results)
    
    # Guardar el modelo
    save_model(model, file_path="models/trained_model.joblib")
    

if __name__ == "__main__":
    main()
