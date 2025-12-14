import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_doctor_thor.keras") # Modelo Thor
DATASET_DIR = os.path.join(BASE_DIR, "data", "val") # Usamos VALIDACIÓN, no Train

IMG_SIZE = (224, 224)
# Asegurarse de que este orden sea idéntico al del entrenamiento (alfabético)
CLASS_NAMES = ['deficiencia', 'fusario', 'sanas'] 

def evaluate_thor():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No encuentro el modelo: {MODEL_PATH}")
        return

    print(f"🔄 Cargando modelo Thor: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    y_true = []
    y_pred = []
    
    print("📸 Generando predicciones sobre el set de validación...")
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"   - Analizando clase real: {class_name}...")
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        for f in files:
            img_path = os.path.join(class_dir, f)
            try:
                # Preprocesamiento EXACTO de EfficientNet (0-255)
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.float32)
                
                # ¡IMPORTANTE! NO dividimos por 255.0 aquí.
                # EfficientNet tiene una capa interna de normalización.
                
                img_array = np.expand_dims(img_array, axis=0)
                
                # Inferencia
                prediction = model.predict(img_array, verbose=0)
                predicted_idx = np.argmax(prediction)
                
                y_true.append(class_idx)
                y_pred.append(predicted_idx)
                
            except Exception as e:
                print(f"Error en {f}: {e}")

    # Generar Matriz
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.xlabel('Lo que dijo la IA')
    plt.ylabel('Lo que es en Realidad')
    plt.title('Matriz de Confusión - Modelo Thor (EfficientNet)')
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_thor.png'))
    print("\n💾 Imagen guardada: confusion_matrix_thor.png")
    
    print("\n📊 --- REPORTE DE CLASIFICACIÓN ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

if __name__ == "__main__":
    evaluate_thor()