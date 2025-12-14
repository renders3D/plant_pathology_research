import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
MODEL_PATH = os.path.join(BASE_DIR, "models", "thesis_model_best.keras")
ERROR_GALLERY_DIR = os.path.join(BASE_DIR, "error_gallery")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def audit_errors():
    if not os.path.exists(MODEL_PATH):
        print("❌ No encuentro el modelo de tesis.")
        return

    # 1. Cargar Modelo
    print(f"🔄 Cargando modelo: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # 2. Generador de Validación (SIN SHUFFLE para mantener orden)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )

    # Obtener nombres de clases y archivos
    class_names = list(val_generator.class_indices.keys())
    filenames = val_generator.filenames
    true_classes = val_generator.classes

    # 3. Predicción Masiva
    print("📸 Ejecutando inferencia sobre validación...")
    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # 4. Análisis de Errores
    if os.path.exists(ERROR_GALLERY_DIR):
        shutil.rmtree(ERROR_GALLERY_DIR) # Limpiar galería anterior
    os.makedirs(ERROR_GALLERY_DIR)

    errors_count = 0
    print("\n🕵️‍♂️ Generando Galería de Errores...")
    
    for i in range(len(filenames)):
        if predicted_classes[i] != true_classes[i]:
            errors_count += 1
            
            # Info del error
            fname = filenames[i]
            true_label = class_names[true_classes[i]]
            pred_label = class_names[predicted_classes[i]]
            confidence = np.max(predictions[i])
            
            # Copiar imagen a la galería de errores
            # Formato nombre: "REAL_vs_PRED_confianza_nombreOriginal.jpg"
            src_path = os.path.join(VAL_DIR, fname)
            
            dst_name = f"REAL-{true_label}_VS_PRED-{pred_label}_conf{confidence:.2f}_{os.path.basename(fname)}"
            dst_path = os.path.join(ERROR_GALLERY_DIR, dst_name)
            
            shutil.copy(src_path, dst_path)

    print(f"\n❌ Se encontraron {errors_count} errores de {len(filenames)} imágenes.")
    print(f"📂 Revisa la carpeta: {ERROR_GALLERY_DIR}")
    print("   Ahí verás las imágenes que confunden al modelo.")

    # 5. Generar Matriz de Confusión Visual
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión (Validación)')
    plt.ylabel('Etiqueta REAL (Carpeta)')
    plt.xlabel('Predicción del Modelo')
    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix_val.png"))
    print("📊 Matriz de confusión guardada como 'confusion_matrix_val.png'")

    # Reporte de texto
    print("\n--- REPORTE DETALLADO ---")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

if __name__ == "__main__":
    audit_errors()