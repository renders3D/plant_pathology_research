import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import cleanlab
from cleanlab.filter import find_label_issues
import pandas as pd
import shutil

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ¿Qué queremos auditar? ¿El examen (val) o el libro de texto (train)?
# Vamos a auditar TRAIN, porque si el libro está mal, el alumno aprende mal.
DATASET_TO_AUDIT = os.path.join(BASE_DIR, "data", "train") 
MODEL_PATH = os.path.join(BASE_DIR, "models", "thesis_model_best.keras")
OUTPUT_CSV = os.path.join(BASE_DIR, "label_issues.csv")
BAD_LABELS_DIR = os.path.join(BASE_DIR, "suspected_label_errors")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def run_cleanlab_audit():
    if not os.path.exists(MODEL_PATH):
        print("❌ Modelo no encontrado.")
        return

    print(f"🔄 Cargando modelo: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # 1. Generador (Sin Shuffle, para mantener orden con los nombres de archivo)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    print(f"📂 Leyendo dataset: {DATASET_TO_AUDIT}")
    generator = datagen.flow_from_directory(
        DATASET_TO_AUDIT,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # VITAL: No barajar para poder mapear predicciones a archivos
    )

    # 2. Obtener Probabilidades (Predicciones)
    print("🧠 Extrayendo probabilidades del modelo (esto puede tardar)...")
    # pred_probs será una matriz [N_imagenes, 3_clases] con % de confianza
    pred_probs = model.predict(generator, verbose=1)
    
    # Etiquetas reales (lo que dicen las carpetas)
    labels = generator.classes
    
    # 3. La Magia de Cleanlab (Confident Learning)
    print("🧹 Ejecutando Cleanlab para hallar errores de etiqueta...")
    
    # find_label_issues devuelve una máscara booleana o índices
    ranked_label_issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence', # Los errores más obvios primero
    )

    print(f"\n⚠️ Cleanlab encontró {len(ranked_label_issues)} posibles errores de etiquetado.")

    # 4. Exportar Reporte y Copiar Imágenes Sospechosas
    if os.path.exists(BAD_LABELS_DIR): shutil.rmtree(BAD_LABELS_DIR)
    os.makedirs(BAD_LABELS_DIR)
    
    report_data = []
    class_names = list(generator.class_indices.keys())

    print("💾 Guardando evidencia...")
    for idx in ranked_label_issues:
        filename = generator.filenames[idx]
        given_label = class_names[labels[idx]] # Lo que dice la carpeta
        predicted_label_idx = np.argmax(pred_probs[idx])
        predicted_label = class_names[predicted_label_idx] # Lo que dice la IA
        confidence = pred_probs[idx][predicted_label_idx]
        
        # Guardar en CSV
        report_data.append({
            "filename": filename,
            "given_label": given_label,
            "predicted_label": predicted_label,
            "model_confidence": confidence
        })
        
        # Copiar imagen para revisión visual
        # Nombre formato: PREDICHO_vs_REAL_nombre.jpg
        src = os.path.join(DATASET_TO_AUDIT, filename)
        dst_name = f"AIsays_{predicted_label}_BUTfolderIs_{given_label}_{os.path.basename(filename)}"
        dst = os.path.join(BAD_LABELS_DIR, dst_name)
        shutil.copy(src, dst)

    # Guardar CSV
    df = pd.DataFrame(report_data)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Auditoría finalizada.")
    print(f"   📂 Imágenes sospechosas: {BAD_LABELS_DIR}")
    print(f"   📄 Reporte CSV: {OUTPUT_CSV}")
    print("\nRECOMENDACIÓN: Abre la carpeta 'suspected_label_errors'.")
    print("Si ves una imagen que la IA dice 'Fusario' pero la carpeta dice 'Sanas',")
    print("y visualmente TIENE manchas, ¡la IA tiene razón y debes mover el archivo!")

if __name__ == "__main__":
    run_cleanlab_audit()