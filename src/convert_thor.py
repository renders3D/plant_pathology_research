import tensorflow as tf
import os
import numpy as np

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_doctor_thor.keras")
TFLITE_PATH = os.path.join(BASE_DIR, "models", "plant_doctor_thor.tflite")

def convert_to_tflite():
    if not os.path.exists(MODEL_PATH):
        print("❌ No encuentro el modelo Thor.")
        return

    print(f"🔄 Cargando modelo Keras: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Convertidor
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizaciones para Edge (Drone)
    # Esto reduce el tamaño y mejora la velocidad en CPU ARM
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("⚙️ Convirtiendo a TFLite (con optimización dinámica)...")
    tflite_model = converter.convert()

    # Guardar
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    # Reporte de tamaño
    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"✅ Conversión exitosa.")
    print(f"💾 Guardado en: {TFLITE_PATH}")
    print(f"📏 Tamaño final: {size_mb:.2f} MB")

    print("\n⚠️ NOTA IMPORTANTE PARA EL DRONE/API:")
    print("Este modelo usa EfficientNetB0.")
    print("1. Input Shape: (224, 224)")
    print("2. Preprocesamiento: NO DIVIDIR POR 255. Pasar valores crudos (0-255).")

if __name__ == "__main__":
    convert_to_tflite()