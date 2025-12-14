import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import os
import datetime

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Modelo previo (el mejor que guardó el script anterior)
PREV_MODEL_PATH = os.path.join(MODELS_DIR, "mobilenet_v2_best.keras")
NEW_MODEL_PATH = os.path.join(MODELS_DIR, "mobilenet_v2_finetuned.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
# Tasa de aprendizaje MUY BAJA (Vital para Fine-Tuning)
FINE_TUNE_LR = 1e-5 

def run_finetuning():
    if not os.path.exists(PREV_MODEL_PATH):
        print("❌ No encuentro el modelo previo. Ejecuta src/train.py primero.")
        return

    print(f"🔄 Cargando modelo base: {PREV_MODEL_PATH}")
    model = load_model(PREV_MODEL_PATH)

    # 1. ESTRATEGIA DE DESCONGELAMIENTO
    # Como usamos la API funcional, las capas de MobileNet están 'planas' en model.layers.
    # Total capas aprox: 155 (MobileNet) + 4 (Nuestras) = ~159
    
    total_layers = len(model.layers)
    print(f"🔍 El modelo tiene un total de {total_layers} capas.")
    
    # Queremos re-entrenar las últimas ~50 capas (parte alta de MobileNet + nuestro clasificador)
    TRAINABLE_LAYERS = 50
    freeze_until = total_layers - TRAINABLE_LAYERS
    
    # Primero descongelamos todo para resetear estados
    model.trainable = True
    
    # Luego congelamos desde la 0 hasta la N
    print(f"❄️ Congelando las primeras {freeze_until} capas...")
    for layer in model.layers[:freeze_until]:
        layer.trainable = False
        
    print(f"🔥 Las últimas {TRAINABLE_LAYERS} capas se entrenarán con LR={FINE_TUNE_LR}")

    # 2. RE-COMPILAR (Necesario después de cambiar trainable)
    model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Imprimir resumen breve
    # model.summary() # Descomentar para ver la lista gigante (arquitectura del modelo)

    # 3. DATOS (Mismo generador)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=2,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("🔄 Generando flujos de datos...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )

    # 4. CALLBACKS
    checkpoint = ModelCheckpoint(NEW_MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # Paciencia alta porque el Fine-Tuning mejora muy lento
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    
    # Nuevo log para TensorBoard
    log_dir = os.path.join(LOGS_DIR, "fit", "finetune_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)

    # 5. ENTRENAR
    print("🚀 Iniciando Fine-Tuning...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, tensorboard]
    )
    
    print(f"✅ Fine-Tuning terminado. Modelo experto guardado en: {NEW_MODEL_PATH}")

if __name__ == "__main__":
    run_finetuning()