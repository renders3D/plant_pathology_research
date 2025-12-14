import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import os
import datetime

# --- CONFIGURACIÓN DE INGENIERÍA ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

IMG_SIZE = (224, 224) # Tamaño nativo de MobileNetV2
BATCH_SIZE = 32
EPOCHS = 30 # Le damos espacio, el EarlyStopping cortará si es necesario
LEARNING_RATE = 1e-4 # 0.0001 - Fine Tuning suave

def train_pipeline():
    # 1. GENERADORES DE DATOS + AUGMENTATION
    # Solo aplicamos Augmentation agresivo al entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Normalización 0-1 (Vital para convergencia)
        rotation_range=2,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Para validación, SOLO re-escalamos. No queremos distorsionar la "realidad" del examen.
    val_datagen = ImageDataGenerator(rescale=1./255)

    print(f"🔄 Cargando datos desde: {TRAIN_DIR}")
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f"📂 Clases detectadas: {train_generator.class_indices}")

    # 2. ARQUITECTURA (TRANSFER LEARNING)
    # Descargamos MobileNetV2 con pesos de ImageNet (El cerebro educado)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Congelamos las capas base para no destruir lo que ya sabe
    base_model.trainable = False 

    # Añadimos nuestra "Cabeza" personalizada
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Apaga el 50% de neuronas al azar para evitar memorización
    predictions = Dense(3, activation='softmax')(x) # 3 clases: Deficiencia, Fusario, Sanas

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. CALLBACKS (INSTRUMENTACIÓN)
    
    # A. Checkpoint: Guarda el mejor modelo basado en val_accuracy
    checkpoint_path = os.path.join(MODELS_DIR, "mobilenet_v2_best.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    # B. Early Stopping: Si no mejora en 5 épocas, para.
    early_stop = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)
    
    # C. TensorBoard
    # Crea una carpeta única por ejecución basada en la hora
    log_dir = os.path.join(LOGS_DIR, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1, 
        write_graph=False 
    )
    
    # D. Reduce LR: Si se estanca, baja la velocidad de aprendizaje
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    # 4. START TRAINING
    print("🚀 Iniciando entrenamiento... (Abrir TensorBoard para monitorear)")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, tensorboard_callback, reduce_lr]
    )
    
    print(f"✅ Entrenamiento finalizado. Modelo guardado en {checkpoint_path}")

if __name__ == "__main__":
    # Asegurar directorios
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
    
    train_pipeline()