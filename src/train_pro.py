import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
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

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def train_professional():
    # 1. GENERADORES DE DATOS (Usando preprocess_input OFICIAL)
    # NOTA: No usamos rescale=1./255 aquí porque preprocess_input ya se encarga de todo.
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, # <--- ESTO GARANTIZA QUE LA ENTRADA SEA EXACTA (-1 a 1)
        rotation_range=2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input # Validación también debe llevarlo
    )

    print("🔄 Generando flujos de datos...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    # 2. ARQUITECTURA (MobileNetV2 - Ligero y Potente)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # FASE 1: Congelar TODO el cuerpo
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x) # Ayuda a estabilizar el Loss
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x) # Dropout moderado
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. COMPILACIÓN INICIAL
    # Learning Rate alto para que el Head aprenda rápido
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, "fit", f"mobilenet_pro_{timestamp}")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
    
    checkpoint_path = os.path.join(MODELS_DIR, "plant_doctor_final.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    print("\n🔥 FASE 1: Calentamiento (Entrenando solo el clasificador)...")
    history_warmup = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=10, # Pocas épocas, solo para despertar
        callbacks=[tensorboard, checkpoint]
    )

    # --- FASE 2: FINE TUNING ---
    print("\n🔓 FASE 2: Descongelando capas superiores (Fine Tuning)...")
    
    base_model.trainable = True
    
    # Congelamos las primeras 100 capas, entrenamos las últimas 55
    # MobileNetV2 tiene 155 capas
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # RE-COMPILAR con Learning Rate DIMINUTO
    # Esto es vital para no destruir lo aprendido en Fase 1
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)

    print("🚀 Iniciando Fine-Tuning profundo...")
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=20, # Épocas adicionales
        callbacks=[checkpoint, early_stop, tensorboard, reduce_lr] # Usamos early stop aquí
    )
    
    print(f"✅ Entrenamiento completo. Modelo guardado en {checkpoint_path}")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
    train_professional()