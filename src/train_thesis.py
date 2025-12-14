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

# --- HIPERPARÁMETROS DE TESIS DE MAESTRÍA (Algoritmo EDA) ---
THESIS_ALPHA  = 0.000071  # Learning Rate
THESIS_BETA_1 = 0.750077  # Momentum (Menor inercia que el default 0.9)
THESIS_BETA_2 = 0.962037  # RMSProp factor (Adaptación más rápida que 0.999)

def train_thesis_config():
    # 1. DATOS (Augmentation Agresivo para combatir Overfitting)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print("🔄 Generando flujos de datos...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    # 2. ARQUITECTURA
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Estrategia: Descongelamos las últimas capas desde el principio
    # Con tus hiperparámetros finos, podemos permitirnos entrenar más capas sin romper los pesos.
    base_model.trainable = True
    
    # Congelamos las primeras 100 (capas de características básicas)
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x) 
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout alto (50%) para forzar generalización
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. OPTIMIZADOR "TESIS CONFIG"
    optimizer = Adam(
        learning_rate=THESIS_ALPHA,
        beta_1=THESIS_BETA_1,
        beta_2=THESIS_BETA_2,
        epsilon=1e-07 # Epsilon estándar para estabilidad numérica
    )

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. CALLBACKS
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(MODELS_DIR, "thesis_model_best.keras")
    
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # Paciencia media
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    
    # TensorBoard
    log_dir = os.path.join(LOGS_DIR, "fit", f"thesis_run_{timestamp}")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
    
    # Reduce LR: Si se estanca, dividimos el alpha a la mitad
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

    print(f"🚀 Iniciando Entrenamiento Experimental (Configuración Tesis MSc)...")
    print(f"   Alpha: {THESIS_ALPHA} | Beta1: {THESIS_BETA_1} | Beta2: {THESIS_BETA_2}")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=30, # Le damos tiempo para converger
        callbacks=[checkpoint, early_stop, tensorboard, reduce_lr]
    )
    
    print(f"✅ Experimento finalizado. Modelo guardado en {checkpoint_path}")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
    train_thesis_config()