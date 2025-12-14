import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os
import datetime

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

IMG_SIZE = (224, 224) # EfficientNetB0 nativo
BATCH_SIZE = 32 # Si da error de memoria en GPU, bajarlo a 16
EPOCHS = 25
LEARNING_RATE = 1e-4

def train_efficientnet():
    # 1. GENERADORES (Con Preprocesamiento Nativo de EfficientNet)
    # EfficientNet espera valores 0-255, su preprocess_input interno se encarga del resto.
    # NO dividimos por 255 aquí manualmente.
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, # <--- CLAVE: Usar la función oficial
        rotation_range=40,      # Más agresivo
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,     # Las hojas no tienen "arriba" o "abajo" estricto
        brightness_range=[0.8, 1.2], # Importante para cambios de luz
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print(f"🔄 Cargando datos...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    # 2. ARQUITECTURA (EfficientNetB0)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Congelamos base
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Capa Densa con Regularización L2 (Anti-Overfitting)
    x = BatchNormalization()(x) # Estabiliza el aprendizaje
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x) 
    x = Dropout(0.5)(x) # Apaga 50% neuronas: obliga a aprender caminos redundantes
    
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. CALLBACKS
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    checkpoint_path = os.path.join(MODELS_DIR, "efficientnet_b0_best.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    # Paciencia media: si no mejora en 7 épocas, paramos.
    early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
    
    log_dir = os.path.join(LOGS_DIR, "fit", f"effnet_{timestamp}")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    print("🚀 Iniciando entrenamiento con EfficientNetB0...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, tensorboard, reduce_lr]
    )
    
    print(f"✅ Entrenamiento finalizado.")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
    train_efficientnet()