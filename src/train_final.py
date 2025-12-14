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

def train_final_pipeline():
    # 1. GENERADORES DE DATOS
    # Usamos preprocess_input para asegurar la normalización exacta de MobileNet (-1 a 1)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
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

    # 2. ARQUITECTURA BASE
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # FASE 1: Congelamos TODO el cerebro base
    base_model.trainable = False 

    # Cabeza personalizada (Clasificador)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x) # Estabiliza
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Reduce overfitting
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. COMPILACIÓN INICIAL (Warmup)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, "fit", f"final_run_{timestamp}")
    
    # TENSORBOARD ACTIVADO 📈
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
    
    checkpoint_path = os.path.join(MODELS_DIR, "plant_doctor_final_v2.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    print("\n🔥 FASE 1: Warmup (Entrenando solo la cabeza)...")
    history_warmup = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=8, # Pocas épocas para estabilizar
        callbacks=[tensorboard, checkpoint]
    )

    # --- FASE 2: FINE TUNING (El secreto del éxito) ---
    print("\n🔓 FASE 2: Descongelando MobileNet (Fine Tuning)...")
    
    # Descongelamos la base
    base_model.trainable = True
    
    # Pero volvemos a congelar las primeras 100 capas (Low-level features)
    # MobileNetV2 tiene 155 capas. Entrenaremos las últimas 55.
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # RE-COMPILAR con Learning Rate MUY BAJO (1e-5)
    # Esto evita romper los pesos que ya aprendimos
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
        epochs=20, 
        initial_epoch=8, # Continuamos donde dejamos la Fase 1
        callbacks=[checkpoint, early_stop, tensorboard, reduce_lr]
    )
    
    print(f"✅ Entrenamiento completo. Modelo guardado en {checkpoint_path}")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
    train_final_pipeline()
