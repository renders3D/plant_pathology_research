import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.utils import class_weight
import numpy as np
import os
import datetime

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32 # EfficientNet consume más VRAM, si falla bajar a 16

def train_thor():
    # 1. GENERADORES DE DATOS
    # ¡OJO! EfficientNetB0 espera valores 0-255. NO USAR rescale=1./255
    
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True, # Las hojas no tienen "arriba" estricto
        brightness_range=[0.8, 1.2], # Importante para luz exterior
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator() # Sin rescale

    print("🔄 Generando flujos de datos...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    # 2. CALCULAR CLASS WEIGHTS (Equilibrio Matemático)
    # Esto ayuda si hay clases que el modelo ignora
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"⚖️ Pesos de Clase Calculados: {class_weights_dict}")

    # 3. ARQUITECTURA (EfficientNetB0)
    # include_top=False, weights='imagenet'
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    
    # Congelamos TODO el cuerpo base
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x) # Capa más ancha
    x = Dropout(0.5)(x) # Dropout agresivo
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 4. COMPILACIÓN
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOGS_DIR, "fit", f"thor_effnet_{timestamp}")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
    
    checkpoint_path = os.path.join(MODELS_DIR, "plant_doctor_thor.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    print("\n🔥 FASE ÚNICA: Entrenamiento Controlado con EfficientNet...")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=25,
        class_weight=class_weights_dict, # <--- ¡AQUÍ ESTÁ LA AYUDA!
        callbacks=[tensorboard, checkpoint, reduce_lr, early_stop]
    )
    
    print(f"✅ Entrenamiento completo. Modelo guardado en {checkpoint_path}")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)
    train_thor()