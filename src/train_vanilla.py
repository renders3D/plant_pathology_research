import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# --- CONFIGURACIÓN ESTÁNDAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "val")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def train_vanilla():
    # 1. Datos
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20, # Augmentation suave
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )

    # 2. Modelo Simple
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
    base_model.trainable = False # Congelado totalmente

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. Optimizador ADAM ESTÁNDAR (Defaults de Keras)
    # Sin betas personalizados, Learning Rate estándar
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    print("🚀 Iniciando Entrenamiento VANILLA (Control de Daños)...")
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=2)
        ]
    )

if __name__ == "__main__":
    train_vanilla()