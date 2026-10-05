import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

# 1. Dataset Path and Hyperparameters
DATASET_DIR = 'TomatoLeaf-Dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE_1 = 15
EPOCHS_PHASE_2 = 15

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 2. Data Augmentation & Generators
# EfficientNet expects inputs in range [0, 255], it has its own internal normalization.
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling/validation split for validation data
val_datagen = ImageDataGenerator(validation_split=0.2)

print("\nLoading Training Data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("\nLoading Validation Data...")
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get class names and print them
class_names = list(train_generator.class_indices.keys())
print("\nDetected Classes:", class_names)
NUM_CLASSES = len(class_names)

# 3. Build Model (EfficientNetB0)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model for Phase 1
base_model.trainable = False 

# Add Custom Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x) # Dropout to prevent overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Callbacks for Smart Training
checkpoint = ModelCheckpoint('best_tomato_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

# ==========================================
# PHASE 1: Train only the custom head
# ==========================================
print("\n--- PHASE 1: Training Custom Head ---")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_1,
    callbacks=[checkpoint, reduce_lr]
)

# ==========================================
# PHASE 2: Fine-Tuning the whole model
# ==========================================
print("\n--- PHASE 2: Fine-tuning entire model ---")
# Unfreeze the base model
base_model.trainable = True

# We recompile with a MUCH LOWER learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE_2,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

print("\n✅ Training Complete! The best model is saved as 'best_tomato_model.h5'")
