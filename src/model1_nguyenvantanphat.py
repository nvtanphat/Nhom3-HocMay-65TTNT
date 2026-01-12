"""
Xception Model with Transfer Learning for Brain Tumor Classification
Author: [Ho Va Ten]
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_xception_model(input_shape=(299, 299, 3), num_classes=4):
    """Build Xception model with transfer learning"""
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='max'
    )
    base_model.trainable = False  # Freeze in phase 1
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='Xception_BrainTumor'), base_model


def compile_model(model, learning_rate=0.001):
    """Compile model with Adamax optimizer"""
    model.compile(
        optimizer=Adamax(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_path='best_xception.keras', patience=3):
    """Get training callbacks"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]


def unfreeze_base_model(base_model, freeze_layers=50):
    """Unfreeze base model for fine-tuning, keep first N layers frozen"""
    base_model.trainable = True
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    return base_model


def train_xception(model, base_model, train_gen, valid_gen, model_path='best_xception.keras'):
    """Two-phase training: feature extraction + fine-tuning"""
    
    # Phase 1: Feature Extraction
    print("\n" + "="*50)
    print("Phase 1: Feature Extraction (base frozen)")
    print("="*50)
    
    model = compile_model(model, learning_rate=0.001)
    callbacks = get_callbacks(model_path, patience=3)
    
    history1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=10,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tuning
    print("\n" + "="*50)
    print("Phase 2: Fine-tuning (partial unfreeze)")
    print("="*50)
    
    base_model = unfreeze_base_model(base_model, freeze_layers=50)
    model = compile_model(model, learning_rate=0.0001)
    callbacks = get_callbacks(model_path, patience=5)
    
    history2 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=15,
        callbacks=callbacks
    )
    
    return history1, history2
