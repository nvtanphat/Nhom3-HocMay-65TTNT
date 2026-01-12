"""
ResNet50 Model with Transfer Learning
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_resnet50_model(input_shape=(224, 224, 3), num_classes=4):
    """Build ResNet50 model with transfer learning"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze in phase 1
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='ResNet50_BrainTumor'), base_model


def compile_model(model, learning_rate=1e-3):
    """Compile model with Adam optimizer"""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks_stage1(model_path='best_resnet50_stage1.keras', patience=8):
    """Get training callbacks for stage 1 (frozen base)"""
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
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


def get_callbacks_stage2(model_path='best_resnet50_finetuned.keras', patience=12):
    """Get training callbacks for stage 2 (fine-tuning)"""
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
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]


def unfreeze_base_model(base_model, unfreeze_layers=10):
    """Unfreeze last N layers of base model for fine-tuning"""
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    return base_model


def train_resnet50(model, base_model, train_gen, valid_gen, 
                   model_path='best_resnet50_finetuned.keras',
                   epochs_stage1=20, epochs_stage2=50):
    """Two-stage training: feature extraction + fine-tuning"""
    
    # Stage 1: Train only top layers (base frozen)
    print("\n" + "="*50)
    print("STAGE 1: Training top layers (base frozen)")
    print("="*50)
    
    model = compile_model(model, learning_rate=1e-3)
    callbacks = get_callbacks_stage1()
    
    history1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs_stage1,
        callbacks=callbacks
    )
    
    # Stage 2: Fine-tune last 10 layers
    print("\n" + "="*50)
    print("STAGE 2: Fine-tuning last 10 layers")
    print("="*50)
    
    base_model = unfreeze_base_model(base_model, unfreeze_layers=10)
    trainable_count = sum([1 for l in base_model.layers if l.trainable])
    print(f"Trainable layers: {trainable_count}/{len(base_model.layers)}")
    
    model = compile_model(model, learning_rate=1e-4)
    callbacks = get_callbacks_stage2(model_path)
    
    history2 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs_stage2,
        callbacks=callbacks
    )
    
    return history1, history2

