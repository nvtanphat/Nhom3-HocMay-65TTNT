"""
Mô hình CNN from scratch với kiến trúc kiểu ResNet cho phân loại U não.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def residual_block(x, filters, stride=1):
    """ Khối Squeeze-and-Excitation (SE Block) """
    shortcut = x
    
    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(filters, (3,3), padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, padding='same',
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('swish')(x)
    return x


def se_block(x, ratio=16):
    """Khối Squeeze-and-Excitation (SE Block)"""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='swish')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def residual_block_se(x, filters, stride=1):
    """Khối Residual kết hợp với SE Block (ResNet + Attention)"""
    shortcut = x
    
    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(filters, (3,3), padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, ratio=16)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, padding='same',
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('swish')(x)
    return x


def build_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    """Xây dựng kiến trúc mô hình CNN"""
    inputs = layers.Input(shape=input_shape)
    
    # STEM
    x = layers.Conv2D(32, (3,3), strides=2, padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(32, (3,3), padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(64, (3,3), padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    
    # Block 1: 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.Dropout(0.15)(x)
    
    # Block 2: 128 filters
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = layers.Dropout(0.2)(x)
    
    # Block 3: 256 filters + SE
    x = residual_block_se(x, 256, stride=2)
    x = residual_block_se(x, 256)
    x = layers.Dropout(0.25)(x)
    
    # Block 4: 512 filters + SE
    x = residual_block_se(x, 512, stride=2)
    x = residual_block_se(x, 512)
    x = layers.Dropout(0.3)(x)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='CNN_Pro_99')


def compile_model(model, learning_rate=0.001):
    """Compile model with Adamax optimizer"""
    model.compile(
        optimizer=Adamax(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_path='best_cnn_pro.keras'):
    """Get training callbacks"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=12,
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


def train_model(model, train_generator, valid_generator, epochs=70, model_path='best_cnn_pro.keras'):
    """Train the model"""
    callbacks = get_callbacks(model_path)
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    return history
