import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 30
CLASSES     = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
NUM_CLASSES = len(CLASSES)


def build_model_scratch(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Custom CNN from scratch.
    Architecture: Conv2D → BN → MaxPool (×4) → GlobalAvgPool → Dense
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.35),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),

        # Classifier head
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ], name='WasteClassifier_Scratch')

    return model


def get_data_generators(dataset_path='dataset'):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen, test_gen


def train_scratch_model(dataset_path='dataset', save_path='models'):
    os.makedirs(save_path, exist_ok=True)

    train_gen, val_gen, test_gen = get_data_generators(dataset_path)

    model = build_model_scratch()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    cbs = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss'),
        callbacks.ModelCheckpoint(
            os.path.join(save_path, 'scratch_best.keras'),
            save_best_only=True, monitor='val_accuracy'
        ),
    ]

    print("\n>>> Training CNN from scratch...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=cbs,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"\nTest Accuracy (Scratch): {test_acc:.4f}")

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_dict['test_accuracy'] = float(test_acc)
    hist_dict['test_loss']     = float(test_loss)
    with open(os.path.join(save_path, 'scratch_history.json'), 'w') as f:
        json.dump(hist_dict, f, indent=2)

    _plot_history(history, 'scratch', save_path)
    return model, history, test_acc


def _plot_history(history, model_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Training History – {model_name.upper()}', fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'],     label='Train', color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='#FF5722')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train', color='#2196F3')
    axes[1].plot(history.history['val_loss'], label='Val',   color='#FF5722')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_curves.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → {save_path}/{model_name}_curves.png")


if __name__ == '__main__':
    train_scratch_model()