import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

IMG_SIZE        = 128
BATCH_SIZE      = 32
EPOCHS_HEAD     = 15
EPOCHS_FINETUNE = 20
CLASSES         = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
NUM_CLASSES     = len(CLASSES)


def build_transfer_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='WasteClassifier_Transfer')
    return model, base_model


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


def train_transfer_model(dataset_path='dataset', save_path='models'):
    os.makedirs(save_path, exist_ok=True)

    train_gen, val_gen, test_gen = get_data_generators(dataset_path)
    model, base_model = build_transfer_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print("\n>>> Phase 1: Training classification head...")
    history1 = model.fit(
        train_gen, epochs=EPOCHS_HEAD, validation_data=val_gen,
        callbacks=[
            callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor='val_accuracy'),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        ], verbose=1
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n>>> Phase 2: Fine-tuning top layers of MobileNetV2...")
    history2 = model.fit(
        train_gen, epochs=EPOCHS_FINETUNE, validation_data=val_gen,
        callbacks=[
            callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7),
            callbacks.ModelCheckpoint(
                os.path.join(save_path, 'transfer_best.keras'),
                save_best_only=True, monitor='val_accuracy'
            ),
        ], verbose=1
    )

    combined = {k: history1.history[k] + history2.history[k] for k in history1.history}

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"\nTest Accuracy (Transfer): {test_acc:.4f}")

    hist_dict = {k: [float(v) for v in vals] for k, vals in combined.items()}
    hist_dict['test_accuracy'] = float(test_acc)
    hist_dict['test_loss']     = float(test_loss)
    hist_dict['phase2_start']  = len(history1.history['accuracy'])
    with open(os.path.join(save_path, 'transfer_history.json'), 'w') as f:
        json.dump(hist_dict, f, indent=2)

    _plot_history(combined, 'transfer', save_path,
                  phase2_start=len(history1.history['accuracy']))
    return model, combined, test_acc


def _plot_history(history, model_name, save_path, phase2_start=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Training History – {model_name.upper()} (MobileNetV2)', fontsize=14, fontweight='bold')

    for ax, metric in zip(axes, ['accuracy', 'loss']):
        ax.plot(history[metric],           label='Train', color='#4CAF50')
        ax.plot(history[f'val_{metric}'],  label='Val',   color='#FF9800')
        if phase2_start:
            ax.axvline(x=phase2_start, color='red', linestyle='--',
                       label=f'Fine-tune (ep {phase2_start})', alpha=0.7)
        ax.set_title(metric.capitalize()); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_curves.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → {save_path}/{model_name}_curves.png")


if __name__ == '__main__':
    train_transfer_model()