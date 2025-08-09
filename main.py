import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import pickle

# Paths to dataset
DATA_PATHS = {
    'train': r"C:\Users\DELL\PycharmProjects\PythonProject22\train",
    'val': r"C:\Users\DELL\PycharmProjects\PythonProject22\val",
    'test': r"C:\Users\DELL\PycharmProjects\PythonProject22\test"
}

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20


# Data validation
def validate_data_paths():
    """Validate dataset paths and check for data integrity."""
    for split, path in DATA_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        classes = os.listdir(path)
        if not classes:
            raise ValueError(f"No classes found in {split} directory")
        for class_name in classes:
            class_path = os.path.join(path, class_name)
            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if not images:
                raise ValueError(f"No images found in {class_path}")
    print("Data validation completed successfully.")


# Data preprocessing and augmentation
def create_data_generators():
    """Create data generators with augmentation for training and rescaling for validation/test."""
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        DATA_PATHS['train'],
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb'
    )
    val_generator = val_test_datagen.flow_from_directory(
        DATA_PATHS['val'],
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb'
    )
    test_generator = val_test_datagen.flow_from_directory(
        DATA_PATHS['test'],
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )

    print(f"Sample batch shape: {next(train_generator)[0].shape}")
    return train_generator, val_generator, test_generator, train_generator.num_classes


# Build custom CNN model
def build_cnn_model(num_classes):
    """Build a custom CNN model from scratch."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Build and fine-tune pre-trained models
def build_pretrained_model(base_model, num_classes):
    """Build and fine-tune a pre-trained model."""
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train and save model
def train_model(model, train_generator, val_generator, model_name):
    """Train the model and save the best one based on validation accuracy."""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{model_name}_best.h5", monitor='val_accuracy', save_best_only=True, mode='max'
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3
    )
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, lr_scheduler],
        class_weight=class_weights_dict
    )
    with open(f"{model_name}_history.pkl", 'wb') as f:
        pickle.dump(history.history, f)
    return history


# Evaluate model
def evaluate_model(model, test_generator, model_name):
    """Evaluate model and print metrics."""
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(),
                yticklabels=test_generator.class_indices.keys())
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()


# Plot training history
def plot_history(history, model_name):
    """Plot training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()

    plt.savefig(f"{model_name}_training_history.png")
    plt.close()


def main():
    # Validate data
    validate_data_paths()

    # Create data generators
    train_generator, val_generator, test_generator, num_classes = create_data_generators()

    # Train custom CNN
    cnn_model = build_cnn_model(num_classes)
    cnn_history = train_model(cnn_model, train_generator, val_generator, "custom_cnn")
    evaluate_model(cnn_model, test_generator, "Custom CNN")
    plot_history(cnn_history, "Custom CNN")

    # Pre-trained models
    pretrained_models = {
        "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        "MobileNet": MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        "InceptionV3": InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    }

    for model_name, base_model in pretrained_models.items():
        print(f"Training {model_name}...")
        try:
            model = build_pretrained_model(base_model, num_classes)
            history = train_model(model, train_generator, val_generator, model_name.lower())
            evaluate_model(model, test_generator, model_name)
            plot_history(history, model_name)
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")


if __name__ == "__main__":
    main()