# cervical-cancer-classification-colposcopy
#You need to run it in tensorflow and keras with TPU environment
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

# Directories for training and test datasets
data_dir = 'data-path'
test_dir = 'data-path'

# Parameters
batch_size = 32
img_size = (224, 224)
num_classes = 3
epochs_initial = 40
epochs_finetune = 40

# Load Datasets
def load_datasets(data_dir, test_dir):
    train_dataset = image_dataset_from_directory(
        data_dir, validation_split=0.2, subset='training', seed=42,
        image_size=img_size, batch_size=batch_size, label_mode='categorical')

    val_dataset = image_dataset_from_directory(
        data_dir, validation_split=0.2, subset='validation', seed=42,
        image_size=img_size, batch_size=batch_size, label_mode='categorical')

    test_dataset = image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size, label_mode='categorical')

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_dataset.prefetch(buffer_size=AUTOTUNE),
            val_dataset.prefetch(buffer_size=AUTOTUNE),
            test_dataset.prefetch(buffer_size=AUTOTUNE))

train_dataset, val_dataset, test_dataset = load_datasets(data_dir, test_dir)
#Model
def se_block(input_tensor, channels, reduction=16):
    """
    Squeeze-and-Excitation Block
    """
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // reduction, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = tf.reshape(se, [-1, 1, 1, channels])
    return layers.Multiply()([input_tensor, se])

def build_deep_model(input_shape=(224, 224, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # Convolutional Block 1 + SE
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x, channels=64)

    # Convolutional Block 2 with Residual Connection + SE
    shortcut = x
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    shortcut = layers.Conv2D(128, (1, 1), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x, channels=128)

    # Convolutional Block 3 + SE
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x, channels=256)

    # Convolutional Block 4 + SE
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x, channels=512)

    # Convolutional Block 5 + SE
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x, channels=512)

    # Flatten and Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_deep_model()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


# Initial Training
history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=epochs_initial,
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# Plot Training and Validation Accuracy/Loss
def plot_training(history, title):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title(f'{title} Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f'{title} Loss')
    plt.show()

plot_training(history, "Initial Training")

# Grad-CAM Visualization
def grad_cam(model, image, class_idx):
    grad_model = models.Model([model.inputs], [model.get_layer('conv2d_5').output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, axis=0))
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

sample_images, _ = next(iter(test_dataset))
image = sample_images[0]
heatmap = grad_cam(model, image, class_idx=0)

plt.figure(figsize=(8, 8))
plt.imshow(image.numpy().astype("uint8"))
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title('Grad-CAM')
plt.axis('off')
plt.show()

# ROC-AUC Plot
def plot_roc_auc(model, dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i+1} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()

plot_roc_auc(model, test_dataset)
