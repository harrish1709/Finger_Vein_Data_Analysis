pip install git+https://www.github.com/keras-team/keras-contrib.git

# Step 2: Import necessary libraries
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Reshape, Lambda, Flatten, Dense, Dropout, BatchNormalization
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras_contrib.layers import Capsule
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 3: Set seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Step 4: Load grayscale images
images, labels = [], []

def load_images(folder):
    for file in os.listdir(folder):
        if file.endswith('.bmp'):
            path = os.path.join(folder, file)
            img = Image.open(path).convert('L').resize((64, 64))  # Grayscale
            images.append(np.expand_dims(np.array(img), axis=-1))
            if 'index' in file:
                labels.append(0)
            elif 'middle' in file:
                labels.append(1)
            elif 'ring' in file:
                labels.append(2)

data_dir="/content/drive/MyDrive/data"
load_images(data_dir)

X = np.array(images) / 255.0
y = to_categorical(np.array(labels))
num_classes = y.shape[1]
print(num_classes)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# Step 5: Model
input_shape = X_train.shape[1:]
x = Input(shape=input_shape)

conv1 = Conv2D(64, (9, 9), activation='relu', padding='valid', kernel_initializer='he_normal')(x)
conv1 = BatchNormalization()(conv1)
conv1 = Dropout(0.3)(conv1)

primary_caps = Conv2D(64, (9, 9), activation='relu', strides=2, padding='valid', kernel_initializer='he_normal')(conv1)
primary_caps = Reshape(target_shape=(-1, 64))(primary_caps)
primary_caps = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), output_shape=lambda s: s)(primary_caps)

digit_caps = CapsuleLayer(num_capsule=num_classes, dim_capsule=8, routings=3)(primary_caps)
out_caps = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)), output_shape=lambda s: (s[0], s[1], 1))(digit_caps)
out_caps = Flatten()(out_caps)
out = Dense(3, activation='softmax')(out_caps)

model = Model(x, out)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(1e-4), metrics=['accuracy'])

# Step 6: Train
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))))

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

callbacks = [EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)]

train_gen = datagen.flow(X_train, y_train, batch_size=32, seed=seed)
model.fit(train_gen, validation_data=(X_val, y_val), epochs=100, callbacks=callbacks, class_weight=class_weights)

# Step 7: Evaluation
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)
print(classification_report(y_true, y_pred))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
