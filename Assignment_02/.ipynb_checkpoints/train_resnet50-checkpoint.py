import tensorflow.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, UpSampling2D
from tensorflow.keras.optimizers import SGD
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set visible devices to use only GPU 0 (change index as needed, e.g., 1 for GPU 1)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("Using GPU:", gpus[0])
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Set random seed for reproducibility
seed = 20
np.random.seed(seed)
tf.random.set_seed(seed)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data
# ResNet50 expects input size of at least 197x197, so we upsample CIFAR-10 images (32x32)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize pixel values to [0, 1]
x_train /= 255.0
x_test /= 255.0

# Convert labels to categorical
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the model
model = Sequential()
# Upsample CIFAR-10 images from 32x32 to 224x224 to match ResNet50 input
model.add(UpSampling2D(size=(7, 7), input_shape=(32, 32, 3)))
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Freeze the ResNet50 layers (uncomment to freeze)
# for layer in base_model.layers:
#     layer.trainable = False

# Compile the model
optimizer = SGD(learning_rate=0.01, momentum=0.9)  # Use learning_rate instead of lr
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')
datagen.fit(x_train, seed=seed)  # Set seed for data augmentation

# Training parameters
batch_size = 256
epochs = 20

# Create dataset from generator
dataset = tf.data.Dataset.from_generator(
    lambda: datagen.flow(x_train, y_train, batch_size=batch_size, seed=seed),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 32, 32, 3], [None, num_classes])
).prefetch(tf.data.AUTOTUNE)

# Train the model
model.fit(
    dataset,
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    workers=4
)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save(f'resnet50_cifar10_seed{seed}.h5')