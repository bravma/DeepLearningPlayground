import glob
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import os

# Supress warning and informational messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])


def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )


img_width = 299
img_height = 299
num_epochs = 2
batch_size = 32
num_fc_neurons = 1024

train_dir = "./data/train"
test_dir = "./data/test"

num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_test_samples = get_num_files(test_dir)

train_img_gen = create_img_generator()
test_img_gen = create_img_generator()

train_generator = train_img_gen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    seed=42  # Random seed for reproducability
)

test_generator = test_img_gen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    seed=42  # Random seed for reproducability
)

# Load inception v3 model and load it with it's pretrained weights
# Exclude final fully connected layer
inception_base_model = InceptionV3(weights="imagenet", include_top=False)  # Excludes final FC layer

# Define the layers in the new classification prediction
x = inception_base_model.output
x = GlobalAveragePooling2D(x)
x = Dense(num_fc_neurons, activation="relu")(x)  # new FC layer, random init
predictions = Dense(num_classes, activation="softmax")(x)  # New Softmax layer

model = Model(inputs=inception_base_model.input, output=predictions)

# Prints model structure diagram
print(model.summary())

#  Transfer Learning with fine tuning
# Retrain the end few layers (called the top layers) of the inception model
print("\nPerforming Transfer Learning")

# Freeze layers
layers_to_freeze = 172
for layer in inception_base_model.layers[:layers_to_freeze]:
    layer.trainable = False
for layer in inception_base_model.layers[layers_to_freeze:]:
    layer.trainable = True

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])

# Fit the transfer learning model to the data from the generators.
# By using generators we can ask continue to request sample images and the generators will pull images
# from the training or validation folders and alter them slightly
history_transfer_learning = model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=num_train_samples // batch_size,
    validation_data=test_generator,
    validation_steps=num_test_samples // batch_size,
    class_weight="auto"
)

# Save transfer learning model
model.save("inceptionv3-transfer-learning.model")