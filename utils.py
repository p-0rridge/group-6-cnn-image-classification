
import os
import pathlib
import numpy as np
import tensorflow as tf
import kagglehub


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def load_animals10_split_data():
    split_data_path = "/content/drive/MyDrive/Colab Notebooks/w3_group-6-cnn-image-classification/data/animals10_split_data.npz"

    data = np.load(split_data_path, allow_pickle=True)

    x_train = data["x_train"]
    x_val = data["x_val"]
    x_test = data["x_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    class_names = data["class_names"].tolist()

    return x_train, x_val, x_test, y_train, y_val, y_test, class_names


def download_animals10_dataset():
    current_dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")
    current_data_dir = pathlib.Path(current_dataset_path) / "raw-img"

    return current_data_dir


def remap_image_paths(old_paths, new_data_dir):
    new_paths = []

    for old_path in old_paths:
        relative_path = old_path.split("raw-img/")[-1]
        new_path = os.path.join(str(new_data_dir), relative_path)
        new_paths.append(new_path)

    return np.array(new_paths)


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)

    image = image / 255.0

    return image, label


def create_dataset(x, y, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(x),
            seed=42
        )

    dataset = (
        dataset
        .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return dataset


def load_animals10_datasets():
    x_train, x_val, x_test, y_train, y_val, y_test, class_names = load_animals10_split_data()

    current_data_dir = download_animals10_dataset()

    x_train = remap_image_paths(x_train, current_data_dir)
    x_val = remap_image_paths(x_val, current_data_dir)
    x_test = remap_image_paths(x_test, current_data_dir)

    train_dataset = create_dataset(x_train, y_train, shuffle=True)
    val_dataset = create_dataset(x_val, y_val, shuffle=False)
    test_dataset = create_dataset(x_test, y_test, shuffle=False)

    return train_dataset, val_dataset, test_dataset, class_names
