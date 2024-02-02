import random

import cv2

## Data Augmentation HyperParams
RESCALE_MIN = 0.8
RESCALE_MAX = 1.2


def rescale(image, labels):
    # randomly rescale the image and return the augmented images + labels
    rescale_factor = random.uniform(RESCALE_MIN, RESCALE_MAX)

    # resize
    image = cv2.resize(image, None, fx=rescale_factor, fy=rescale_factor)

    # center crop
    if rescale_factor > 1:


def shift(image, labels):
    raise NotImplementedError


def rotation(image, labels):
    raise NotImplementedError


def saturation(image, labels):
    raise NotImplementedError


def noise(image, labels):
    raise NotImplementedError


def data_augmentation(image, labels, options=["rescaling", "shifting", "rotation", "saturation", "noise"]):
    """
    Data Augmentation Main Function. Augmentations can be chosen from:
    - Rescaling
    - Shifting
    - Rotation
    - Saturation
    - Random Noise
    """

    for augment_option in options:
        assert augment_option in ["rescaling", "shifting", "rotation", "saturation", "noise"]
        if augment_option == "rescaling":
            image, labels = rescale(image, labels)
        elif augment_option == "shifting":
            image, labels = shift(image, labels)
        elif augment_option == "rotation":
            image, labels = rotation(image, labels)
        elif augment_option == "saturation":
            image, labels = saturation(image, labels)
        elif augment_option == "noise":
            image, labels = noise(image, labels)
        else:
            raise ValueError(f"Undefined augmentation option: {augment_option}")

    return image, labels
