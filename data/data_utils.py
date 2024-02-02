
def rescale(images, labels):
    # randomly rescale the image and return the augmented images + labels
    raise NotImplementedError


def shift(images, labels):
    raise NotImplementedError


def rotation(images, labels):
    raise NotImplementedError


def saturation(images, labels):
    raise NotImplementedError


def noise(images, labels):
    raise NotImplementedError


def data_augmentation(images, labels, options=["rescaling", "shifting", "rotation", "saturation", "noise"]):
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
            images, labels = rescale(images, labels)
        elif augment_option == "shifting":
            images, labels = shift(images, labels)
        elif augment_option == "rotation":
            images, labels = rotation(images, labels)
        elif augment_option == "saturation":
            images, labels = saturation(images, labels)
        elif augment_option == "noise":
            images, labels = noise(images, labels)
        else:
            raise ValueError(f"Undefined augmentation option: {augment_option}")

    return images, labels
