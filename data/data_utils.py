
def rescale(images, labels):
    # randomly rescale the image and return the augmented images + labels
    raise NotImplemented


def shift(images, labels):
    raise NotImplemented


def data_augmentation(images, labels, options=["rescaling", "shifting", "rotation", "saturation", "random noise"]):
    """
    Data Augmentation Main Function. Augmentations can be chosen from:
    - Rescaling
    - Shifting
    - Rotation
    - Saturation
    - Random Noise
    """

    for augment_option in options:
        assert augment_option in ["rescaling", "shifting", "rotation", "saturation", "random noise"]
        if augment_option == "rescaling":
            images, labels = rescale(images, labels)
        elif augment_option == "shifting":
            images, labels = shift(images, labels)
