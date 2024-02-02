import numpy as np
import random
import copy

import cv2


# Data Augmentation HyperParams
RESCALE_MAX = 1.2
RESCALE_MIN = 0.8


def rescale(image, label):
    orig_shape = (image.shape[1], image.shape[0])  # (width, height)
    rescale_factor = random.uniform(RESCALE_MIN, RESCALE_MAX)

    # resize
    image = cv2.resize(image, None, fx=rescale_factor, fy=rescale_factor)
    label = [(c * rescale_factor if c != -1 else c) for c in label]

    # center padding
    if rescale_factor < 1:
        pass

    # center cropping
    if rescale_factor > 1:
        new_center = (image.shape[1] // 2, image.shape[0] // 2)
        crop_tl = np.array(
            (image.shape[1] // 2 - orig_shape[0] // 2,
             image.shape[0] // 2 - orig_shape[1] // 2)
        )
        image = image[
                crop_tl[1]:(crop_tl[1] + orig_shape[1]),
                crop_tl[0]:(crop_tl[0] + orig_shape[0])
        ]
        # adjust label
        label = np.array(label).reshape(-1, 2)
        label = np.array([(xy - crop_tl) if xy[0] != -1 else xy for xy in label])
        # check if new label is in the image
        label = np.array([xy if xy[0] >= 0 and xy[0] < orig_shape[1] and xy[1] >= 0 and xy[1] < orig_shape[1] else [-1, -1] for xy in label])

    label = label.reshape(-1)
    return image, label

    """
    max_scale_length = 80
    # randomly rescale the image and return the augmented images + labels
    #raise NotImplemented
    #assume we only shrink the data.
    #center_x = 1280/2, center_y = 720/2
    pooling_factor = random.randint(1,max_scale_length)
    return_images = copy.deepcopy(list(images))
    return_labels = copy.deepcopy(list(labels))
    # rescale for the labels
    for i in range(len(labels)):
        return_labels[i] = [label // pooling_factor for label in labels[i]]


    #rescale for the image (maxpooling + black filling)
    print(images)
    for j in range(len(images)):
        #print(type(images[j][0]))
        image_size = [len(images[j][0]),len(images[j][:])]
        image_matrix = return_images[j]
        pooled_matrix = np.zeros((image_size[0] // pooling_factor, image_size[1] // pooling_factor, 3), dtype=np.uint8)
        for k in range(0, image_size[0] // pooling_factor * pooling_factor, pooling_factor):
            for w in range(0, image_size[1] // pooling_factor * pooling_factor, pooling_factor):
                pooled_matrix[k // pooling_factor, w // pooling_factor] = np.max(image_matrix[k:k+pooling_factor, w:w+pooling_factor], axis=(0, 1))

        #padding
        padding_color = (0,0,0)
        padded_matrix = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        padded_matrix[:pooled_matrix.shape[0], :pooled_matrix.shape[1]] = pooled_matrix
        return_images[j] = padded_matrix
    return return_images, return_labels
    """



def shift(images, labels):
    #raise NotImplemented
    return_labels = copy.deepcopy(list(labels))
    return_images = copy.deepcopy(list(images))

    for i in range(len(images)):
        image_size = [len(images[i][0]), len(images[i][:])]

        # Find the upper bound and lower bound of labels for x and y
        x_label_list = labels[i][::2]
        y_label_list = labels[i][1::2]
        x_label_max, x_label_min = max(x_label_list), min(x_label_list)  # down, up
        y_label_max, y_label_min = max(y_label_list), min(y_label_list)  # right, left

        # Calculate bounds for shifting
        upper_bound = x_label_min
        lower_bound = x_label_max - image_size[0]
        left_bound = y_label_min
        right_bound = y_label_max - image_size[1]

        # Generate random shifts
        vertical_shift = random.randint(int(lower_bound), int(upper_bound))
        horizontal_shift = random.randint(int(right_bound), int(left_bound))

        # Perform image shift
        black_image = np.zeros((image_size[0], image_size[1]))
        for j in range(image_size[0]):
            for k in range(image_size[1]):
                new_row = min(max(0, j + vertical_shift), image_size[0] - 1)
                new_col = min(max(0, k + horizontal_shift), image_size[1] - 1)
                black_image[new_row, new_col] = images[i][j][k]

        # Update return_images and return_labels with shifted data
        return_images[i] = black_image
        return_labels[i][::2] = [label + vertical_shift for label in return_labels[i][::2]]
        return_labels[i][1::2] = [label + horizontal_shift for label in return_labels[i][1::2]]

    return return_images, return_labels



def rotate(images, labels):
    #raise NotImplemented
    return_images = copy.deepcopy(list(images))
    return_labels = copy.deepcopy(list(labels))

    for i in range(len(images)):
        image = return_images[i]
        height, width, _ = image.shape

        # random rotation angle
        rotation_angle = np.random.uniform(-180, 180)

        # rotated matrix value
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)

        # black figure
        #black_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # apply and fill black
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return_images[i] = rotated_image

        # rotate labels
        rotated_labels = []
        for label in return_labels[i]:
            x, y = label[0], label[1]
            # change notation
            rotated_x = int(x * np.cos(np.radians(rotation_angle)) - y * np.sin(np.radians(rotation_angle)))
            rotated_y = int(x * np.sin(np.radians(rotation_angle)) + y * np.cos(np.radians(rotation_angle)))
            rotated_labels.append([rotated_x, rotated_y])

        return_labels[i] = rotated_labels

    return return_images, return_labels

def saturation(images,labels):
    #raise NotImplemented
    return_images = copy.deepcopy(list(images))
    return_labels = copy.deepcopy(list(labels))
    for i in range(len(images)):
        image = return_images[i]
        # saturation range
        saturation_factor = np.random.uniform(0.5, 1.5)  #sample
        #  BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # adjust
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
        #  HSV to BGR
        saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return_images[i] = saturated_image


    return return_images, return_labels

def random_Noise(images,labels):
    #raise NotImplemented
    return_labels = copy.deepcopy(list(labels))
    return_images = copy.deepcopy(list(images))
    for i  in range(len(images)):
        image_size = [len(images[i][0]), len(images[i][:])]
        noise = np.random.normal(loc=0, scale=1, size=image_size)
        return_images[i] +=noise
    return return_images, return_labels



def data_augmentation(image, label, options=["rescaling", "shifting", "rotation", "saturation", "random noise"]):
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
            image, label = rescale(image, label)
        elif augment_option == "shifting":
            image, label = shift(image, label)
        elif augment_option == "rotation":
            image, label = rotate(image, label)
        elif augment_option == "saturation":
            image, label = saturation(image, label)
        elif augment_option == "random noise":
            image, label = random_Noise(image, label)