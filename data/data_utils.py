import numpy as np
import random
import copy
import cv2

#rescale_hyperparam
RESCALE_MIN = 0.8
RESCALE_MAX = 1.2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize(image, target_size):
    raise NotImplementedError


def normalization(image):
    raise NotImplementedError


#here the image and keypoints will be only for one image.
def rescale(image, labels):
    rescale_factor = random.uniform(RESCALE_MIN, RESCALE_MAX)
    #find center
    cent_x = image.shape[0]//2
    cent_y = image.shape[1]//2
    #resize
    image = cv2.resize(image, None, fx = rescale_factor, fy = rescale_factor)

    #center coorp for labels (for now I will assume it is center-based)
    keypt_pair = [(labels[i], labels[i+1]) for i in range(0, len(labels), 2)]
    #move
    trans_pair = [(x - cent_x, y - cent_y) for x, y in keypt_pair]
    #rescale
    scaled_pts = [(x * rescale_factor, y * rescale_factor) for x, y in trans_pair]
    #move back
    final_points = [(x + cent_x, y + cent_y) for x, y in scaled_pts]
    #change back to pure list
    final_points_flat_list = [coord for point in final_points for coord in point]
    #check if the point is out of range (only work for enlarge)
    return_keypt = []
    for i in range(0, len(final_points_flat_list), 2):
        x, y = final_points_flat_list[i], final_points_flat_list[i + 1]
        if 0 <= x <= image.shape[0] and 0 <= y <= image.shape[1]:
            return_keypt.extend([x, y])
        else:
            return_keypt.extend([-1, -1])


    return image,return_keypt




def shift(image, labels):
    #limit
    shift_max_x = image.shape[0]//2
    shift_max_y = image.shape[1]//2
    #shift random
    shift_x = random.randint(0,shift_max_x)
    shift_y = random.randint(0,shift_max_y)
    #shift matrix
    trans_mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    image = cv2.warpAffine(image,trans_mat,(image.shape[1], image.shape[0]))
    #labels
    shifted_points = [(x + shift_x, y + shift_y) for x, y in zip(labels[0::2], labels[1::2])]
    final_points_flat_list = [coord for pair in shifted_points for coord in pair]
    #check if the point is out of range (only work for enlarge)
    return_keypt = []
    for i in range(0, len(final_points_flat_list), 2):
        x, y = final_points_flat_list[i], final_points_flat_list[i + 1]
        if 0 <= x <= image.shape[0] and 0 <= y <= image.shape[1]:
            return_keypt.extend([x, y])
        else:
            return_keypt.extend([-1, -1])


    return image,return_keypt


def rotate(image, labels):
    #raise NotImplemented
    #return_images = copy.deepcopy(list(images))
    #return_labels = copy.deepcopy(list(labels))



    height, width, _ = image.shape

    # random rotation angle
    rotation_angle = np.random.uniform(-180, 180)

    # rotated matrix value
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)

    # black figure
    #black_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # apply and fill black
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # rotate labels
    #transform
    points_array = np.array(labels).reshape(-1, 2) #N*2
    center = (image.shape[0] // 2, image.shape[1] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    rotated_points_array = cv2.transform(np.array([points_array]), rotation_matrix).squeeze()
    return_labels = rotated_points_array.flatten().tolist()

    return rotated_image, return_labels

def saturation(image,labels):
    #raise NotImplemented
    #return_images = copy.deepcopy(list(images))
    #return_labels = copy.deepcopy(list(labels))

    # saturation range
    saturation_factor = np.random.uniform(0.5, 1.5)  #sample
    #  BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # adjust
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
    #  HSV to BGR
    return_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)



    return return_image, labels

def random_Noise(image,labels):
    #raise NotImplemented
    #return_labels = copy.deepcopy(list(labels))
    #return_images = copy.deepcopy(list(images))
    image_size = image.shape
    noise = np.random.normal(loc=0, scale=1, size=image_size).astype(np.uint8)
    image+=noise
    return image, labels



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
        elif augment_option == "rotation":
            images, labels = rotate(images, labels)
        elif augment_option == "saturation":
            images, labels = saturation(images, labels)
        elif augment_option == "random noise":
            images, labels = random_Noise(images, labels)
    return images, labels