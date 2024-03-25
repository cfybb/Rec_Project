import numpy as np


def heatmaps_to_keypoints(heatmaps, scale, threshold, neighbour=4):
    """
    heatmaps: ndarray of 1 x 8 x height x width
    scale: ratio of original image to model input
    threshold: keypoint detection threshold
    neighbour: pixel distance of peak neighbourhood
    """
    heatmaps = heatmaps[0]  # 8 x height x width
    num_keypoints, height, width = heatmaps.shape
    max_id_flatten = heatmaps.reshape(num_keypoints, -1).argmax(-1)  # (8,)
    max_id = np.unravel_index(max_id_flatten, (height, width))  # (8, 2) -> peak coordinate with integer precision


    grid = np.array(range(2 * neighbour + 1), dtype=np.float32)
    print("grid",grid)
    keypoints = []
    for i, (y, x) in enumerate(zip(*max_id)):
        print(heatmaps[i, y, x])
        print(x,y)
        if heatmaps[i, y, x] < threshold or x < neighbour or x >= width - neighbour or y < neighbour or y >= height - neighbour:
            keypoints.append((-1., -1.))
            continue
        # print("y scale", range(y-neighbour,y+neighbour+1))
        # print("x scale", range(x-neighbour,x+neighbour+1))
        sub_heatmap = heatmaps[i, y-neighbour:y+neighbour+1, x-neighbour:x+neighbour+1]  # (2 * neighbour + 1,  2 * neighbour + 1)
        # print("sub heatmap sum(1)",sub_heatmap.sum(1))
        # print("heatmap",sub_heatmap)
        shift_x = (sub_heatmap.sum(0) * grid).sum() / sub_heatmap.sum() - neighbour  # weighted sum to get x coordinate with float precision
        shift_y = (sub_heatmap.sum(1) * grid).sum() / sub_heatmap.sum() - neighbour  # weighted sum to get y coordinate with float precision
        keypoints.append(((x + shift_x) * scale, (y + shift_y) * scale))

    return keypoints