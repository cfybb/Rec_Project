from multiprocessing import Process, Queue
import sys
import torch
sys.path.append('C:/prdue/job_preperation_general/support_company/project/Rec_Project/utils')
sys.path.append('C:/prdue/job_preperation_general/support_company/project/Rec_Project')
import time
from preprocessing import camera_stream_to_tensor
import torchvision.models as models
from postprocess import heatmaps_to_keypoints
from PIL import Image
import torchvision.transforms as transforms
import cv2

def preprocess_one(name, input_queue):
    print(f"{name}: preprocessing ...")
    #time.sleep(interval / 1000)
    camera_tensor, camera_frame = camera_stream_to_tensor()
    input_queue.put((camera_tensor,camera_frame))
    # save result
    # to_pil = transforms.ToPILImage()
    # image = to_pil(camera_tensor)
    # image.save('camera_image.jpg')



def processing_one(name, input_queue, output_queue):
    (inc,camera_frame) = input_queue.get()
    print(f"{name}: forwarding frame_{inc} -> output_{inc}...")
    with torch.no_grad():
        model = torch.load("model_epoch_7.th")
        model.to('cpu')
        inc = inc.unsqueeze(0)
        heatmap = model(inc)
    # time.sleep(interval / 1000)
    output_queue.put((heatmap,camera_frame))


def postprocess_one(name, output_queue):
    (inc,camera_frame) = output_queue.get()
    print(f"{name}: postprocessing output_{inc}...")
    keypoints = heatmaps_to_keypoints(inc,2,0.9)
    # show the result
    # camera_image = Image.open('camera_image.jpg')
    # draw = ImageDraw.Draw(camera_image)
    # for point in keypoints:
        # draw.ellipse([point[0]-2, point[1]-2, point[0]+2, point[1]+2], fill='red', outline='red')

    # Display the image with keypoints
    # camera_image.show()

    # cv2
    for point_pair in keypoints:
        (x, y) = point_pair
        cv2.circle(camera_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    return camera_frame

    # time.sleep(interval / 1000)


def preprocess_f(name, input_queue):
    inc = 0
    while True:
        inc += 1
        preprocess_one(name, input_queue)


def processing_f(name, input_queue, output_queue):
    while True:
        processing_one(name, input_queue, output_queue)


def postprocess_f(name, output_queue):
    while True:
        camera_frame = postprocess_one(name, output_queue)
        cv2.imshow("",camera_frame)
        cv2.waitKey(1)



if __name__ == "__main__":

    input_queue = Queue(5)
    output_queue = Queue(5)

    # singular processing version
    # inc = 0
    # while True:
    #     inc += 1
    #     preprocess_one("preprocess", inc, input_queue, 200)
    #     processing_one("inference", input_queue, output_queue, 500)
    #     postprocess_one("postprocess", output_queue, 300)

    # parallel processing version
    p1 = Process(target=preprocess_f, args=("preprocess", input_queue))
    p2 = Process(target=processing_f, args=("inference", input_queue, output_queue))
    p3 = Process(target=postprocess_f, args=("postprocess", output_queue))
    for p in (p1, p2, p3):
        p.start()
        time.sleep(1)
    for p in (p1, p2, p3):
        p.join()