from multiprocessing import Process, Queue

import time


def preprocess_one(name, inc, input_queue, interval):
    print(f"{name}: preprocessing frame_{inc}...")
    time.sleep(interval / 1000)
    input_queue.put(inc)


def processing_one(name, input_queue, output_queue, interval):
    inc = input_queue.get()
    print(f"{name}: forwarding frame_{inc} -> output_{inc}...")
    time.sleep(interval / 1000)
    output_queue.put(inc)


def postprocess_one(name, output_queue, interval):
    inc = output_queue.get()
    print(f"{name}: postprocessing output_{inc}...")
    time.sleep(interval / 1000)


def preprocess_f(name, input_queue, interval):
    inc = 0
    while True:
        inc += 1
        preprocess_one(name, inc, input_queue, interval)


def processing_f(name, input_queue, output_queue, interval):
    while True:
        processing_one(name, input_queue, output_queue, interval)


def postprocess_f(name, output_queue, interval):
    while True:
        postprocess_one(name, output_queue, interval)


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
    p1 = Process(target=preprocess_f, args=("preprocess", input_queue, 200))
    p2 = Process(target=processing_f, args=("inference", input_queue, output_queue, 500))
    p3 = Process(target=postprocess_f, args=("postprocess", output_queue, 300))
    for p in (p1, p2, p3):
        p.start()
        time.sleep(1)
