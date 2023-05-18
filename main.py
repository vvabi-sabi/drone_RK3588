from multiprocessing import Process, Queue

from utils import PostProcesses, Odometry, Visualizer
from devices import Cam, RK3588


def run(device, visualizer, post_process, odometry):
    device.camera_set()
    device.start()
    if post_process is not None and odometry is None:
        while True:
            frame, outputs = device.get_neuro_outputs()
            frame = post_process(frame, outputs)
            visualizer.show_frame(frame)
    elif odometry is not None and post_process is None:
        while True:
            _, outputs = device.get_neuro_outputs()
            coords = odometry(outputs)
            visualizer.show_trajectory(coords)
    if post_process is not None  and odometry is not None:
        while True:
            frame, outputs = device.get_neuro_outputs()
            post_process(frame, outputs)
            coords = odometry(outputs)
            visualizer.show_trajectory(coords)
            visualizer.show_frame(frame)
    else:
        while True:
            _, outputs = device.get_neuro_outputs()

def main(source):
    """Runs inference and addons (if mentions)
    Creating storages and sending data to them
    """
    post_process = True
    show_trajectory = False
    queue_size = 5
    q_pre = Queue(maxsize=queue_size)
    #q_outs = Queue(maxsize=cfg["inference"]["buf_size"])
    q_post = Queue(maxsize=queue_size)
    
    camera = Cam(source=source,
                 queue=q_pre)
    device = RK3588(camera, models, queue=q_post)
    models = ['YOLO'] # ['ResNet', 'UNet', 'Encoder']
    post_processes = PostProcesses(models, post_process)
    odometry_algs = Odometry(models, show_trajectory)
    visualizer = Visualizer()
    try:
        run(device, visualizer, post_processes, odometry_algs)
    except Exception as e:
        print("Main exception: {}".format(e))
        exit()


if __name__ == "__main__":
    camera_source = 11 # 'path/to/video.mp3'
    main(camera_source)
