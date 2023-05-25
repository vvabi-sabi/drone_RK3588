from multiprocessing import Queue

from utils import PostProcesses, Odometry, Visualizer
from devices import Cam, RK3588


def run(device, visualizer, post_process, odometry):
    device._camera.run()
    device._neuro.run_inference()
    #device.start()
    if post_process is not None and odometry is None:
        post_process.run()
        while True:
            frame, outputs = device.get_neuro_outputs()
            frame = post_process(frame, outputs)
            visualizer.show_frame(frame)
    elif odometry is not None and post_process is None:
        odometry.run()
        while True:
            _, outputs = device.get_neuro_outputs()
            coords = odometry(outputs)
            visualizer.show_trajectory(coords)
    if post_process is not None  and odometry is not None:
        post_process.run()
        odometry.run()
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
    models_list = ['YOLO'] # ['ResNet', 'UNet', 'Encoder']
    camera = Cam(source=source,
                 queue=q_pre,
                 models=models_list)
    camera.set()
    device = RK3588(camera, models_list)
    post_processes = PostProcesses(models_list, post_process, queue=q_post)
    odometry_algs = Odometry(models_list, show_trajectory)
    visualizer = Visualizer()
    try:
        run(device, visualizer, post_processes, odometry_algs)
    except Exception as e:
        print("Main exception: {}".format(e))
        exit()


if __name__ == "__main__":
    camera_source = 11 # 'path/to/video.mp3'
    main(camera_source)
