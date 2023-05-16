from multiprocessing import Process, Queue

from utils import PostProcesses, Odometry, Visualizer
from devices import Cam, RK3588


def run(device, visualizer, post_process, odometry):
    device.start()
    if post_process != None and odometry is None:
        while True:
            frame, outputs = device.get_outputs()
            frame = post_process(frame, outputs)
            visualizer.show_frame(frame)
    elif odometry != None and post_process is None:
        while True:
            _, outputs = device.get_outputs()
            coords = odometry(outputs)
            visualizer.show_trajectory(coords)
    if post_process != None  and odometry != None:
        while True:
            frame, outputs = device.get_outputs()
            post_process(frame, outputs)
            coords = odometry(outputs)
            visualizer.show_trajectory(coords)
            visualizer.show_frame(frame)
    else:
        while True:
            outputs = device.get_outputs()

def main(source):
    """Runs inference and addons (if mentions)
    Creating storages and sending data to them
    """
    q_pre = Queue(maxsize=cfg["inference"]["buf_size"])
    #q_outs = Queue(maxsize=cfg["inference"]["buf_size"])
    q_post = Queue(maxsize=cfg["inference"]["buf_size"])
    
    models = ['YOLO'] # 'ResNet', 'UNet', 'Encoder'
    params = {'post_process': True, 'show_trajectory': False}
    
    camera = Cam(source = source,
                 q_in = q_post,
                 q_out = q_pre)
    visualizer = Visualizer()
    device = RK3588(camera, models)
    post_processes = PostProcesses(models, params['post_process'])
    odometry_algs = Odometry(models, params['show_trajectory'])
    try:
        run(device, visualizer, post_processes, odometry_algs)
    except Exception as e:
        print("Main exception: {}".format(e))
        exit()


if __name__ == "__main__":
    main()
