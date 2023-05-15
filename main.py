import json
import time
from pathlib import Path

from firefly import RK3588
from camera import Cam

CONFIG_FILE = str(Path(__file__).parent.absolute()) + "/config.json"
with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)


def main(source):
    """Runs inference and addons (if mentions)
    Creating storages and sending data to them
    """
    kwargs = {'models': ['YOLO'], 'post_process': True, 'show_trajectory': False}
    camera = Cam(source = source,
                 q_in = q_post,
                 q_out = q_pre)
    device = RK3588(camera, **kwargs)
    device.start()
    try:
        while True:
            camera.show(time.time())
    except Exception as e:
        print("Main exception: {}".format(e))
        exit()


if __name__ == "__main__":
    main()
