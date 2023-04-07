import json
import time
from pathlib import Path
from threading import Thread

from base import Rk3588
from utils import fill_storages, show_frames_localy

CONFIG_FILE = str(Path(__file__).parent.absolute()) + "/config.json"
with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)

class InfFrStrg():
    pass

inferenced_frames_storage = InfFrStrg()


def main():
    """Runs inference and addons (if mentions)
    Creating storages and sending data to them
    """
    rk3588 = Rk3588()
    start_time = time.time()
    rk3588.start()
    if not cfg["storages"]["state"]:
        try:
            while True:
                rk3588.show(start_time)
        except Exception as e:
            print("Main exception: {}".format(e))
            exit()
    fill_thread = Thread(
        target=fill_storages,
        kwargs={
            "rk3588": rk3588,
            "inf_img_strg": inferenced_frames_storage,
            "start_time": start_time
        },
        daemon=True
    )
    fill_thread.start()
    
    try:
        show_frames_localy(
            inf_img_strg=inferenced_frames_storage,
            start_time=start_time
        )
    except Exception as e:
        print("Main exception: {}".format(e))
    finally:
        inferenced_frames_storage.clear_buffer()


if __name__ == "__main__":
    main()