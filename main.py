import json
import time
from pathlib import Path
from threading import Thread

import base.storages as strgs
from base import RK3588
from utils import fill_storages, show_frames_localy

CONFIG_FILE = str(Path(__file__).parent.absolute()) + "/config.json"
with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)

# raw_frames_storage = RawFrStrg()
# inferenced_frames_storage = InfFrStrg()
# detections_storage = DetectStrg()


def main():
    """Runs inference and addons (if mentions)
    Creating storages and sending data to them
    """
    rk3588 = RK3588()
    start_time = time.time()
    rk3588.start()
    if not cfg["storages"]["state"]:
        try:
            while True:
                rk3588.show(start_time)
        except Exception as e:
            print("Main exception: {}".format(e))
            exit()
    raw_frames_storage = strgs.ImageStorage(
        strgs.StoragePurpose.RAW_FRAME
    )
    inferenced_frames_storage = strgs.ImageStorage(
        strgs.StoragePurpose.INFERENCED_FRAME
    )
    detections_storage = strgs.DetectionsStorage()
    fill_thread = Thread(
        target=fill_storages,
        kwargs={
            "rk3588": rk3588,
            "raw_img_strg": raw_frames_storage,
            "inf_img_strg": inferenced_frames_storage,
            "dets_strg": detections_storage,
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
        raw_frames_storage.clear_buffer()
        inferenced_frames_storage.clear_buffer()
        detections_storage.clear_buffer()


if __name__ == "__main__":
    main()