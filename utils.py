import json
import time
from pathlib import Path

import cv2

import base.storages as strgs
from base import RK3588


CONFIG_FILE = str(Path(__file__).parent.absolute()) + "/config.json"
with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)


def fill_storages(
        rk3588: RK3588,
        raw_img_strg: strgs.ImageStorage=None,
        inf_img_strg: strgs.ImageStorage=None,
        dets_strg: strgs.DetectionsStorage=None, # YOLOv5
        coords_strg: strgs.CoordinatesStorage=None, # ResNet
        start_time: float=None
):
    """Fill storages with raw frames, frames with bboxes, numpy arrays with
    detctions
    Args
    -----------------------------------
    rk3588: Rk3588
        Object of Rk3588 class for getting data after inference
    raw_img_strg: storages.ImageStorage
        Object of ImageStorage for storage raw frames
    inf_img_strg: storages.ImageStorage
        Object of ImageStorage for storage inferenced frames
    dets_strg: storages.DetectionsStorage
        Object of DetectionsStorage for numpy arrays with detctions
    start_time: float
        Program start time
    -----------------------------------
    """
    storages = [strg for strg in [coords_strg, dets_strg, inf_img_strg, raw_img_strg] if strg is not None]
    while True:
            output = rk3588.get_data()
            if output is not None:
                #print('output', output)
                frame_id = output.pop()
                for strg in storages:
                    strg.set_data(
                        data=output.pop(),
                        id=frame_id,
                        start_time=start_time
                    )


def show_frames_localy(
        inf_img_strg: strgs.ImageStorage,
        start_time: float
):
    """Show inferenced frames with fps on device
    
    Args
    -----------------------------------
    inf_img_strg: storages.ImageStorage
        Object of ImageStorage for storage inferenced frames
    start_time: float
        Program start time
    -----------------------------------
    """
    cur_index = -1
    counter = 0
    calculated = False
    begin_time = time.time()
    fps = 0
    cfg_fps = cfg["camera"]["fps"]
    stored_data_amount = cfg["storages"]["stored_data_amount"]
    while True:
        last_index = inf_img_strg.get_last_index()
        if cfg["debug"]["showed_frame_id"] and cur_index != last_index:
            with open(cfg["debug"]["showed_id_file"], 'a') as f:
                f.write("{}\t{:.3f}\n".format(cur_index,
                                              time.time() - start_time
                                            )
                        )
        print(f"cur - {cur_index} last - {last_index}")
        frame =\
            inf_img_strg.get_data_by_index(last_index % stored_data_amount)
        if cfg["camera"]["show"]:
            cv2.putText(
                img=frame,
                text="{:.2f}".format(fps),
                org=(5, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        if last_index > cur_index:
            counter += 1
            cur_index = last_index
        if counter % cfg_fps == 0 and not calculated:
            calculated = True
            fps = cfg_fps/(time.time() - begin_time)
            begin_time = time.time()
        if counter % cfg_fps != 0:
            calculated = False
