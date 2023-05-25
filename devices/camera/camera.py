import os
import json
import time
from datetime import datetime
from multiprocessing import Queue, Value
from pathlib import Path

import cv2
import numpy as np


class Camera():
    """
    """
    def __init__(self, source: int, q_in: Queue, model_list: list):
        self._q_in = q_in
        self.source = source
        self.model_list = model_list
        self.frame = None
        self._count = 0
        self._begin = 0

    def cap_set(self, cap):
        if os.path.isfile(self.source) is False:
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter.fourcc(*cfg["camera"]["pixel_format"])
            )
            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                cfg["camera"]["width"]
            )
            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                cfg["camera"]["height"]
            )
            cap.set(
                cv2.CAP_PROP_FPS,
                cfg["camera"]["fps"]
            )
        return cap

    def _bgr2rgb(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(
        #     frame,
        #     (cfg["inference"]["net_size"], cfg["inference"]["net_size"])
        # )
        return frame

    def record(self):
        cap = cv2.VideoCapture(self.source)
        cap = self.cap_set(cap)
        if(not cap.isOpened()):
            print("Bad source")
            raise SystemExit
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Camera stopped!")
                    raise SystemExit
                self.frame = frame
                #frame = self._bgr2rgb(frame)
                #self._q_out.put((frame, frame_processed))
            cap.release()
        except Exception as e:
            print("Stop recording loop. Exception {}".format(e))

    

    def show(self, start_time):
        raw_frame, frame, _, frame_id = self._q_in.get()
        self._count+=1
        if frame_id < self._last_frame_id:
            return
        if self._count % 30 == 0:
            self._fps = 30/(time.time() - self._begin)
            if self._fps > self._max_fps:
                self._max_fps = self._fps
            self._begin = time.time()

        frame = cv2.putText(
            img = frame,
            text = "id: {}".format(frame_id),
            org = (5, 30),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (0,255,0), 
            thickness = 1,
            lineType = cv2.LINE_AA
        )
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        if cfg["debug"]["showed_frame_id"]:
            with open(cfg["debug"]["showed_id_file"], 'a') as f:
                f.write(
                    "{}\t{:.3f}\n".format(
                        frame_id,
                        time.time() - start_time
                    )
                )

    def release(self):
        self._stop_record.value = 1 # type: ignore