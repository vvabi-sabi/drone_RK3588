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
        self._count = 0
        self._begin = 0

    def _bgr2rgb(self, frame):
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        return frame

    def _bgr2gray(self, frame):
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        return frame

    @property
    def frames(self):
        cap = cv2.VideoCapture(self.source)
        if(not cap.isOpened()):
            print("Bad source")
            raise SystemExit
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Camera stopped!")
                    raise SystemExit
                yield frame
            cap.release()
        except Exception as e:
            print("Stop recording loop. Exception {}".format(e))

    def pre_process(self):
        for frame in self.frames:
            rgb_frame = self._bgr2rgb(frame)
            gray_frame = self._bgr2gray(frame)
            self._q_out.put((frame, [rgb_frame, gray_frame]))