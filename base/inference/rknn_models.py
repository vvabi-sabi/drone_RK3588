import json
import os
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite

from base.pre_process import pre_yolov5, pre_unet, pre_resnet, pre_autoencoder
from base.post_process import post_yolov5, post_unet, post_resnet, post_autoencoder

ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS = str(ROOT) + "/models/"
CONFIG_FILE = str(ROOT) + "/config.json"
RESNET_INPUT_SIZE = 64
UNET_INPUT_SIZE = 224

with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)


class ModelsFactory():
    """Class for inference on RK3588/RK3588S
    Args
    ---------------------------------------------------------------------------
    proc : int
        Number of running process for inference
    core : int
        Index of NPU core(s) that will be used for inference
        default (RKNNLite.NPU_CORE_AUTO) : sets all cores for inference 
        one frame
    ---------------------------------------------------------------------------
    Attributes
    ---------------------------------------------------------------------------
    _core : int
        Index of NPU core that will be used for inference
    _proc : int
        Number of running process for inference
    ---------------------------------------------------------------------------
    Methods
    ---------------------------------------------------------------------------
    _load_model(model: str) : Literal[-1, 0]
        Load rknn model on RK3588/RK3588S device
    inference() : NoReturn
        inference resized raw frames
    ---------------------------------------------------------------------------
    """
    
    def __init__(
            self,
            proc: int,
            core: int = RKNNLite.NPU_CORE_AUTO
        ):
        self.MODEL_PATH = MODELS + cfg["inference"]["default_model"]
        self._core = core
        self._proc = proc
        #Check new model loaded
        try:
            self._rknnlite = self.load_model(
                                    MODELS + cfg["inference"]["new_model"],
                                    self._core
                                )
        except Exception as e:
            print("Cannot load model. Exception {}".format(e))
            raise SystemExit
        #Load pre/post processes
        self._pre_process, self._post_process = self.__class__.load_processes()

    def load_model(self, model: str, core: int):
        if os.path.isfile(model) is False:
            model = self.MODEL_PATH
        rknnlite = RKNNLite(
                verbose=cfg["debug"]["verbose"],
                verbose_file=str(ROOT) + "/" + cfg["debug"]["verbose_file"]
                )
        print("Export rknn model")
        ret = rknnlite.load_rknn(model)
        if ret != 0:
            print(f'Export {model} model failed!')
            return ret
        print('Init runtime environment')
        ret = rknnlite.init_runtime(
                    async_mode=cfg["inference"]["async_mode"],
                    core_mask = core
                )
        if ret != 0:
            print('Init runtime environment failed!')
            return ret
        print(f'{model} model loaded' )
        return rknnlite

    @classmethod
    def load_processes(Class):
        if Class.__name__ == 'Yolov5':
            return pre_yolov5, post_yolov5
        elif Class.__name__ == 'UNet':
            return pre_unet, post_unet
        elif Class.__name__ == 'ResNet':
            return pre_resnet, post_resnet
        elif Class.__name__ == 'AutoEncoder':
            return pre_autoencoder, post_autoencoder

    def inference(self, q_in, q_out):
        while True:
            frame, raw_frame, frame_id = q_in.get()
            frame = self._pre_process(frame)
            outputs = self._rknnlite.inference(inputs=[frame])
            q_out.put((outputs, raw_frame, frame_id))
    
    def post_process(self, q_in, q_out):
        while True:
            outputs, raw_frame, frame_id = q_in.get()
            frame = raw_frame.copy()
            results = self._post_process(outputs, frame)
            q_out.put([raw_frame, *results, frame_id])


class Yolov5(ModelsFactory):
    pass


class UNet(ModelsFactory):
    pass


class ResNet(ModelsFactory):
    pass


class AutoEncoder(ModelsFactory):
    
    def inference(self, q_in, q_out):
        while True:
            frame, raw_frame, frame_id = q_in.get()
            imgs_list = self._pre_process(frame) # len = 80
            for transform_img in imgs_list:
                vector = self._rknnlite.inference(inputs=[transform_img])
                outputs.append(vector)
            outputs = np.array(outputs)
            q_out.put((outputs, raw_frame, frame_id))

