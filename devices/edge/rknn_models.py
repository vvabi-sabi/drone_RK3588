from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from multiprocessing import Process, Queue
from typing import Any
import numpy as np
from rknnlite.api import RKNNLite

ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS_PATH = str(ROOT) + "/models/"


class RKNNModelNames():
    NAMES_DICT = {'YOLO':'yolov5.rknn',
                  'ResNet':'resnet.rknn',
                  'UNet':'unet.rknn',
                  'Encoder':'encoder.rknn'
                  }
    
    #@classmethod
    def get_model_names(self, model_list):
        path_list = []
        for model in model_list:
            path_list.append(self.NAMES_DICT.get(model))
        return path_list


class RKNNModelLoader():
    """
    """
    
    def __init__(self, model_list, cores:list):
        self.verbose = False
        self.verbose_file = 'verbose.txt'
        self.rknn_name_list = RKNNModelNames().get_model_names(model_list)
        self._rknn_auto = self.rknn_name_list[0]
        if len(model_list) > 1:
            self.async_mode = True
        else:
            self.async_mode = False
        self._cores = cores
        self._core_auto = self._cores[0]
        self.rknn_list = self.load_rknn_models()

    def get_core(self):
        try:
            return self._cores.pop(0)
        except IndexError:
            return self._core_auto

    def get_model_path(self):
        try:
            return MODELS_PATH+self.rknn_name_list.pop(0)
        except IndexError:
            return self._rknn_auto

    def load_rknn_models(self):
        rknn_lite_list = []
        for i in range(len(self.rknn_name_list)):
            try:
                print(f'Init model - {i}')
                rknn_lite_list.append(self.get_rknn_model())
            except Exception as e:
                print("Cannot load rknn model. Exception {}".format(e))
                raise SystemExit
        return rknn_lite_list
    
    def get_rknn_model(self):
        core = self.get_core()
        model = self.get_model_path()
        rknnlite = RKNNLite(verbose=self.verbose,
                            verbose_file=self.verbose_file
                            )
        print(f"Export rknn model - {model}")
        ret = rknnlite.load_rknn(model)
        if ret != 0:
            print(f'Export {model} model failed!')
            return ret
        print('Init runtime environment')
        ret = rknnlite.init_runtime(async_mode=self.async_mode,
                                    core_mask = core
                                    )
        if ret != 0:
            print('Init runtime environment failed!')
            return ret
        print(f'{model} is loaded')
        return rknnlite


class ModelBuilder():
    
    def __init__(self, rknn_models, q_input):
        self._rknn_list = rknn_models
        self.net_list = self.build_models(q_input)
    
    def build_models(self, q_input, q_output):
        nets = []
        for rknn_model in self._rknn_list:
            q_output = Queue(maxsize=3)
            net = self.new('BaseModel', rknn_model, q_input, q_output)
            nets.append(net)
        return nets
    
    def new(self, model_name, rknn_model, q_input, q_output):
        return globals().get(model_name)(rknn_model, q_input, q_output)


class Inference(Process):
        def __init__(self, input, output):
            super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
            self.input = input
            self.output = output
        
        def run(self):
            while True:
                frame_list = self.input.get()
                outputs = []
                for frame in frame_list:
                    vector = self._rknnlite.inference(inputs=[frame])
                    outputs.append(vector)
                self.output.put(np.array(outputs))


class BaseModel():
    """
    """
    
    def __init__(self, rknn_model, q_input, q_output):
        self._rknnlite = rknn_model
        self.inference = Inference(q_input, q_output)