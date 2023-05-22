from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite

ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS_PATH = str(ROOT) + "/models/"

class ModelNames():
    NAMES_DICT = {'YOLO':'yolov5.rknn',
                  'ResNet':'resnet.rknn',
                  'UNet':'unet.rknn',
                  'Encoder':'encoder.rknn'
                  }
    #@classmethod
    def get_path(self, models_list):
        path_list = []
        for model in models_list:
            path_list.append(self.NAMES_DICT.get(model))
        return path_list

class ModelLoader():
    """
    """
    
    def __init__(self, models_list, cores:list):
        self.verbose = False
        self.verbose_file = 'verbose.txt'
        self.model_names_list = ModelNames().get_path(models_list)
        if len(models_list) > 1:
            self.async_mode = True
        else:
            self.async_mode = False
        self.nets = []
        self._cores = cores

    def load_models(self):
        for n, model_name in enumerate(self.model_names_list):
            try:
                self.nets.append(self.load(MODELS_PATH+model_name),
                                 self._cores[n]
                                )
            except Exception as e:
                print("Cannot load model. Exception {}".format(e))
                raise SystemExit
            return self.nets
    
    def load(self, model: str, core: int):
        rknnlite = RKNNLite(verbose=self.verbose,
                            verbose_file=self.verbose_file
                            )
        print(f"Export rknn model - {model}")
        ret = rknnlite.load_rknn(model)
        if ret != 0:
            print(f'Export {model} model failed!')
            return ret
        print('Init runtime environment')
        ret = rknnlite.init_runtime(
                    async_mode=self.async_mode,
                    core_mask = core
                )
        if ret != 0:
            print('Init runtime environment failed!')
            return ret
        print(f'{model} is loaded')
        return rknnlite


class Yolov5(ModelsFactory):
    pass


class UNet(ModelsFactory):
    pass


class ResNet(ModelsFactory):
    pass


class AutoEncoder(ModelsFactory):
    
    def inference(self, q_in, q_out):
        # TODO
        # add multi_input/output, C-Python insert
        # VAE model
        # RGB input
        while True:
            frame, raw_frame, frame_id = q_in.get()
            imgs_list = self._pre_process(frame) # len = 18
            outputs = []
            for transform_img in imgs_list:
                input_img = transform_img
                vector = self._rknnlite.inference(inputs=[input_img])
                vector = vector[0].reshape(2000)
                outputs.append(vector)
            outputs = np.array(outputs)
            q_out.put((outputs, raw_frame, frame_id))

