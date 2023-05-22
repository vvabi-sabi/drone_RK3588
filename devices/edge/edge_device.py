from multiprocessing import Process, Queue
from rknnlite.api import RKNNLite

from rknn_models import ModelLoader


class NeuroModule():
    
    def __init__(self, cores_list, models_list):
        self.model_loader =  ModelLoader()
        self.net_list = self.model_loader(models_list)
        pass

    def inference(self, frame):
        output = self.model.inference(frame)


class EdgeDevice():
    """
    """
    _CORES = [RKNNLite.NPU_CORE_0,
              RKNNLite.NPU_CORE_1,
              RKNNLite.NPU_CORE_2,
              RKNNLite.NPU_CORE_AUTO
              ]

    def __init__(self, camera, models_list, queue):
        self._camera = camera
        self._queue = queue
        self._neuro = NeuroModule(self._CORES, models_list) # ['Model_1', 'Model_2'])
        
    def start(self):
        self._rec.start()
        for inference in self._pre_inf: inference.start()
        for post_process in self._post: post_process.start()
        self._map.start()

    def show(self, start_time):
        self._cam.show(start_time)

    def get_data(self):
        if self._q_post.empty():
            return None
        return self._q_post.get()
