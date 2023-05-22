from multiprocessing import Process, Queue
from rknnlite.api import RKNNLite

from rknn_models import RKNNModelLoader, ModelBuilder


class NeuroModule():
    
    _PROCESSES_NUMBER = 3

    def __init__(self, cores_list, models_list):
        self.rknn_loader =  RKNNModelLoader(models_list, cores_list)
        self.rknn_list = self.rknn_loader.rknn_list
        self.model_builder = ModelBuilder(models_list, self.rknn_list)
        self.net_list = self.model_builder.net_list
        self._processes_number = len(self.net_list)
        pass

    def get_nets(self):
        n = 0
        while True:
            if n >= len(self.net_list):
                yield self.net_list[0]
            else:
                yield self.net_list[n]
            n += 1

    def run_inference(self):
        nets = self.get_nets()
        for proc in range(self._PROCESSES_NUMBER):
            inf = Process(target=next(nets).inference, # pre_process + inf
                          kwargs={'q_in' : self._q_pre,
                                  'q_out' : self._q_outs,
                                  },
                          daemon=True)
            inf.start()
            print(f'Process {proc} - start inference.')

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
        self._neuro = NeuroModule(self._CORES[3], models_list) # ['Model_1', 'Model_2'])

    def start(self):
        #self._neuro.run_inference()
        pass

    def get_neuro_outputs(self):
        self._neuro.get_outputs()

    def show(self, start_time):
        self._cam.show(start_time)

    def get_data(self):
        if self._q_post.empty():
            return None
        return self._q_post.get()
