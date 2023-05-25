from multiprocessing import Process, Queue
from rknnlite.api import RKNNLite

from rknn_models import RKNNModelLoader, ModelBuilder


class NeuroModule():
    
    _PROCESSES_NUMBER = 3

    def __init__(self, cores_list, models_list, q_input):
        self.rknn_loader =  RKNNModelLoader(models_list, cores_list)
        self.rknn_list = self.rknn_loader.rknn_list
        self.model_builder = ModelBuilder(models_list, self.rknn_list, q_input)
        self.net_list = self.model_builder.net_list
        self.nets = self.net_init()

    def get_nets(self):
        n = 0
        while True:
            if n >= len(self.net_list):
                yield self.net_list[0]
            else:
                yield self.net_list[n]
            n += 1

    def net_init(self):
        net_list = []
        nets = self.get_nets()
        for _ in range(self._PROCESSES_NUMBER):
            net_list.append(next(nets))
        return net_list

    def forward(self):
        outputs = []
        for net in self.nets:
            outputs.append(net.inference.output.get())
        return outputs

    def run_inference(self):
        for net in self.nets:
            inf_process = net.inference
            inf_process.start()


class EdgeDevice():
    """
    """
    _CORES = [RKNNLite.NPU_CORE_0,
              RKNNLite.NPU_CORE_1,
              RKNNLite.NPU_CORE_2,
              RKNNLite.NPU_CORE_AUTO
              ]

    def __init__(self, camera, models_list):
        self._camera = camera
        self._neuro = NeuroModule(self._CORES[3],
                                  models_list,
                                  self._camera._queue) # ['Model_1', 'Model_2'])

    def get_neuro_outputs(self):
        outputs = self._neuro.forward()
        return outputs
