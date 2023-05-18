from edge_device import EdgeDevice


class RK3588(EdgeDevice):
    """
    """
    def __init__(self, camera, models, queue):
        super().__init__( camera, models, queue)
