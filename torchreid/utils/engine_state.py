import queue
from heapq import heappush


class EngineStateListener:
    def batch_completed(self):
        pass

    def epoch_started(self):
        pass

    def epoch_completed(self):
        pass

    def training_started(self):
        pass

    def training_completed(self):
        pass

    def test_completed(self):
        pass

    def run_completed(self):
        pass


class EngineState(EngineStateListener):
    __main_engine_state = None  # type: Optional[EngineState]

    @classmethod
    def current_engine_state(cls):
        # type: () -> EngineState
        return cls.__main_engine_state

    def __init__(self, start_epoch, max_epoch):
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.epoch = start_epoch
        self.batch = 0
        self.global_step = 0
        self.estimated_num_batches = 0
        self.lr = 0
        self.is_training = False
        self.listeners = []
        self.last_listeners = []

        EngineState.__main_engine_state = self

    def add_listener(self, listener, last=False):
        # FIXME ugly
        if last:
            self.last_listeners.append(listener)
        else:
            self.listeners.append(listener)

    def batch_completed(self):
        for listener in self.listeners + self.last_listeners:
            listener.batch_completed()
        self.batch += 1
        self.global_step += 1

    def epoch_started(self):
        for listener in self.listeners + self.last_listeners:
            listener.epoch_started()
        self.batch = 0

    def epoch_completed(self):
        for listener in self.listeners + self.last_listeners:
            listener.epoch_completed()
        if self.epoch != self.max_epoch - 1:
            self.epoch += 1

    def training_started(self):
        for listener in self.listeners + self.last_listeners:
            listener.training_started()
        self.is_training = True

    def training_completed(self):
        for listener in self.listeners + self.last_listeners:
            listener.training_completed()
        self.is_training = False

    def test_completed(self):
        for listener in self.listeners + self.last_listeners:
            listener.test_completed()

    def run_completed(self):
        for listener in self.listeners + self.last_listeners:
            listener.run_completed()

    def update_lr(self, lr):
        self.lr = lr