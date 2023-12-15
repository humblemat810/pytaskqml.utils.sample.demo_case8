import logging

# Create a logger
logger = logging.getLogger()

from pytaskqml.task_dispatcher import Socket_Producer_Side_Worker, Task_Worker_Manager
class my_yolo_pytorch_dispatcher(Task_Worker_Manager):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        from collections import Counter
        self.count_dict = Counter()
    def on_shutdown(self):
        for k,v in self.count_dict.items():
            print(k,v)
        return super().on_shutdown()

    def send_output(self, fresh_output_minibatch):
        for task_info, counter in fresh_output_minibatch:
            self.count_dict += counter
            # for k in counter.keys():
            #     print(k,self.count_dict.get(k))
        pass
    pass


class my_yolo_pytorch_socket_producer_side_worker(Socket_Producer_Side_Worker):
    
    def _parse_task_info(self, single_buffer_result):
        from pytaskqml.utils.sample.serialisers.yolo_detection_pb2 import ResponseData
        uuid = ResponseData.messageuuid
        predictions = ResponseData.Detection
        
        task_time = self.uuid_to_time.get(uuid)
        if task_time is not None:
            task_info = (self.uuid_to_time[uuid], uuid)
            success = True
            map_result = predictions
        else: # previous retry already solved in a race condition
            success = False
            task_info = None
            map_result = None
        
        return success, task_info, map_result
    
    def dispatch(self, task_info, *args, **kwargs):
        # this has to be implemented to tell what exactly to do to dispatch data to worker
        # such as what protocol e.g. protobuf, to use to communicate with worker and also #
        # transfer data or ref to data
        from pytaskqml.utils.sample.serialisers.yolo_detection_pb2 import RequestData
        message = RequestData()
        message.messageuuid = task_info[0][1]
        img = task_info[1]
        message.height, message.width, _ = img.shape
        print("added size to image", task_info[1].shape)
        message.image.extend(task_info[1].reshape([-1]))
        
        serialized_data = message.SerializeToString()
        
        return serialized_data