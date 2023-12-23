import logging

# Create a logger
logger = logging.getLogger()

from pytaskqml.task_dispatcher import Socket_Producer_Side_Worker, Task_Worker_Manager
class my_yolo_pytorch_dispatcher(Task_Worker_Manager):
    def send_output(self, fresh_output_minibatch):
        for task_info, frame_detection in fresh_output_minibatch:
            for d in frame_detection:
                print(d['classlabel'])
                self.logger.info(f"{d['classlabel']} found with confidence {d['confidence']} with bb {d['bbc']}")
        
    


class my_yolo_pytorch_socket_producer_side_worker(Socket_Producer_Side_Worker):
    def parse_task_info(self, serialized_data):
        
        from yolo_detection_pb2 import ResponseData
        
        
        
        response_data = ResponseData()
        response_data.ParseFromString(serialized_data)
        uuid = response_data.messageuuid        
        predictions = response_data.detection
        prediction_parsed = [{
            "bbc": {'x_center': i.bbc.x_center, "y_center": i.bbc.y_center, 'width': i.bbc.width, 'height': i.bbc.height},
            'classlabel' : i.classlabel,
            'confidence': i.confidence
        } for i in predictions]
        return prediction_parsed


    
    def dispatch(self, task_info, *args, **kwargs):
        # this has to be implemented to tell what exactly to do to dispatch data to worker
        # such as what protocol e.g. protobuf, to use to communicate with worker and also #
        # transfer data or ref to data
        import pickle
        img = task_info[1]
        serialized_data = pickle.dumps({
            "messageuuid": task_info[0][1],
            "height": img.shape[0], 
            "width"  :img.shape[1],
            'image' : img
        })
        return serialized_data
        from pytaskqml.utils.sample.serialisers.yolo_detection_pb2 import RequestData
        message = RequestData()
        message.messageuuid = task_info[0][1]
        img = task_info[1]
        message.height, message.width, _ = img.shape
        print("added size to image", task_info[1].shape)
        message.image.extend(img.reshape([-1]))
        
        serialized_data = message.SerializeToString()
        
        return serialized_data