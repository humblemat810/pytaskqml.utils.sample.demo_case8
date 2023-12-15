import numpy as np
from yolov4_tiny_pytorch.yolo import YOLO
from pytaskqml.task_worker import base_socket_worker
import cv2

from PIL import Image

class yolo_v4_worker(base_socket_worker):
    def __init__(self, *arg, **kwarg):
        ## yolov4_tiny_pytorch stuff:
        mode = "predict"
        crop            = False
        count           = False
        self.yolo : YOLO = YOLO(crop = crop, mode = mode, count = count)
        super().__init__(*arg, **kwarg)
        # copied for reference, not useful for non local video capture
        # video_path      = 0
        # video_save_path = ""
        # video_fps       = 25.0
        # test_interval   = 100
        # fps_image_path  = "img/street.jpg"
        # dir_origin_path = "img/"
        # dir_save_path   = "img_out/"
        # heatmap_save_path = "model_data/heatmap_vision.png"
    def workload(self, received_data):
        from pytaskqml.utils.sample.serialisers.yolo_detection_pb2 import RequestData
        request_message = RequestData()
        request_message.ParseFromString(received_data)
        width: int = int(request_message.width)
        height : int = int(request_message.height)
        frame : np.ndarray = np.array(request_message.image).reshape((width, height,3))
        
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        predictions = self.yolo.detect_image(frame, bbox_only = True)

        from pytaskqml.utils.sample.serialisers.yolo_detection_pb2 import ResponseData
        ResponseData.messageuuid = request_message.messageuuid
        ResponseData.Detection = predictions
        serialized_detection = ResponseData.SerializeToString()
        return serialized_detection
    
if __name__ == "__main__":
    yolo_v4_worker()