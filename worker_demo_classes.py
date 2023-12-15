
from pytaskqml.task_worker import base_socket_worker
from typing import List

class torch_yolov4_worker(base_socket_worker):
    def __init__(self):
        pass
    def workload(self, received_data):
        from yolo_detection_pb2 import RequestData,  ResponseData
        request_data = RequestData()
        request_data.ParseFromString(received_data)


        response_data = ResponseData()
        ResponseData.messageuuid = response_data.messageuuid
        ResponseData.Detection : List
        serialized_data = ResponseData.SerializeToString()
        return serialized_data
    pass
# message bbc{
#   float x_center = 1;
#   float y_center = 2;
#   int32 width = 3;
#   int32 height = 4;
# }

# message Detection{
#   string classlabel = 1;
#   bbc bbc = 2;
#   float confidence = 3;
# }


# class echo_worker(base_socket_worker):
#     def workload(self, received_data):
#         from pytaskqml.utils.sample.serialisers.plain_string_message_pb2 import plain_string_message_pb2
#         request_array = plain_string_message_pb2.MyMessage()
#         request_array.ParseFromString(received_data)
#         request_array.my_field = "hello" + request_array.my_field
#         response_data_array = request_array
#         serialized_data = response_data_array.SerializeToString()
#         return serialized_data

# class word_count_worker(base_socket_worker):

#     def workload(self, received_data):
#         from pytaskqml.utils.sample.serialisers.plain_string_message_pb2 import MyMessage
#         request_message = MyMessage()
#         request_message.ParseFromString(received_data)
        
#         import pickle
#         import re
#         from collections import Counter
#         delimiters = [",", ";", "|", "."]

# # Create a regular expression pattern by joining delimiters with the "|" (OR) operator
#         pattern = "|".join(map(re.escape, delimiters))

#         # Split the string using the pattern as the delimiter
#         serialized_data = pickle.dumps({'uuid': request_message.messageuuid, 'counts': Counter(re.split(pattern, request_message.strmessage))})
#         return serialized_data
