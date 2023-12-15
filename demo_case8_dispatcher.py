
import argparse
parser = argparse.ArgumentParser(description="Demo server side for")
parser.add_argument('--management-port', type=str, help='Management Port number', dest = 'management_port', default="8000")
parser.add_argument('--log-level', dest="log_level", help='Configuration file path')
parser.add_argument('--config', help='Configuration file path')
parser.add_argument('--log-screen', action='store_const', const=True, default=False, help='Enable log to screen', dest="log_screen")
parser.add_argument('--n-worker', help='num of workers', dest= "n_worker", type= int)
parser.add_argument('--port', help='port for data ingestion if itself not a producer', dest= "port", type= int)
args = parser.parse_args()

import configparser
config = configparser.ConfigParser()
if args.config:    
    config.read(args.config)
management_port = config.get("dispatcher", "management-port")
if management_port is None:
    management_port = args.management_port
management_port = int(management_port)
log_level = config.get("logger", "level")
if log_level is None:
    log_level = args.log_level
log_screen = config.get("logger", "logscreen")
if log_screen is None:
    log_screen = args.log_screen
if type(log_screen) is str:
    if log_screen.upper() == 'FALSE':
        log_screen = False
    elif log_screen.upper() == 'TRUE':
        log_screen = True
    else:
        raise ValueError(f"incorrect config for log_screen, expected UNION[FALSE, TRUE], get {log_screen}")
from pytaskqml.utils.confparsers import dispatcher_side_worker_config_parser
worker_config = dispatcher_side_worker_config_parser(config=config, parsed_args= args)

import logging
# Create a logger
modulelogger = logging.getLogger()
modulelogger.addHandler(logging.NullHandler)
logger = logging.getLogger("demo_case1")
logger.setLevel(log_level)

# Create a console handler and set its format

formatter = logging.Formatter("%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s",
                              )



# Add the console handler to the logger
if log_screen:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
#fh = logging.FileHandler(filename='demo_case1.log', mode='w', encoding='utf-8')
#fh.setLevel(log_level)
#fh.setFormatter(formatter)
#logger.addHandler(fh)
from pytaskqml.utils.logutils import QueueFileHandler
log_file_name = config.get("logger", "file")
qfh = QueueFileHandler(log_file_name)
qfh.setLevel(log_level)
qfh.setFormatter(formatter)
logger.addHandler(qfh)
import pytaskqml.task_dispatcher as task_dispatcher
task_dispatcher.modulelogger = logger
from dispatcher_side_demo_case8_classes import my_yolo_pytorch_dispatcher, my_yolo_pytorch_socket_producer_side_worker
import threading
import time
from types import MethodType
from pytaskqml.task_worker import base_socket_worker
class Send_Mixin():
    def __init__(self):
        self.min_time_lapse_ns = 0
        self.last_run = 0
        self.logger = logger
        self._send = MethodType(base_socket_worker._send, self)
my_send_class = Send_Mixin()    
def main():

    import sys
    print(sys.modules[__name__])

    my_worker_factory = my_yolo_pytorch_socket_producer_side_worker
    my_task_worker_manager = my_yolo_pytorch_dispatcher(
                worker_factory= my_worker_factory, 
                worker_config = worker_config,
                output_minibatch_size = 24,
                management_port = management_port,
                logger = logger,
                retry_watcher_on = True
            )
    
    th = threading.Thread(target = my_task_worker_manager.start, args = [])
    th.start()

    # this demo shows how to write own loop to dispatch data
    import hashlib
    logger.debug('signal start')

    def dispatch_from_main():
        # this function demonstrate main dispatch data such as video frame data or ref to video frame for example
        from pytaskqml.utils.sample.serialisers.plain_string_message_pb2 import MyMessage
        
        #print('dispatcher socket input found and connected')
        cnt = 0
        last_run = time.time()
        while not my_task_worker_manager.stop_flag.is_set():
            cnt+=1
            import cv2
            video_save_path = "sample.mp4"
            capture = cv2.VideoCapture(filename = video_save_path)
            #fps = capture.get(cv2.CAP_PROP_FPS)
            if video_save_path!="":
                fourcc  = cv2.VideoWriter_fourcc(*'XVID')
                size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            #    out     = cv2.VideoWriter(video_save_path, fourcc, fps, size)

            
            while True:
                ret, frame_data = capture.read()#{"frame_number": cnt, "face": np.random.rand(96,96,6), 'mel': np.random.rand(80,16)}
                if ret:
                    pass
                else:
                    break
                #from copy import deepcopy
                my_task_worker_manager.dispatch(frame_data)
                now_time = time.time()
                logger.debug(f'__main__ : cnt {cnt} added, interval = {now_time - last_run}')
                last_run = now_time
        
            
        
    th = threading.Thread(target = dispatch_from_main, args = [], name='dispatch_from_main')
    th.start()
    while not my_task_worker_manager.stop_flag.is_set():
        time.sleep(0.4)
    
    while True:
        try:
            my_task_worker_manager.graceful_stopped.wait(timeout=2)
        except:
            continue
        break
    qfh.stop_flag.set()
if __name__ == '__main__':
    main()