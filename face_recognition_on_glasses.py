import os
import cv2
import face_recognition
import numpy as np
import pickle
from gtts import gTTS
import pygame
import argparse
import sys
import signal
import aria.sdk as aria
from common import update_iptables
from projectaria_tools.core.sensor_data import ImageDataRecord
from queue import Queue
from tqdm import tqdm
from collections import defaultdict

class Face:
    def __init__(self, name, feature_vector):
        self.name = name
        self.feature_vector = feature_vector

def detect_faces(image_test, faces, known_names, threshold=0.6):
    locs_test = face_recognition.face_locations(image_test, model='hog')
    vecs_test = face_recognition.face_encodings(image_test, locs_test, num_jitters=1)
    
    for loc_test, vec_test in zip(locs_test, vecs_test):
        distances = [face_recognition.face_distance([face.feature_vector], vec_test)[0] for face in faces]
        
        if np.min(distances) > threshold:
            pred_name = 'unknown'
        else:
            pred_index = np.argmin(distances)
            pred_name = faces[pred_index].name
        
        if pred_name not in known_names:
            known_names.add(pred_name)
            print(pred_name)
        
        image_test = draw_bounding_box(image_test, loc_test)
        image_test = draw_name(image_test, loc_test, pred_name)
    
    return image_test

def draw_bounding_box(image_test, loc_test):
    top, right, bottom, left = loc_test
    line_color = (0, 255, 0)
    line_thickness = 2
    cv2.rectangle(image_test, (left, top), (right, bottom), line_color, line_thickness)
    return image_test

def draw_name(image_test, loc_test, pred_name):
    top, right, bottom, left = loc_test 
    font_style = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    font_thickness = 1
    
    text_size, _ = cv2.getTextSize(pred_name, font_style, font_scale, font_thickness)
    bg_top_left = (left, top - text_size[1])
    bg_bottom_right = (left + text_size[0], top)
    line_color = (0, 255, 0)
    line_thickness = -1

    cv2.rectangle(image_test, bg_top_left, bg_bottom_right, line_color, line_thickness)   
    cv2.putText(image_test, pred_name, (left, top), font_style, font_scale, font_color, font_thickness)
    
    return image_test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", dest="streaming_interface", type=str, required=True, help="Type of interface to use for streaming. Options are usb or wifi.", choices=["usb", "wifi"])
    parser.add_argument("--update_iptables", default=False, action="store_true", help="Update iptables to enable receiving the data stream, only for Linux.")
    parser.add_argument("--profile", dest="profile_name", type=str, default="profile18", required=False, help="Profile to be used for streaming.")
    parser.add_argument("--device-ip", help="IP address to connect to the device over wifi")
    return parser.parse_args()

class StreamingClientObserver:
    def __init__(self):
        self.images = {}
        self.identifications = set()  
        self.object_queue = Queue()  

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    device = device_client.connect()

    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    streaming_manager.start_streaming()
    print(f"Streaming state: {streaming_manager.streaming_state}")

    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.security_options.use_ephemeral_certs = True
    streaming_client.subscription_config = config

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()
    rgb_window = "Face Recognition"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 640, 480)

    with open('trained_faces.pkl', 'rb') as f:
        faces = pickle.load(f)
    
    known_names = set()

    def signal_handler(sig, frame):
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        if aria.CameraId.Rgb in observer.images:
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            frame_with_faces = detect_faces(rgb_image, faces, known_names)
            
            bgr_frame_with_faces = cv2.cvtColor(frame_with_faces, cv2.COLOR_RGB2BGR)
            
            cv2.imshow(rgb_window, bgr_frame_with_faces)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    signal_handler(None, None)

if __name__ == "__main__":
    main()