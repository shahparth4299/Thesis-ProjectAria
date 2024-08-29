import argparse
import sys
import signal
import aria.sdk as aria
import cv2
import numpy as np
import math
from ultralytics import YOLO
from common import update_iptables
from projectaria_tools.core.sensor_data import ImageDataRecord
from screeninfo import get_monitors
from queue import Queue
import threading
from tts_speaker import speak_text  

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()

class StreamingClientObserver:
    def __init__(self):
        self.images = {}
        self.identifications = set()  # HashSet to store identified class names
        self.object_queue = Queue()   # Queue for identified objects

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    # Initialize DeviceClient and configure connection
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    device = device_client.connect()

    # Set up streaming manager and client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # Start streaming
    streaming_manager.start_streaming()
    print(f"Streaming state: {streaming_manager.streaming_state}")

    # Configure subscription to RGB data
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.security_options.use_ephemeral_certs = True
    streaming_client.subscription_config = config

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    print("Start listening to image data")

    # Set up the display window with smaller dimensions
    rgb_window = "Object Detection"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 640, 480)  # Smaller window dimensions

    model = YOLO("yolov8m.pt")
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                 ]

    def signal_handler(sig, frame):
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        cv2.destroyAllWindows()
        sys.exit(0)

    speak_text("Objects Identified in front of you are")
    def tts_worker():
        while True:
            if not observer.object_queue.empty():
                identified_object = observer.object_queue.get()
                speak_text(f"{identified_object}")  # Assuming speak_text handles TTS

    signal.signal(signal.SIGINT, signal_handler)

    # Start a thread for TTS
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    while True:
        if aria.CameraId.Rgb in observer.images:
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Resize image to window dimensions
            rgb_image = cv2.resize(rgb_image, (640, 480))

            results = model(rgb_image, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    # Store identified class names in HashSet to avoid repetition
                    if class_name not in observer.identifications:
                        observer.identifications.add(class_name)
                        observer.object_queue.put(class_name)

                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(rgb_image, class_name, org, font, fontScale, color, thickness)

            cv2.imshow(rgb_window, rgb_image)
            del observer.images[aria.CameraId.Rgb]

        if cv2.waitKey(1) == ord('q'):
            break

    signal_handler(None, None)

if __name__ == "__main__":
    main()
