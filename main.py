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
from screeninfo import get_monitors  # Added for getting screen resolution

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

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:f
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
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")

    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to image data")
    streaming_client.subscribe()

    rgb_window = "Object Detection"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get screen resolution using screeninfo
    monitor = get_monitors()[0]  # Assuming a single monitor setup
    screen_width, screen_height = monitor.width, monitor.height

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
        print("Stop listening to image data")
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

            # Resize image to screen resolution
            rgb_image = cv2.resize(rgb_image, (screen_width, screen_height))

            results = model(rgb_image, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(rgb_image, classNames[cls], org, font, fontScale, color, thickness)

            cv2.imshow(rgb_window, rgb_image)
            del observer.images[aria.CameraId.Rgb]

        if cv2.waitKey(1) == ord('q'):
            break

    signal_handler(None, None)

if __name__ == "__main__":
    main()
