import argparse
import sys

import aria.sdk as aria

import cv2
import numpy as np
from common import quit_keypress, update_iptables
from ultralytics import YOLO

from projectaria_tools.core.sensor_data import ImageDataRecord

'''
This file was originally created using bits of code from the 'streaming_start.py' and 'device_stream.py' files from the ARIA examples page: https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/samples

object detection inspired by: https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/
'''

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


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # STEP. Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # STEP. Connect to the device
    device = device_client.connect()

    # STEP. Retrieve the streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # STEP. Set custom config for streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name

    #    Note: by default streaming uses Wifi
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb

    #    Use ephemeral streaming certificates
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # STEP. Start streaming
    streaming_manager.start_streaming()

    # STEP. Get streaming state
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")

    #  STEP. Configure subscription to listen to Aria's RGB and SLAM streams.
    # @see StreamingDataType for the other data types
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.Slam | aria.StreamingDataType.EyeTrack | aria.StreamingDataType.Audio  
    )

    # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
    # For visualizing the images, we only need the most recent frame so set the queue size to 1
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.Slam] = 1

    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # STEP. Create and attach observer
    class StreamingClientObserver:
        def __init__(self):
            self.images = {}

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    # STEP. Start listening
    print("Start listening to image data")
    streaming_client.subscribe()

    # STEP. Visualize the streaming data until we close the window
    rgb_window = "Aria RGB"
    slam_window = "Aria SLAM"
    eye_window = 'Aria EYE'
    audio_window = 'Aria Audio'

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(slam_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slam_window, 480 * 2, 640)
    cv2.setWindowProperty(slam_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(slam_window, 1100, 400)

    cv2.namedWindow(eye_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_window, 820, 320)
    cv2.setWindowProperty(eye_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eye_window, 1100, 50)


    cv2.namedWindow(audio_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(audio_window, 820, 320)
    cv2.setWindowProperty(audio_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(audio_window, 1100, 50)
    # prepare classifier to use in object recognition
    model = YOLO("yolov8m.pt")
    
    while not quit_keypress():
        # Render the RGB image
        if aria.CameraId.Rgb in observer.images:
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # object recognition
            results = model.predict(rgb_image, conf = 0.5)
            rgb_image = results[0].plot()

            cv2.imshow(rgb_window, rgb_image)
            del observer.images[aria.CameraId.Rgb]

        if aria.CameraId.EyeTrack in observer.images:
            eye_image = np.rot90(observer.images[aria.CameraId.EyeTrack], 0)
            cv2.imshow(eye_window, eye_image)
            del observer.images[aria.CameraId.EyeTrack]

        # Stack and display the SLAM images
        if (
            aria.CameraId.Slam1 in observer.images
            and aria.CameraId.Slam2 in observer.images
        ):
            slam1_image = np.rot90(observer.images[aria.CameraId.Slam1], -1)
            slam2_image = np.rot90(observer.images[aria.CameraId.Slam2], -1)
            cv2.imshow(slam_window, np.hstack((slam1_image, slam2_image)))
            del observer.images[aria.CameraId.Slam1]
            del observer.images[aria.CameraId.Slam2]

    # STEP. Stop streaming and disconnect the device
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)

if __name__ == "__main__":
    main()