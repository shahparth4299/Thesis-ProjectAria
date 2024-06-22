import argparse
import sys
import signal
import aria.sdk as aria
import cv2
import numpy as np
from common import update_iptables
from projectaria_tools.core.sensor_data import ImageDataRecord
from screeninfo import get_monitors
from easy_ocr_read import read_text_from_image
from tts_speaker import speak_text
import threading
import pvporcupine
import pyaudio
import speech_recognition as sr

ACCESS_KEY = '2HEVbzGtkFD6kt8RmmIoiIrZLdzT6ViysO/an4cKZoo9Taw2vx/76A=='
exit_flag = threading.Event()
all_text = ""

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

def recognize_speech(timeout=7):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for commands...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
        except sr.WaitTimeoutError:
            print("Listening timed out, speak Computer again to activate")
            return None

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Command Identified: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def listen_for_commands():
    global all_text
    porcupine = pvporcupine.create(access_key=ACCESS_KEY, keywords=["computer"])
    pa = pyaudio.PyAudio()

    default_device_index = pa.get_default_input_device_info()['index']

    def create_audio_stream():
        return pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
            input_device_index=default_device_index
        )

    stream = None
    try:
        print("Listening for wake word...")

        while not exit_flag.is_set():
            if stream is None or stream.is_stopped():
                stream = create_audio_stream()

            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = np.frombuffer(pcm, dtype=np.int16)

            if porcupine.process(pcm) >= 0:
                print("Wake word detected!!!")
                stream.stop_stream()
                stream.close()
                stream = None

                command = recognize_speech()
                if command:
                    if "capture image" in command:
                        cv2.imwrite('captured_image.png', rgb_image)
                        rint("Capture Image Command Executed")
                    elif "Identify Text" in command:
                        all_text = read_text_from_image('captured_image.png')
                        print("Text Identified: ")
                        print(all_text)
                        print("Identify Text Command Executed")
                    elif "text to speech" in command:
                        if all_text:
                            speak_text(all_text)
                            print("Text to Speech Command Executed")
                        else:
                            print("No text available to read. Please perform OCR operation again.")
                    elif "stop" in command or "listen" in command:
                        print("Stop command detected, exiting program...")
                        exit_flag.set()
                        break
    finally:
        if stream and not stream.is_stopped():
            stream.stop_stream()
        if stream:
            stream.close()
        pa.terminate()
        porcupine.delete()

def main():
    global rgb_image
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

    rgb_window = "Streaming"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    monitor = get_monitors()[0]  
    screen_width, screen_height = monitor.width, monitor.height

    def signal_handler(sig, frame):
        print("Stop listening to image data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        cv2.destroyAllWindows()
        exit_flag.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    command_thread = threading.Thread(target=listen_for_commands)
    command_thread.start()

    try:
        while not exit_flag.is_set():
            if aria.CameraId.Rgb in observer.images:
                rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

                # Resize image to screen resolution
                rgb_image = cv2.resize(rgb_image, (screen_width, screen_height))

                cv2.imshow(rgb_window, rgb_image)
                del observer.images[aria.CameraId.Rgb]

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
