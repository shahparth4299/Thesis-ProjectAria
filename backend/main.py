from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import sys
import asyncio
import threading
import math
import logging
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
from easy_ocr_read import read_text_from_image
from queue import Queue, Empty
from ultralytics import YOLO
import pickle
import face_recognition
from fastapi.middleware.cors import CORSMiddleware 
import firebase_admin
from firebase_admin import credentials, storage
import uuid
import tempfile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'visualassistant-f410b.appspot.com'
})

origins = [
    "http://localhost:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True, 
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def upload_image_to_firebase(image_path):
    bucket = storage.bucket()
    file_name = f"captured_images/{uuid.uuid4()}.png"  
    blob = bucket.blob(file_name)
    blob.upload_from_filename(image_path)

    blob.make_public()

    return blob.public_url


def upload_face_image_to_firebase(image_data, filename):
    
    bucket = storage.bucket()
    file_name = f"captured_images/{filename}"  
    blob = bucket.blob(file_name)
    blob.upload_from_string(image_data, content_type='image/jpeg')
    blob.make_public()
    return blob.public_url


class Face:
    def __init__(self, name, feature_vector):
        self.name = name
        self.feature_vector = feature_vector

class StreamingClientObserver:
    def __init__(self):
        self.images = {}
        self.identifications = set()
        self.object_queue = Queue()

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

observer = StreamingClientObserver()
streaming_client = None
streaming_manager = None
device_client = None
streaming_state = "stopped"
device = None
known_names = set()

class FaceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Face':
            return Face
        return super().find_class(module, name)

with open('trained_faces.pkl', 'rb') as f:
    faces = FaceUnpickler(f).load()

def start_streaming(streaming_interface="usb", update_iptables=False, profile_name="profile18", device_ip=None):
    global streaming_client, streaming_manager, device_client, observer, streaming_state, device
    
    try:
        if update_iptables and sys.platform.startswith("linux"):
            update_iptables()

        aria.set_log_level(aria.Level.Info)

        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if device_ip:
            client_config.ip_v4_address = device_ip
        device_client.set_client_config(client_config)
        device = device_client.connect()

        streaming_manager = device.streaming_manager
        streaming_client = streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = profile_name
        if streaming_interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = True
        streaming_manager.streaming_config = streaming_config

        streaming_manager.start_streaming()
        streaming_state = "started"
        logger.info(f"Streaming state: {streaming_manager.streaming_state}")

        config = streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        config.message_queue_size[aria.StreamingDataType.Rgb] = 1
        config.security_options.use_ephemeral_certs = True
        streaming_client.subscription_config = config

        streaming_client.set_streaming_client_observer(observer)
        streaming_client.subscribe()
        logger.info("Streaming started....")
    
    except Exception as e:
        logger.error(f"Failed to start streaming: {e}", exc_info=True)
        raise

def stop_streaming():
    global streaming_client, streaming_manager, device_client, streaming_state, device

    try:
        if streaming_client:
            streaming_client.unsubscribe()
        if streaming_manager:
            streaming_manager.stop_streaming()
        if device_client:
            device_client.disconnect(device)

        streaming_state = "stopped"
        logger.info("Streaming stopped....")
    
    except Exception as e:
        logger.error(f"Failed to stop streaming: {e}", exc_info=True)
        raise

def detect_objects(stop_event):
    global observer
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
                  "teddy bear", "hair drier", "toothbrush"]

    while not stop_event.is_set():
        if aria.CameraId.Rgb in observer.images:
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = model(rgb_image, stream=True)
            identified_classes = set()
            logger.error ("Image Taken")

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    if class_name not in observer.identifications:
                        observer.identifications.add(class_name)
                        identified_classes.add(class_name)

            if identified_classes:
                observer.object_queue.put(list(identified_classes))

            del observer.images[aria.CameraId.Rgb]

    logger.info(f"Identified objects: {list(observer.identifications)}")

def detect_faces(image_test, faces, known_names, threshold=0.6):
    locs_test = face_recognition.face_locations(image_test, model='hog')
    vecs_test = face_recognition.face_encodings(image_test, locs_test, num_jitters=1)
    
    identified_faces = []
    
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
            top, right, bottom, left = loc_test
            face_image = image_test[top:bottom, left:right]
            _, image_buffer = cv2.imencode('.jpg', face_image)
            image_data = image_buffer.tobytes()
            image_url = upload_face_image_to_firebase(image_data, f"{uuid.uuid4()}.jpg")
            identified_faces.append({"name": pred_name, "image_url": image_url})
        
        image_test = draw_bounding_box(image_test, loc_test)
        image_test = draw_name(image_test, loc_test, pred_name)
    
    return image_test, identified_faces

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

@app.get("/startStreaming")
async def start_streaming_endpoint():
    global streaming_state
    if streaming_state == "started":
        logger.warning("Streaming already started")
        return JSONResponse(content={"message": "Streaming already started"})
    
    try:
        start_streaming()
        logger.info("Streaming started via endpoint")
        return JSONResponse(content={"message": "Streaming started"})
    except Exception as e:
        logger.error(f"Error in starting streaming via endpoint: {e}", exc_info=True)
        return JSONResponse(content={"message": f"Error in starting streaming: {e}"}, status_code=500)

@app.get("/stopStreaming")
async def stop_streaming_endpoint():
    global streaming_state
    if streaming_state == "stopped":
        logger.warning("Streaming already stopped")
        return JSONResponse(content={"message": "Streaming already stopped"})
    
    try:
        stop_streaming()
        logger.info("Streaming stopped via endpoint")
        return JSONResponse(content={"message": "Streaming stopped"})
    except Exception as e:
        logger.error(f"Error in stopping streaming via endpoint: {e}", exc_info=True)
        return JSONResponse(content={"message": "Error in stopping streaming"}, status_code=500)

@app.get("/captureAndReadText")
async def capture_and_read_text():
    global observer

    if aria.CameraId.Rgb not in observer.images:
        logging.warning("No image available for capture")
        return JSONResponse(content={"message": "No image available for capture"}, status_code=400)

    image = observer.images.get(aria.CameraId.Rgb)
    captured_image_path = 'captured_image.png'
    cv2.imwrite(captured_image_path, image)
    logging.info("Image captured successfully")

    image_url = upload_image_to_firebase(captured_image_path)
    logging.info(f"Image uploaded to Firebase: {image_url}")

    try:
        text = read_text_from_image(captured_image_path)
        logging.info(f"Text identified: {text}")
        return JSONResponse(content={"identified_text": text, "image_url": image_url})
    except Exception as e:
        logging.error(f"Error in identifying text: {e}", exc_info=True)
        return JSONResponse(content={"message": f"Error in identifying text: {e}"}, status_code=500)

@app.get("/getIdentifiedObjects")
async def get_identified_objects():
    global observer
    identified_objects = list(observer.identifications)
    return JSONResponse(content={"identified_objects": identified_objects})

@app.get("/startObjectDetection")
async def start_object_detection():
    global streaming_state
    if streaming_state == "stopped":
        try:
            start_streaming()
        except Exception as e:
            logger.error(f"Error in starting streaming for object detection: {e}", exc_info=True)
            return JSONResponse(content={"message": f"Error in starting streaming: {e}"}, status_code=500)

    identified_objects = list()
    stop_event = threading.Event()
    detection_thread = threading.Thread(target=detect_objects, args=(stop_event,), daemon=True)
    detection_thread.start()

    await asyncio.sleep(7)
    stop_event.set()
    detection_thread.join()

    identified_objects = list(observer.identifications)
    logger.info(f"Object Detection completed. Identified objects: {identified_objects}")

    return JSONResponse(content={"Identified Objects": identified_objects})

@app.get("/identifyFaces")
async def identify_faces():
    global observer, known_names

    known_names = set()

    if aria.CameraId.Rgb not in observer.images:
        logger.warning("No RGB image available for face identification")
        return JSONResponse(content={"message": "No RGB image available for face identification"}, status_code=400)

    rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (640, 480))

    frame_with_faces, identified_faces = detect_faces(rgb_image, faces, known_names)
    
    logger.info(f"Identified faces: {identified_faces}")

    return JSONResponse(content={"identified_faces": identified_faces})

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
