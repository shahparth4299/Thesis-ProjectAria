import cv2
import face_recognition
import numpy as np
import pickle
#from speech import speak_text

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
            #speak_text("I can see" + pred_name + "in front of you")
        
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

with open('trained_faces.pkl', 'rb') as f:
    faces = pickle.load(f)

def webcam_face_recognition(faces):
    video_capture = cv2.VideoCapture(0)
    known_names = set()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_with_faces = detect_faces(rgb_frame, faces, known_names)
        
        bgr_frame_with_faces = cv2.cvtColor(frame_with_faces, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Video', bgr_frame_with_faces)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

webcam_face_recognition(faces)
