import os
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Warning: Image at path {path} could not be loaded.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image(image):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_image(img_path)
        if img is not None:
            images.append(img)
    return images

class Face:
    def __init__(self, name, feature_vector):
        self.name = name
        self.feature_vector = feature_vector

def create_database(training_folder):    
    faces_dict = defaultdict(list)
    
    person_folders = [os.path.join(training_folder, person_folder) 
                      for person_folder in os.listdir(training_folder) 
                      if os.path.isdir(os.path.join(training_folder, person_folder))]
    
    for person_folder in tqdm(person_folders, total=len(person_folders)):
        person_name = os.path.basename(person_folder)
        images = load_images_from_folder(person_folder)
        
        for image in images:
            locs = face_recognition.face_locations(image, model='hog')
            if locs:
                loc = locs[0]
                vec = face_recognition.face_encodings(image, [loc], num_jitters=20)[0]
                faces_dict[person_name].append(vec)
    
    faces = []
    for name, vecs in faces_dict.items():
        avg_vec = np.mean(vecs, axis=0)  
        face = Face(name=name, feature_vector=avg_vec)
        faces.append(face)
    
    return faces

def detect_faces(image_test, faces, threshold=0.6):
    locs_test = face_recognition.face_locations(image_test, model='hog')
    vecs_test = face_recognition.face_encodings(image_test, locs_test, num_jitters=1)
    
    for loc_test, vec_test in zip(locs_test, vecs_test):
        distances = [face_recognition.face_distance([face.feature_vector], vec_test)[0] for face in faces]
        
        if np.min(distances) > threshold:
            pred_name = 'unknown'
        else:
            pred_index = np.argmin(distances)
            pred_name = faces[pred_index].name
        
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

training_folder = 'Train'
faces = create_database(training_folder)

show_image(detect_faces(load_image('testing/test1.jpg'), faces))
show_image(detect_faces(load_image('testing/test2.jpg'), faces))
show_image(detect_faces(load_image('testing/test3.jpg'), faces))
