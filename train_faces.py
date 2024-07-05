import os
import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Warning: Image at path {path} could not be loaded.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

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

training_folder = 'Train'
faces = create_database(training_folder)

with open('trained_faces.pkl', 'wb') as f:
    pickle.dump(faces, f)
