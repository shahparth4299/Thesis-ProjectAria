import os
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #1
    return image

def show_image(image):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return


filenames = os.listdir('training')
images = [load_image(f'training/{filename}') for filename in filenames]

show_image(images[0])

class Face:
    def __init__(self, bounding_box, cropped_face, name, feature_vector):
        self.bounding_box = bounding_box      
        self.cropped_face = cropped_face      
        self.name = name                      
        self.feature_vector = feature_vector  


def create_database(filenames, images):    
    faces = []
    for filename, image in tqdm(zip(filenames, images), total=len(filenames)):
        loc = face_recognition.face_locations(image, model='hog')[0]  
        vec = face_recognition.face_encodings(image, [loc], 
                                              num_jitters=20)[0]      
        
        top, right, bottom, left = loc    
        
        cropped_face = image[top:bottom, left:right]    
        
        face = Face(bounding_box=loc, cropped_face=cropped_face, 
                    name=filename.split('.')[0], feature_vector=vec)  

        faces.append(face)    

    return faces

faces = create_database(filenames, images)

show_image(faces[0].cropped_face)

print(faces[1].bounding_box)
print(faces[1].name)
print(faces[1].feature_vector)

def detect_faces(image_test, faces, threshold=0.6):    
    locs_test = face_recognition.face_locations(image_test, model='hog')  
    vecs_test = face_recognition.face_encodings(image_test, locs_test, 
                                                num_jitters=1)            
    
    for loc_test, vec_test in zip(locs_test, vecs_test):    
        distances = []
        for face in faces:
            distance = face_recognition.face_distance([vec_test], 
                                                      face.feature_vector)  
            distances.append(distance)
            
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
    
    cv2.rectangle(image_test, (left, top), (right, bottom), 
                  line_color, line_thickness)

    return image_test

def draw_name(image_test, loc_test, pred_name):
    print(pred_name)
    top, right, bottom, left = loc_test 
    
    font_style = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    font_thickness = 1
    
    text_size, _ = cv2.getTextSize(pred_name, font_style, font_scale, font_thickness)
    
    bg_top_left = (left, top-text_size[1])
    bg_bottom_right = (left+text_size[0], top)
    line_color = (0, 255, 0)
    line_thickness = -1

    cv2.rectangle(image_test, bg_top_left, bg_bottom_right, 
                  line_color, line_thickness)   

    cv2.putText(image_test, pred_name, (left, top), font_style, font_scale, font_color, font_thickness)
    
    return image_test

show_image(detect_faces(load_image('testing/test1.jpg'), faces))
