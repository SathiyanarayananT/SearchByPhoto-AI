import face_recognition as fr
import cv2
import os

myFace = fr.load_image_file("Dhoni_ref.jpg")
myFace = cv2.resize(myFace,(0,0) , fx=0.25, fy=0.25)
myFace_encoding = fr.face_encodings(myFace)

allPhotos = {}


folder = "C:\\Users\\i343453\\Desktop\\MyPhotos"
for filename in os.listdir(folder):
    photo = cv2.imread(os.path.join(folder,filename))
    if photo is not None:
        allPhotos[filename] = photo

photos = list(allPhotos.values())
for i in range(len(photos)):
    originalPhoto = photos[i]
    photo = cv2.resize(originalPhoto,(0,0) , fx=0.25, fy=0.25)

    rgbPhoto = photo[:,:,::-1]

    face_locations = fr.face_locations(rgbPhoto)
    face_encodings = fr.face_encodings(rgbPhoto, face_locations)
   for j in range(len(face_encodings)):
        face_encoding = face_encodings[j]
        matchesBool = fr.compare_faces(myFace_encoding , face_encoding)
        face_distances = fr.face_distance(myFace_encoding , face_encoding)
        if matchesBool and face_distances < 0.5:
            top = face_locations[j][0]
            right = face_locations[j][1]
            bottom = face_locations[j][2]
            left = face_locations[j][3]
            cv2.rectangle(photo, (left,top), (right, bottom), (0, 0, 255), 2)
            cv2.imshow("Visualize", photo)
            cv2.waitKey(0)
            print("Found : " , list(allPhotos)[i] , "--" , face_distances)
            break
    i += 1
