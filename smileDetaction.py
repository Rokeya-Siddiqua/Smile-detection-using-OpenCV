import cv2

#face detactor
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#smile detactor
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')


######### smile detect for a video start ###########
webcam = cv2.VideoCapture(0)  #video_name.mp4

while True:
    sucessful_frame_read, frame = webcam.read()
    #abort if there is an error
    if not sucessful_frame_read:
        break
    # frame in gray scale
    gray_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    #detect face and smile
    faces = face_detector.detectMultiScale(gray_scaled_frame)
    # run face detection within each of these faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),5)
        #create the face sub-image
        face = frame[y:y+h, x:x+w]
        #gray scale the face
        face_gray_scale = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)

        smiles = smile_detector.detectMultiScale(gray_scaled_frame, scaleFactor = 1.7, minNeighbors = 20)
        
        #for (x_smile,y_smile,w_smile,h_smile) in smiles:
            #draw a rectangle around the simle
        #    cv2.rectangle(face, (x_smile,y_smile),(x_smile+w_smile,y_smile+h_smile),(50,55,200),5)

        #level the face as smiling
        if (len(smiles>0)):
            cv2.putText(frame,'smiling', (x, y+h+40), fontScale=3, fontFace = cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
        
    #display
    cv2.imshow('why so serious', frame)
    key = cv2.waitKey(1)

    #stop if Q/q key is pressed
    if key==81 or key==113:
        break

# clean up
webcam.release()
cv2.destroyAllWindows()  

    

        
 
######### smile detect for a video end ###########
print("code completed")