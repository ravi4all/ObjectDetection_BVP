import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

#0 - for camera
#capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('video_1.mp4')
count = 0
while True:
    flag,frame = capture.read()
    frame = cv2.resize(frame,None,fx=0.4,fy=0.4)
    if flag:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)
        faces_len = len(faces)
        for i in range(faces_len):
            cv2.rectangle(frame,(faces[i][0],faces[i][1]),
                          (faces[i][0]+faces[i][2],faces[i][1]+faces[i][3]),
                          (0,255,255),4)
            cv2.putText(frame, str(i), (faces[i][0],faces[i][1]), font, 1, (255, 255, 0), 2)

            face = frame[faces[i][1]:faces[i][1]+faces[i][3],faces[i][0]:faces[i][0]+faces[i][2],:]
            count += 1
            cv2.imwrite('face_{}.png'.format(count),face)
            
        cv2.imshow('result',frame)
        if cv2.waitKey(5) == 27:
            break
    else:
        print("Camera not installed...or not working")

capture.release()
cv2.destroyAllWindows()
