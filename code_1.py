import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

#0 - for camera
#capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('video_1.mp4')
i = 0
while True:
    flag,frame = capture.read()
    frame = cv2.resize(frame,None,fx=0.4,fy=0.4)
    if flag:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)
        #faces = [[x,y,w,h],[x,y,w,h]]
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,255),4)
            cv2.putText(frame, "face", (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('result',frame)
        i += 1
        cv2.imwrite('img_{}.png'.format(i),frame)
        if cv2.waitKey(5) == 27:
            break
    else:
        print("Camera not installed...or not working")

capture.release()
cv2.destroyAllWindows()
