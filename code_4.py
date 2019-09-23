import cv2

dataset = cv2.CascadeClassifier('cars.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

capture = cv2.VideoCapture('video_3.wmv')
while True:
    flag,frame = capture.read()
    frame = cv2.resize(frame,None,fx=0.4,fy=0.4)
    if flag:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        car = dataset.detectMultiScale(gray)
        for x,y,w,h in car:
            cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,255),4)
        cv2.imshow('result',frame)
        if cv2.waitKey(5) == 27:
            break
    else:
        print("Camera not installed...or not working")

capture.release()
cv2.destroyAllWindows()
