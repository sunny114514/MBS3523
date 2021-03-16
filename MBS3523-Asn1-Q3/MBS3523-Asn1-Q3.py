import cv2
print(cv2.__version__)
import random

capture = cv2.VideoCapture('Resources/MBS3523-Asn1-Q3video.mp4')
pedestrians_cascade1 = cv2.CascadeClassifier('Resources/haarcascade_fullbody.xml')
cars_cascade2 = cv2.CascadeClassifier('Resources/cars.xml')
while True:
    success, img = capture.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    human = pedestrians_cascade1.detectMultiScale(imgGray, 1.1, 3)
    cars =  cars_cascade2.detectMultiScale(imgGray, 1.1, 3)
    for (x, y, w, h) in human:
        color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), (color1), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in cars:
        color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), (color2), 2)
    cv2.imshow('Frame', img)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()