import cv2

path = '../data/videos/cabc30fc-e7726578.mov'
cap = cv2.VideoCapture(path)
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    cv2.imshow('video', image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
