import cv2
import os

path = '../data/videos/cabc30fc-e7726578.mov'
cap = cv2.VideoCapture(path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

if not os.path.exists('../outputs'):
    os.mkdir('../outputs')

out = cv2.VideoWriter('../outputs/output.wmv', fourcc, fps, (width, height))

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    cv2.imshow('video', image)
    out.write(image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
