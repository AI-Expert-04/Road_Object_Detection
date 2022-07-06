import cv2

file_name = 'ae7f4103-b1fbdc93.jpg'
image = cv2.imread('../data/images/' + file_name)

cv2.putText(image, 'PYTHON', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.waitKey(0)
