import cv2

file_name = 'ae7f4103-b1fbdc93.jpg'
image = cv2.imread('../data/images/' + file_name)

cv2.rectangle(image, (10, 10), (100, 100), (0, 0, 255), 3)

cv2.imshow('image', image)
cv2.waitKey(0)
