import cv2

file_name = 'ae7f4103-b1fbdc93.jpg'
image = cv2.imread('../data/images/' + file_name)

cv2.imshow('image', image)
cv2.waitKey(0)
