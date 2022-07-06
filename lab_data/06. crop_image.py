import cv2

file_name = 'ae7f4103-b1fbdc93.jpg'
image = cv2.imread('../data/images/' + file_name)

crop_image = image[10:100, 10:200]

cv2.imshow('image', image)
cv2.imshow('crop image', crop_image)
cv2.waitKey(0)
