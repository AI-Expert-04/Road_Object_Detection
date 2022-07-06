import cv2
import os

file_name = 'ae7f4103-b1fbdc93.jpg'
image = cv2.imread('../data/images/' + file_name)

resize_image = cv2.resize(image, (100, 200))

if not os.path.exists('abcd'):
    os.mkdir('abcd')

cv2.imwrite('abcd/1.jpg', resize_image)
