# import cv2
#
# image = cv2.imread('images/cat_face/cat_5_mask.png')
# image = cv2.resize(image, (224, 224))
# cv2.imwrite('curl_test/cat_5_mask.png', image)
import re
name = 'images_00020.png'
itemset = re.split('images_|.png', name)
items = name.replace('images_','')
items = items.replace('.png','')
print(itemset)
print(items)