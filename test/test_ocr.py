from common import *
import time

image = cv2.imread('./data/ocr3.jpg')

start = time.process_time()
out = ocr_get_texts(image)
end = time.process_time()
print((end - start) * 1000)

print(out)
cv2.imshow("original", image)
cv2.waitKey()
cv2.destroyAllWindows()