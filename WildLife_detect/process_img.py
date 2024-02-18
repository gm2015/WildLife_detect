from PIL import Image
from ultralytics import YOLO
from utils.constant import *
import os

confidence = 0.5
IMG_NAME = 'test.jpg'
file_path_name = os.path.join(TEST_DATA_FOLDER,IMG_NAME).replace('\\', '/')
out_path_name = os.path.join(OUPUT_FOLDER,f'{IMG_NAME[:-4]}_{int(100*confidence)}.jpg').replace('\\', '/')

# Load a model
model = YOLO(YOLO_MODEL_PATH)  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(file_path_name, conf = confidence)  # return a list of Results objects

# Process results list
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save(out_path_name)  # save image
    print(r.boxes.conf)
    print(r.boxes.cls)