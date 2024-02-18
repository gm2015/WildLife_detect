from ultralytics import YOLO
import supervision as sv

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_IMAGE = 'deer.jpg'
#PATH_TO_IMAGE = 'cat_dog.jpg'
CONFIDENCE = 0.1
HIGH_CONF = 0.7
selected_classes_id = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77]
model = YOLO('yolov8n.pt', CONFIDENCE)
image = cv2.cvtColor(cv2.imread(PATH_TO_IMAGE), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
ax = plt.gca()

# Perform object detection on the image
result = model.predict(source=PATH_TO_IMAGE)[0]

for box in result.boxes:
  class_id = box.cls[0].item()
  if np.isin(class_id,selected_classes_id):
    class_name = 'animal'
  else:
    class_name = 'other'
    break
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(box.conf[0].item(), 2)
  
  if conf > HIGH_CONF:
    print("Object type:", class_name)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("---")
    color = 'blue'
  else:
    color = 'green'

  x0, y0 = cords[0], cords[1]
  w, h = cords[2] - cords[0], cords[3] - cords[1]
  ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))    
  plt.text(x0, y0, class_name, color=color, fontsize = 10)


plt.axis('off')
plt.show()
  
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 

"""

# Convert Bounding Box to Segmentation Mask using SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
image = cv2.cvtColor(cv2.imread(PATH_TO_IMAGE), cv2.COLOR_BGR2RGB)

cv2.imshow('Animal',image)



sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

input_box = np.array(bbox)
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=bbox[None, :],
    multimask_output=False,
)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.show()


# Background Removal
segmentation_mask = masks[0]

# Convert the segmentation mask to a binary mask
binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
white_background = np.ones_like(image) * 255

# Apply the binary mask
new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]

plt.imshow(new_image.astype(np.uint8))
plt.axis('off')
plt.show()
"""