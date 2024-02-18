from ultralytics import YOLO
import supervision as sv
import cv2

#PATH_TO_IMAGE = 'deer.jpg'
PATH_TO_IMAGE = 'cat_dog.jpg'
CONFIDENCE = 0.1

selected_classes_id = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77]
model = YOLO('yolov8n.pt', CONFIDENCE)
image = cv2.imread(PATH_TO_IMAGE)
result = model.predict(source=PATH_TO_IMAGE)[0]
detections = sv.Detections.from_ultralytics(result)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = list()
print(detections.class_id)
for item in detections.class_id:
    if item in selected_classes_id:
        labels.append('animal')
    else:
        labels.append('other')

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(annotated_image)
