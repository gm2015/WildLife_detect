import numpy as np
#from ultralytics import YOLO
import supervision as sv
from common import *

def detect_frame(frame, model, config, info): 
    confidence = config.confidence
    verbose = config.verbose
    bbcolor = config.bbcolor
    lcolor = config.lcolor
    selected_ids = config.selected_ids
    high_conf = config.high_conf
    # Run YOLOv8 inference on the frame
    result = model.predict(frame, conf = confidence, verbose = verbose)[0]
    #if len(result) < 1:
    #    cv2.imshow("Animal Detection", frame)
    #    continue
        
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[np.isin(detections.class_id, selected_ids)]
    bounding_box_annotator = sv.BoundingBoxAnnotator(color=bbcolor)
    label_annotator = sv.LabelAnnotator(color=lcolor)

    labels = list()
    
    annotated_frame = frame
    #if len(detections.class_id) < 1:
    #    return annotated_frame, info
    for _ in detections.class_id:
        conf = round(result.boxes.conf[0].item(),2)
        #print(conf)
        labels.append(f'animal - {conf}')            
        if conf > high_conf:               
            info.has_animal = True
        else:
            info.has_animal = False
        info.labels = labels
        info.confidence = conf
        info.box = result.boxes.xyxy[0]          
   
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, 
        detections=detections
        )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections, 
        labels=labels
        )
    return annotated_frame, info