import cv2
from ultralytics import YOLO
from detect_frame import *

def quick_detect_vid(video_path, isShow, break_early):

    HEIGH = 540
    WIDTH = 960
    config = DetectionConfig()    
    model = YOLO('yolov8n.pt', config.confidence)    
    cap = cv2.VideoCapture(video_path)

    save_info = DetectionInfo()
    info = DetectionInfo()
    save_frame = []
    while cap.isOpened():
        # Read a frame from the video
        success, original_frame = cap.read()        
        if success:            
            #height, width, layers = original_frame.shape
            frame = cv2.resize(original_frame, (WIDTH, HEIGH))        
            annotated_frame, info = detect_frame(frame, model, config, info)
            if not np.any(save_frame): #only first time
                save_info.confidence = info.confidence
                save_info.has_animal = info.has_animal
                save_frame = annotated_frame   
            if info.has_animal and info.confidence > save_info.confidence:           
                save_info.confidence = info.confidence
                save_info.has_animal = info.has_animal
                save_frame = annotated_frame
                
            if info.has_animal and info.confidence > config.high_conf and break_early:                        
                break
                
            if isShow:
                cv2.imshow("Animal Detection", annotated_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
        # end of video    
        else:            
            break   
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    
    
    save_frame = add_info_to_frame(save_frame,save_info)
    if isShow:
        cv2.imshow("Result Detection", save_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return save_frame, save_info