import cv2
from ultralytics import YOLO
from .constant import *
from .detect_frame import *

def quick_detect_vid(video_path, isShow, break_early, save_to_mp4):

    config = DetectionConfig()    
    model = YOLO(YOLO_MODEL_PATH, config.confidence)    
    cap = cv2.VideoCapture(video_path)
    
    fname = video_path.split('/')[-1]
    out_path_name = os.path.join(OUPUT_FOLDER,f'detected_{fname}').replace('\\', '/')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path_name, fourcc, 20.0, (config.WIDTH,config.HEIGH))

    save_info = DetectionInfo()
    info = DetectionInfo()
    save_frame = []
    while cap.isOpened():
        # Read a frame from the video
        success, original_frame = cap.read()        
        if success:            
            #height, width, layers = original_frame.shape
            frame = cv2.resize(original_frame, (config.WIDTH, config.HEIGH))        
            annotated_frame, info = detect_frame(frame, model, config, info)
            if save_to_mp4:
                out.write(annotated_frame)
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
    out.release()
    cv2.destroyAllWindows()
    
    
    save_frame = add_info_to_frame(save_frame,save_info)
    if isShow:
        cv2.imshow("Result Detection", save_frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return save_frame, save_info