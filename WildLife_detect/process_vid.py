from utils.detect_vid import *
from utils.constant import *
import cv2
import os

#VIDNAME = "IMG_0123.mp4"
#VIDNAME = "IMG_0023.mp4"

#VIDNAME = "IMG_0035.mp4"
VIDNAME = "IMG_0024.mp4"
#VIDNAME = "IMG_0027.mp4"
#VIDNAME = "IMG_0255.mp4"
#VIDNAME = "IMG_0007.mp4"

SAVE_TO_MP4 = True
IS_SHOW = True
BREAK_EARLY = False
file_path_name = os.path.join(TEST_DATA_FOLDER,VIDNAME).replace('\\', '/')
out_path_name = os.path.join(OUPUT_FOLDER,f'{VIDNAME}.jpg').replace('\\', '/')

save_frame, save_info = quick_detect_vid(
    video_path=file_path_name, 
    isShow=IS_SHOW,
    break_early=BREAK_EARLY,
    save_to_mp4=SAVE_TO_MP4)

cv2.imwrite(out_path_name,save_frame)
out_file_name = os.path.join(OUPUT_FOLDER,'output.txt')
with open(out_file_name,'w') as f:
    k=1
    save_info.update_filename(file_path_name,k)
    save_info_csv(f,save_info)

print(f'File name: {file_path_name}')
#print(f'Date and time: {to_datetime(time)}')
print(f'Have animal:  {save_info.has_animal}')
print(f'Confidence: {save_info.confidence}')
print(f'Bbox: {save_info.box}')
print('-----------------------')

if not IS_SHOW and np.any(save_frame):
    cv2.imshow("Result Detection", save_frame)
    cv2.waitKey()
    cv2.destroyAllWindows()