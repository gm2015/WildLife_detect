from detect_vid import *
import cv2
import os

#VIDNAME = "IMG_0123.mp4"
#VIDNAME = "IMG_0023.mp4"

#VIDNAME = "IMG_0035.mp4"
VIDNAME = "IMG_0024.mp4"
#VIDNAME = "IMG_0027.mp4"
#VIDNAME = "IMG_0255.mp4"
#VIDNAME = "IMG_0007.mp4"

data_folder = "./data"
out_folder = './output'
file_path_name = os.path.join(data_folder,VIDNAME).replace('\\', '/')
out_path_name = os.path.join(out_folder,f'{VIDNAME}.jpg').replace('\\', '/')

isShow = True
save_frame, save_info = quick_detect_vid(
    video_path=file_path_name, 
    isShow=True,
    break_early=False)

cv2.imwrite(out_path_name,save_frame)
out_file_name = os.path.join(out_folder,'output.txt')
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

if not isShow and np.any(save_frame):
    cv2.imshow("Result Detection", save_frame)
    cv2.waitKey()
    cv2.destroyAllWindows()