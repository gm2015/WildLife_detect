from utils.detect_vid import *
import cv2
import os

start_video_order = 603
end_video_order = 605

data_folder = "E:/WVC Raton Pass Project"
pattern   = ".MP4"
out_folder = './output'
isShow = False,
break_early = True
verbose = True

out_txt_name = os.path.join(out_folder,'output.txt')
vid_files = get_all_vid_files(data_folder, pattern)
if not len(vid_files):
    print('DATA folder NOT FOUND!!!')
out_imgs = make_imgs_name_path(out_folder, vid_files)

print(len(vid_files))
#print_list(vid_files)   
#print_list(out_imgs)
with open(out_txt_name,'a') as txt:
    save_info_csv(txt, None)    
    k = 0
    for vid_file, out_img in zip(vid_files,out_imgs):
        k = k + 1
        if k < start_video_order:
            continue
        if k > end_video_order:
            break

        time = get_created_date(vid_file)
        save_frame, save_info = quick_detect_vid(
            video_path=vid_file, 
            isShow=False,
            break_early=False,
            save_to_mp4=False)

        if verbose:
            print(f'File name: {vid_file}')
            #print(f'Date and time: {to_datetime(time)}')
            print(f'Have animal:  {save_info.has_animal}')
            print(f'Confidence: {save_info.confidence}')
            print('-----------------------')

        cv2.imwrite(out_img,save_frame)
        save_info.update_filename(vid_file, k)
        save_info_csv(txt, save_info)
        

