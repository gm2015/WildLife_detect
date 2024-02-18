from common import *
from datetime import datetime


data_folder = "E:/WVC Raton Pass Project"
pattern   = ".MP4"

out_folder = './output'
out_txt_name = os.path.join(out_folder,'output_tmp.txt')
save_info = DetectionInfo()

vid_files = get_all_vid_files(data_folder, pattern)
if not len(vid_files):
    print('DATA folder NOT FOUND!!!')
print(len(vid_files))
print(type(vid_files))
out_imgs = make_imgs_name_path(out_folder, vid_files)

with open(out_txt_name,'w') as txt:
    save_info_csv(txt, None)    
    for i, vid in enumerate(vid_files):
        save_info.update_filename(vid,i)
                
        #print(to_datetime(t))
        #print('-------------')
        save_info_csv(txt, save_info)
        
        

