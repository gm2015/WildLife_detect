import supervision as sv
import cv2
import os
from datetime import datetime
import pytesseract

OCR_PATH = r'C:\Users\TD\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = OCR_PATH

# output_img 980x540, text at bottom: r1:r2, c1:c2

TEXT_ROI = [518,540,250,780] 
def ocr_get_text_roi(image):
    r1 = TEXT_ROI[0]
    r2 = TEXT_ROI[1]
    c1 = TEXT_ROI[2]
    c2 = TEXT_ROI[3]
    img = image[r1:r2, c1:c2]
    return img


def ocr_get_texts(image):
    img = ocr_get_text_roi(image)    
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    txts = pytesseract.image_to_string(img, config=custom_config).split()
    if len(txts) < 5:
        return None
    else:
        temp = txts[0]
        date = txts[2]
        time = txts[3]
        cam = txts[4]
        out_str = f'temperature= {temp}, date= {date}, time= {time}, camera= {cam}'
        return temp, date, time, cam, out_str
    

class DetectionConfig():
    def __init__(self) -> None:
        self.confidence = 0.1
        self.verbose = False
        self.bbcolor = sv.Color.BLUE
        self.lcolor = sv.Color.BLUE
        self.selected_ids = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77]
        self.high_conf = 0.6

class DetectionInfo():
    def __init__(self) -> None:
        self.has_animal = False
        self.labels = ''
        self.confidence = 0.00
        self.box = [0.0,0.0,0.0,0.0]
        self.infile = ''
        self.loc = ''
        self.cam = ''
        self.date = ''
        self.fname = ''
        self.id = 0
    def update_filename(self, filename, order): 
        fn = filename.split('/')
        N = len(fn)
        if N<6:
            print('Data Folder invalid')
            return
        self.loc = fn[N-5]
        self.cam = fn[N-4]
        date = fn[N-3]
        self.date = date[:-6]        
        self.fname = fn[N-1]
        self.infile = filename
        self.id = order


def print_list(a_list):
    for item in a_list:
        print(item)

def get_created_date(file_path):
  creation_time = os.path.getctime(file_path)
  dt = datetime.fromtimestamp(creation_time)
  return dt

def to_date_only(t):
    return t.strftime("%Y-%m-%d")
    
def to_time_only(t):
    return t.strftime("%I:%M %p")

def to_datetime(t):
    return t.strftime("%Y-%m-%d %I:%M %p")

def add_info_to_frame(frame,info):
    frame = cv2.putText(
        frame,
        f'Have animal:  {info.has_animal}. Confidence: {info.confidence}',
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA
        )        
    return frame

def get_all_vid_files(data_folder, pattern):
    vid_files = list()    
    for path, _, files in os.walk(data_folder):
        vid_files.extend([os.path.join(path, file).replace('\\', '/') for file in files
            if str(file).endswith(pattern) and not str(file).startswith('._')])
    return vid_files

def make_imgs_name_path(out_folder, vid_files):
    out_imgs = list()
    for id, vid in enumerate(vid_files):
        fn = vid.split('/')
        N = len(fn)
        loc = fn[N-5]
        cam = fn[N-4]
        date = fn[N-3]
        date = date[:-6]        
        fname = fn[N-1]
        fname = fname[:-3]                
        out_img = f'{id}_{loc}_{cam}_{date}_{fname}jpg'
        out_imgs.append(os.path.join(out_folder,out_img).replace('\\', '/'))
    return out_imgs

def save_info_csv(file_handle, i):
    if i is None:
        line = 'order, location, camera, date, filename, full_path, has animal, confidence, \n'
    else:
        line = f'{i.id}, {i.loc}, {i.cam}, {i.date}, {i.fname}, {i.infile}, {i.has_animal}, {i.confidence}\n'
    
    file_handle.write(line)