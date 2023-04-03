import cv2
from tqdm import tqdm
import os

def video2imgs(videos_path, imgs_save_path):
    videos = os.listdir(videos_path)
    for video_name in tqdm(videos):
        file_name = video_name.split('.')[0]
        img_save_path = os.path.join(imgs_save_path, file_name)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        
        vc = cv2.VideoCapture(os.path.join(videos_path, video_name)) 
        i_frame = 0
        rval=vc.isOpened()

        while rval:  
            i_frame = i_frame + 1
            rval, frame = vc.read()
            if rval:
                cv2.imwrite(os.path.join(img_save_path, '{:05d}.jpg'.format(i_frame)), frame)
            else:
                break
        vc.release()

if __name__ =='__main__':
    videos_path = '/media/wwk/HDD1/dataset/referring_video_segmentation/a2d_sentences/Release/clips320H'
    imgs_save_path = '/media/wwk/HDD2/datasets/referring_video_segmentation/a2d_sentences/Rename_Images'
    video2imgs(videos_path, imgs_save_path)
