import numpy as np


def augmentation(feat, timestamp, duration, augm=True):
    frame_num = feat.shape[0]
    start_frame = int(1.0 * frame_num * timestamp[0] / duration)
    end_frame = int(1.0 * frame_num * timestamp[1] / duration)
    if end_frame >= frame_num:
        end_frame = frame_num - 1
    if start_frame > end_frame:
        start_frame = end_frame
    label = np.asarray([start_frame, end_frame]).astype(np.int32)
    if augm:
        if start_frame // 2 != 0:
            left = np.random.randint(0, start_frame//2)
        else:
            left = 0
        if end_frame != frame_num-1:
            right = np.random.randint(
                end_frame + (frame_num - 1 - end_frame) // 2, frame_num-1)
        else:
            right = frame_num-1
        start_frame = start_frame - left
        end_frame = end_frame - left
        label = np.asarray([start_frame, end_frame]).astype(np.int32)
        feat = feat[left: right, :]
        left_time = 1.0 * duration * left / frame_num
        timestamp = [timestamp[0] - left_time, timestamp[1] - left_time]
    return feat, timestamp, label


if __name__ == '__main__':
    feat = np.ones((87, 1024))
    timestamp = [13.1, 21.0]
    duration = 25
    feat, timestamp, label = augmentation(feat, timestamp, duration)
    print(feat.shape)
    print(timestamp)
    print(label)