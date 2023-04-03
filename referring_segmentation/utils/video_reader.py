import os
import h5py
import numpy as np


def clip_annotation_reader(images_path, annotations_path, instances, clip_size=7, annotation_center=False, dataset='A2D'):

    datas = []
    frames = os.listdir(images_path)
    frames.sort()
    annotations = os.listdir(annotations_path)
    annotations.sort()
    for annotation in annotations:
        name = annotation.split('.')[0]
        name_int = int(name)
        with h5py.File(os.path.join(annotations_path, annotation)) as label:
            instances_anno = list(label['instance'][:])
        for instance in instances.keys():
            # if instance < int(instance_num):
            if int(instance) in instances_anno:
                step = 1
                if not annotation_center:
                    range_frames = step * np.random.randint(- (clip_size - 1), 1)
                else:
                    range_frames = - step * (clip_size // 2)

                initial_frame = name_int + range_frames
                data = {}
                data['dataset'] = dataset
                data['video'] = images_path.split('/')[-1]
                data['frames'] = []
                data['label'] = []
                data['instance'] = instance
                data['sentence'] = instances[instance]
                annotation_num = 0
                for i in range(0, clip_size * step, step):
                    n_frame = initial_frame + i - 1
                    if n_frame < 0:
                        n_frame = 0
                    elif n_frame >= len(frames):
                        n_frame = len(frames) - 1
                    data['frames'].append(os.path.join(images_path, frames[n_frame]))

                    is_anno = 0
                    for anno in annotations:
                        if frames[n_frame].split('.')[0] == anno.split('.')[0]:
                            data['label'].append(os.path.join(annotations_path, anno))
                            annotation_num += 1
                            is_anno += 1

                    if is_anno == 0:
                        data['label'].append('None')
                if annotation_num > 0:
                    datas.append(data)
    return datas


def sequence_reader(images_path, annotations_path, instances, dataset='A2D'):

    datas = []
    frames = [f for f in os.listdir(images_path) if '.png' in f]
    frames.sort()
    annotations = os.listdir(annotations_path)
    annotations.sort()
    for instance in instances.keys():
        data = {}
        data['dataset'] = dataset
        data['video'] = images_path.split('/')[-1]
        data['frames'] = []
        data['label'] = []
        data['sentence'] = instances[instance]
        data['instance'] = instance
        for frame in frames:
            data['frames'].append(os.path.join(images_path, frame))
            name = frame.split('.')[0]
            is_annotated = 0
            if dataset == 'A2D':
                for annotation in annotations:
                    if annotation.split('.')[0] == name:
                        is_annotated += 1
                        with h5py.File(os.path.join(annotations_path, annotation)) as label:
                            instances_anno = list(label['instance'][:])
                        if int(instance) in instances_anno:
                            data['label'].append(os.path.join(annotations_path, annotation))
                        else:
                            data['label'].append('None')
                if is_annotated == 0:
                    data['label'].append('None')
            elif dataset == 'JHMDB':
                data['label'].append(os.path.join(annotations_path, annotations[0]))

        datas.append(data)
    return datas









