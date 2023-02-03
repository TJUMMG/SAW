import os
import csv
import json


def generate_data_list(datasets=['A2D','JHMDB'], root='/media/HardDisk/wwk/video_text/datasets/', save_root='./data/data_list'):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for dataset in datasets:
        dataset_root = os.path.join(root, dataset)
        assert os.path.exists(dataset_root), ('Incorrect dataset path: {}'.format(dataset_root))
        if dataset == 'A2D':
            num_videos = 0
            num_annotated_videos = 0
            videos = os.listdir(os.path.join(dataset_root, 'Rename_Images'))
            ignore_file = open(os.path.join(dataset_root, 'a2d_missed_videos.txt'), 'r')
            ignore_data_list = ignore_file.read().splitlines()
            videoset_file = open(os.path.join(dataset_root, 'videoset.csv'), 'r')
            videosets = list(csv.reader(videoset_file))
            instances = generate_annotation_zip(dataset_root)

            json_train = {}
            json_test = {}
            for videoset in videosets:
                video_name = videoset[0]
                assert  video_name in videos, ('Incorrect video name: {} in csv file: {}'.format(video_name, os.path.join(dataset_root, 'videoset.csv')))
                if video_name not in ignore_data_list:

                    num_videos += 1
                    frames_root = os.path.join(dataset_root, 'Rename_Images', video_name)
                    annotations_root = os.path.join(dataset_root, 'a2d_annotation_with_instances', video_name)
                    if os.path.exists(annotations_root):
                        num_annotated_videos += 1
                        if videoset[-1] == '0':
                            json_train[video_name] = {'frames': frames_root, 'labels': annotations_root, 'instances': instances[video_name]}
                        else:
                            json_test[video_name] = {'frames': frames_root, 'labels': annotations_root,
                                                           'instances': instances[video_name]}
                    else:
                        print('Annotation of video {} in A2D dataset not exits'.format(video_name))

            with open(os.path.join(save_root, 'data_{}_train.json'.format(dataset.lower())), 'w+') as json_train_file:
                json.dump(json_train, json_train_file, indent=1)
            with open(os.path.join(save_root, 'data_{}_test.json'.format(dataset.lower())), 'w+') as json_test_file:
                json.dump(json_test, json_test_file, indent=1)
            print('A2D dataset : Total videos : {} | Annotated videos : {}'.format(num_videos, num_annotated_videos))

        else:
            num_videos = 0
            num_annotated_videos = 0
            video_groups = [f for f in os.listdir(os.path.join(root, dataset, 'Rename_Images')) if '.' not in f]
            json_test = {}
            for video_group in video_groups:
                videos_root = os.path.join(root, dataset, 'Rename_Images', video_group)
                videos = [f for f in os.listdir(videos_root) if '.' not in f]
                for video in videos:
                    video_root = os.path.join(videos_root, video)
                    annotation_root = os.path.join(root, dataset, 'puppet_mask', video_group, video)
                    num_videos += 1
                    if os.path.exists(annotation_root):
                        json_test[video] = {'frames': video_root, 'labels': annotation_root,
                                                 'instances': ['0']}
                        num_annotated_videos += 1
                    else:
                        print('Annotation of video {}/{} in JHMDB dataset not exits'.format(video_group, video))
            with open(os.path.join(save_root, 'data_{}_test.json'.format(dataset.lower())), 'w+') as json_test_file:
                json.dump(json_test, json_test_file, indent=1)
            print('JHMDB dataset : Total videos : {} | Annotated videos : {}'.format(num_videos, num_annotated_videos))


def generate_annotation_zip(root):
    annotation_file = open(os.path.join(root, 'a2d_annotation.txt'))
    annotation_list = list(annotation_file.read().splitlines())
    annotations = {}
    for i in range(1, len(annotation_list)):
        name, instance, desc = annotation_list[i].split(',')
        if name not in annotations.keys():
            annotations[name] = [int(instance)]
        else:
            annotations[name] .append(int(instance))
    return annotations
