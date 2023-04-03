import os
import csv
import json


def generate_annotation_dict(root):
    annotation_file = open(os.path.join(root))
    annotation_list = list(annotation_file.read().splitlines())
    annotations = {}
    for i in range(1, len(annotation_list)):
        if 'a2d' in root:
            name, instance, desc = annotation_list[i].split(',')
        else:
            name, desc = annotation_list[i].split(',')
            instance = '0'
        if name not in annotations.keys():
            annotations[name] = {}
        annotations[name][instance] = desc
    return annotations


def generate_data_list_a2d(dataset_root='/media/wwk/HDD2/datasets/referring_video_segmentation/a2d_sentences/', save_root='./data'):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    assert os.path.exists(
            dataset_root), ('Incorrect dataset path: {}'.format(dataset_root))
    num_videos = 0
    num_annotated_videos = 0
    videos = os.listdir(os.path.join(dataset_root, 'Rename_Images'))
    ignore_file = open(os.path.join(
        dataset_root, 'a2d_missed_videos.txt'), 'r')
    ignore_data_list = ignore_file.read().splitlines()
    videoset_file = open(os.path.join(
        dataset_root, 'videoset.csv'), 'r')
    videosets = list(csv.reader(videoset_file))
    instances = generate_annotation_dict(os.path.join(dataset_root, 'a2d_annotation.txt'))

    json_train = {}
    json_test = {}
    for videoset in videosets:
        video_name = videoset[0]
        assert video_name in videos, ('Incorrect video name: {} in csv file: {}'.format(
            video_name, os.path.join(dataset_root, 'videoset.csv')))
        if video_name not in ignore_data_list:
            num_videos += 1
            frames_root = os.path.join(
                dataset_root, 'Rename_Images', video_name)
            annotations_root = os.path.join(
                dataset_root, 'a2d_annotation_with_instances', video_name)
            if os.path.exists(annotations_root):
                num_annotated_videos += 1
                if videoset[-1] == '0':
                    json_train[video_name] = {
                        'frames': os.path.join('Rename_Images', video_name), 'labels': os.path.join('a2d_annotation_with_instances', video_name), 'instances': instances[video_name]}
                else:
                    json_test[video_name] = {
                        'frames': os.path.join('Rename_Images', video_name), 'labels': os.path.join('a2d_annotation_with_instances', video_name), 'instances': instances[video_name]}
            else:
                print(
                    'Annotation of video {} in A2D dataset not exits'.format(video_name))

    with open(os.path.join(save_root, 'a2d_sentences_train.json'), 'w+') as json_train_file:
        json.dump(json_train, json_train_file, indent=1)
    with open(os.path.join(save_root, 'a2d_sentences_test.json'), 'w+') as json_test_file:
        json.dump(json_test, json_test_file, indent=1)
    print('A2D dataset : Total videos : {} | Annotated videos : {}'.format(
        num_videos, num_annotated_videos))


def generate_data_list_jhmdb(dataset_root='/media/wwk/HDD1/dataset/referring_video_segmentation/jhmdb_sentences', save_root='./data'):
    assert os.path.exists(
            dataset_root), ('Incorrect dataset path: {}'.format(dataset_root))
    num_videos = 0
    num_annotated_videos = 0
    video_groups = [f for f in os.listdir(os.path.join(
        dataset_root, 'Rename_Images')) if '.' not in f]
    instances = generate_annotation_dict(os.path.join(dataset_root, 'jhmdb_annotation.txt'))
    json_test = {}
    for video_group in video_groups:
        videos_root = os.path.join(
            dataset_root, 'Rename_Images', video_group)
        videos = [f for f in os.listdir(videos_root) if '.' not in f]
        for video in videos:
            annotation_root = os.path.join(
                dataset_root, 'puppet_mask', video_group, video)
            num_videos += 1
            if os.path.exists(annotation_root):
                json_test[video] = {'frames': os.path.join('Rename_Images', video), 'labels': os.path.join('puppet_mask', video_group, video),
                                    'instances': instances[video]}
                num_annotated_videos += 1
            else:
                print(
                    'Annotation of video {}/{} in JHMDB dataset not exits'.format(video_group, video))
    with open(os.path.join(save_root, 'jhmdb_sentences_test.json'), 'w+') as json_test_file:
        json.dump(json_test, json_test_file, indent=1)
    print('JHMDB dataset : Total videos : {} | Annotated videos : {}'.format(
        num_videos, num_annotated_videos))


if __name__ == '__main__':
    save_root='./data'
    a2d_dataset_root=''
    jhmdb_dataset_root=''
    generate_data_list_a2d(a2d_dataset_root, save_root)
    generate_data_list_jhmdb(jhmdb_dataset_root, save_root)
