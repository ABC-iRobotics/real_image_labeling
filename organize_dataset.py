from __future__ import annotations
import os
import sys
import cv2
import numpy as np

def organize_data(annotation_path):
    IMAGE_FOLDER = 'images'
    DATASET_FOLDER = 'dataset'
    TRAIN_VAL_SPLIT = 0.8   # 80% of dataset is train data

    train_folder = os.path.join(annotation_path, '..', DATASET_FOLDER, 'datasets', 'train')
    val_folder = os.path.join(annotation_path, '..', DATASET_FOLDER, 'datasets', 'val')
    segmentation_annotation_folder = os.path.join(annotation_path, '..', DATASET_FOLDER, 'Segmentation_annotations')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(segmentation_annotation_folder, exist_ok=True)

    if os.path.exists(annotation_path) and os.path.exists(os.path.join(annotation_path, '..', IMAGE_FOLDER)) and os.path.exists(os.path.join(annotation_path, '..', DATASET_FOLDER)):
        annotation_files = [p for p in os.listdir(annotation_path) if os.path.isfile(os.path.join(annotation_path, p)) and p.split('.')[-1] == 'png']
        train_indices = np.random.choice(len(annotation_files), int(len(annotation_files)*TRAIN_VAL_SPLIT), replace=False)

        for i, annotation_file in enumerate(annotation_files):
            img = cv2.imread(os.path.join(annotation_path, '..', IMAGE_FOLDER, annotation_file.replace('_annotation', '')), cv2.IMREAD_ANYCOLOR)
            mask = cv2.imread(os.path.join(annotation_path, annotation_file), cv2.IMREAD_ANYCOLOR)

            overlay = img + np.array(mask)*0.4

            cv2.imwrite(os.path.join(annotation_path, '..', DATASET_FOLDER, annotation_file.replace('_annotation', '_overlay')), overlay)
            
            cv2.imwrite(os.path.join(segmentation_annotation_folder, annotation_file), mask)

            if i in train_indices:
                cv2.imwrite(os.path.join(train_folder, annotation_file.replace('_annotation', '')), img)
            else:
                cv2.imwrite(os.path.join(val_folder, annotation_file.replace('_annotation', '')), img)
    else:
        raise FileNotFoundError('Could not find image folder ' + IMAGE_FOLDER + ' at ' + os.path.join(annotation_path, '..', IMAGE_FOLDER))

if __name__=='__main__':
    organize_data(os.path.join('projects', 'logos_screws', 'annotations'))