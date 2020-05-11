import argparse
import sys
import os # To handle paths
import glob # Unix style pathname pattern expansion
import shutil
import math
import random
import datetime
import csv
import pandas as pd
import json
import yaml
import time
from tqdm import tqdm
from pathlib import Path

random.seed(42)

VALID_INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1536]

parser = argparse.ArgumentParser(description='Split dataset and regenerate annotations')
parser.add_argument('-d', '--dataset', required=True, type=str, help='id of the dataset you wish to parse')
parser.add_argument('-is', '--input_size', required=True, type=int, help="input size of the dataset's images (width == height)")
parser.add_argument('-s', '--split', type=int, nargs=3, default=[90,10,0], metavar=('train', 'val', 'test'), help='split datasets into train val test. Default: 90 10 0')
args = parser.parse_args()

DATASET_ID = args.dataset
SPLIT = args.split
INPUT_SIZE = args.input_size

if (INPUT_SIZE not in VALID_INPUT_SIZES) or (sum(SPLIT) != 100):
  print('Please verify that the input_size is correct, and that the split sums to 100%')
  sys.exit()

DATASET_HEIGHT = INPUT_SIZE
DATASET_WIDTH = INPUT_SIZE

DATA_FOLDER = f'../datasets/dataset-{DATASET_ID}'
OUTPUT_ANNOTATIONS_FOLDER = f'{DATA_FOLDER}/annotations'
INPUT_ANNOTATIONS = f'{DATA_FOLDER}/annotations.csv'
INPUT_ANNOTATIONS_TEST = f'{DATA_FOLDER}/annotations_test.csv'
INPUT_IMAGES_FOLDER = f'{DATA_FOLDER}/Images'
PROJECT_FILE = f'../projects/dataset-{DATASET_ID}.yml'

INFO = {
  'year': 2020,
  'version': DATASET_ID,
  'date_created': datetime.datetime.utcnow().isoformat(' ')
}

# Helpers
def move_files(filenames, source_folder, destination_folder):
  Path(destination_folder).mkdir(parents=True, exist_ok=True)
  for file in filenames:
    shutil.move(f"{source_folder}/{file}", f"{destination_folder}/{file}")

def coco_annotations_from_csv_to_json(input_annotations, categories, file):
  licences = [] # Leave as empty array

  # Process images
  images = []
  for index, filename in enumerate(list(input_annotations.filename.unique())):
    images.append({
      'id': index,
      'file_name': filename,
      'height': DATASET_HEIGHT, 
      'width': DATASET_WIDTH
    })
  
  # Process annotations
  annotations = []
  for index, input_annotation in tqdm(input_annotations.iterrows()):
    width = abs(input_annotation['x1'] - input_annotation['x2'])
    height = abs(input_annotation['y1'] - input_annotation['y2'])
    annotations.append({
      'id': index,
      'image_id': next(image for image in images if image['file_name'] == input_annotation['filename'])['id'],
      'category_id': next(category for category in categories if category['name'] == input_annotation['category'])['id'],
      'iscrowd': 0,
      'area': width*height,
      'bbox': [
        input_annotation['x1'], # x
        input_annotation['y1'], # y
        width, # width
        height, # height
      ]
    })
  
  # Final json
  data = {
    'info': INFO,
    'images': images,
    'categories': categories,
    'annotations': annotations,
    'licences': licences
  }
  with open(file, 'w') as outfile:
    json.dump(data, outfile)

def split_dataset(annotation, split):
  train_split, val_split, test_split = split
  images = annotations.filename.unique()

  if test_split is not 0:
    # Separate test from dataset taking test_split% off
    test_images = random.sample(set(images), math.ceil(len(images)*(test_split/100)))
    images = list(set(images) - set(test_images))
    # Move test image files
    move_files(test_images, INPUT_IMAGES_FOLDER, f"{DATA_FOLDER}/test{DATASET_ID}")

  if val_split is not 0:
    # Separate val from dataset taking val_split% off
    val_images = random.sample(set(images), math.ceil(len(images)*(val_split/100)))
    images = list(set(images) - set(val_images))
    # Move val image files
    move_files(val_images, INPUT_IMAGES_FOLDER, f"{DATA_FOLDER}/val{DATASET_ID}")

  # Move the rest of the images to train
  move_files(images, INPUT_IMAGES_FOLDER, f"{DATA_FOLDER}/train{DATASET_ID}")

  # Delete Images folder only if it has been left empty
  os.rmdir(INPUT_IMAGES_FOLDER)

def create_annotations(annotations, categories):
  # Convert annotations to Coco Json
  Path(OUTPUT_ANNOTATIONS_FOLDER).mkdir(parents=True, exist_ok=True)
  # category_id is one indexed so we add one to its index
  categories = [{ 'id': index+1, 'name': category, 'supercategory': 'shape' } for index, category in enumerate(categories)]

  if os.path.exists(f"{DATA_FOLDER}/test{DATASET_ID}"):
    test_files = glob.glob(f"{DATA_FOLDER}/test{DATASET_ID}/*.jpg")
    test_annotations = annotations.copy(deep = True)[annotations.filename.isin([test_file.split("/")[-1] for test_file in test_files])]
    print(f'Generating {len(test_annotations)} Test Annotations')
    coco_annotations_from_csv_to_json(test_annotations, categories, f"{OUTPUT_ANNOTATIONS_FOLDER}/instances_test{DATASET_ID}.json")
  else:
    print('Warning: No test set was provided')

  if os.path.exists(f"{DATA_FOLDER}/val{DATASET_ID}"):
    val_files = glob.glob(f"{DATA_FOLDER}/val{DATASET_ID}/*.jpg")
    val_annotations = annotations.copy(deep = True)[annotations.filename.isin([val_file.split("/")[-1] for val_file in val_files])]
    print(f'Generating {len(val_annotations)} Val Annotations')
    coco_annotations_from_csv_to_json(val_annotations, categories, f"{OUTPUT_ANNOTATIONS_FOLDER}/instances_val{DATASET_ID}.json")
  else:
    print('Warning: No val set was provided')
  
  if os.path.exists(f"{DATA_FOLDER}/train{DATASET_ID}"):
    train_files = glob.glob(f"{DATA_FOLDER}/train{DATASET_ID}/*.jpg")
    train_annotations = annotations.copy(deep = True)[annotations.filename.isin([train_file.split("/")[-1] for train_file in train_files])]
    print(f'Generating {len(train_annotations)} Train Annotations')
    coco_annotations_from_csv_to_json(train_annotations, categories, f"{OUTPUT_ANNOTATIONS_FOLDER}/instances_train{DATASET_ID}.json")
  else:
    print('Warning: No train set was provided')

  # Delete old CSV annotation files
  os.remove(INPUT_ANNOTATIONS_TEST)
  os.remove(INPUT_ANNOTATIONS)

def create_project_yml(categories):
  project = {
    # also the folder name of the dataset that under data_path folder
    'project_name': f'dataset-{DATASET_ID}',
    'train_set': f'train{DATASET_ID}',
    'val_set': f'val{DATASET_ID}',
    'test_set': f'test{DATASET_ID}',
    'num_gpus': 1,

    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],

    # this is coco anchors, change it if necessary
    'anchors_scales': '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]',
    'anchors_ratios': '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]',

    # must match your dataset's category_id.
    # category_id is one_indexed,
    # for example, index of 'car' here is 2, while category_id of is 3
    # obj_list: ['person', 'bicycle', 'car', ...]
    'obj_list': list(categories)
  }

  with open(PROJECT_FILE, 'w') as project_file:
    yaml.dump(project, project_file)

if __name__ == '__main__':
  if os.path.exists(INPUT_ANNOTATIONS) and os.path.exists(INPUT_ANNOTATIONS_TEST):
    # Separate images from test set
    test_annotations = pd.read_csv(INPUT_ANNOTATIONS_TEST, names = ['filename', 'x1', 'y1', 'x2', 'y2', 'category'])
    annotations = pd.read_csv(INPUT_ANNOTATIONS, names = ['filename', 'x1', 'y1', 'x2', 'y2', 'category'])

    annotations = annotations.append(test_annotations)
    categories = annotations.category.unique()

    # Split images into folders if it hasn't been done
    if os.path.exists(INPUT_IMAGES_FOLDER):
      split_dataset(annotations, SPLIT)

    create_annotations(annotations, categories)
  
    create_project_yml(categories)
  else:
    print('Please provide old csv annotations')
    sys.exit()

  