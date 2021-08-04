import os
from shutil import copyfile

'''
    Tiny ImageNet URL: https://www.kaggle.com/c/tiny-imagenet
    refactor tiny imagenet dataset in the val set;
    there're some problems with it (e.g. NaN loss) if you don't run this file
'''

DIR = '/path/to/tiny-imagenet-200/folder'
DATASET = 'val'
ANNOTATION = 'val_annotations.txt'

def unique(l): 
    unique_list = [] 
    
    for x in l: 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

image_folder = os.path.join(DIR, DATASET, 'images')
print(len(os.listdir(image_folder)))

annotation_file = os.path.join(DIR, DATASET, ANNOTATION)
with open(annotation_file) as f:
    data = [x.strip().split('\t')[:2] for x in f.readlines()]
    classes = unique([row[1] for row in data])

validation_folder = os.path.join(DIR, 'validation')

if 'validation' not in os.listdir(DIR):
    os.mkdir(validation_folder) 

for c in classes:
    if c not in os.listdir(validation_folder):
        os.mkdir(os.path.join(validation_folder, c))

for d in data:
    copyfile(os.path.join(image_folder, d[0]), os.path.join(validation_folder, d[1], d[0]))