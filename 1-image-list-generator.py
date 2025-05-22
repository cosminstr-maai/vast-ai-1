import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bbox-file', type=str)
parser.add_argument('--object-id', type=str)
parser.add_argument('--range', type=str)

args = parser.parse_args()
index1 = int(args.range.split('-')[0])
index2 = int(args.range.split('-')[1])
# print(args.annotation_bbox_file.split('-')[0])


train_annotations = pd.read_csv(args.bbox_file)
filtered_annotations = train_annotations[train_annotations['LabelName'].isin([args.object_id])]

image_ids = filtered_annotations['ImageID'].unique()

if args.bbox_file.split('-')[0] == 'train':
    open("IMAGE_LIST_TRAIN.txt", "w").close()

    for id in image_ids[index1:index2]:
        with open("IMAGE_LIST_TRAIN.txt", 'a') as f:
            f.write(f"train/{id}\n")

elif args.bbox_file.split('-')[0] == 'validation':
    open("IMAGE_LIST_VALIDATION.txt", "w").close()

    for id in image_ids:
        with open("IMAGE_LIST_VALIDATION.txt", "a") as f:
            f.write(f"validation/{id}\n")

    print(f".txt generated")


