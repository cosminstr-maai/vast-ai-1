import csv, pathlib, tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train-bbox-file', type=str)
parser.add_argument('--val-bbox-file', type=str)
parser.add_argument('--train-range', type=str)
parser.add_argument('--val-range', type=str)
parser.add_argument('--object-id', type=str)

args = parser.parse_args()

index1_train = int(args.train_range.split('-')[0])
index2_train = int(args.train_range.split('-')[1])
index1_val = int(args.val_range.split('-')[0])
index2_val = int(args.val_range.split('-')[1])

train_annotations = pd.read_csv(args.train_bbox_file)
filtered_train_annotations = train_annotations[train_annotations['LabelName'].isin([args.object_id])]
filtered_train_annotations = filtered_train_annotations.drop_duplicates(subset=['ImageID'])

val_annotations = pd.read_csv(args.val_bbox_file)
filtered_val_annotations = val_annotations[val_annotations['LabelName'].isin([args.object_id])]
filtered_val_annotations = filtered_val_annotations.drop_duplicates(subset=['ImageID'])

train_image_ids = filtered_train_annotations['ImageID'].unique()
val_image_ids = filtered_val_annotations['ImageID'].unique()


print("\n-------Generare IMAGE_LIST.txt pentru scriptul downloader.py-------\n")

open("IMAGE_LIST_TRAIN.txt", "w").close()
open("IMAGE_LIST_VALIDATION.txt", "w").close()

for id in train_image_ids[index1_train:index2_train]:
    with open("IMAGE_LIST_TRAIN.txt", "a") as f:
        f.write(f"train/{id}\n")

print("IMAGE_LIST_TRAIN.txt generat")

for id in val_image_ids[index1_val:index2_val]:
    with open("IMAGE_LIST_VALIDATION.txt", "a") as f:
        # Tot train deoarece scriptul downloader.py va returna 404 daca incearca sa descarce 
        # de la validation un id de la train
        # Se asociaza aceasta lista cu /val hardcodat mai tarziu
        f.write(f"train/{id}\n")

print("IMAGE_LIST_VALIDATION.txt generat")

print("\n-------Generare csv-uri cu adnotarile imaginilor pentru fieacre subset pentru scriptul downloader.py-------\n")

filtered_train_annotations[index1_train:index2_train].to_csv('train-annotations.csv')
filtered_val_annotations[index1_val:index2_val].to_csv('validation-annotations.csv')

print("train-annotations.csv si validation-annotations.csv generate")


print("\n-------Conversie din format OpenImageDatasetV7 in format YOLO-------\n")

annotations_file = pathlib.Path(f"/workspace/fine-tune/train-annotations.csv")
images_directory = pathlib.Path("/workspace/fine-tune/images")


labels_directory = images_directory.parent / "labels/train"
labels_directory.mkdir(parents=True, exist_ok=True)

with annotations_file.open() as f:
    reader = csv.DictReader(f)
    for row in tqdm.tqdm(reader, desc="writing YOLO .txt"):
        img_id  = row["ImageID"]
        txt_out = labels_directory / f"{img_id}.txt"
        xmin, xmax = float(row["XMin"]), float(row["XMax"])
        ymin, ymax = float(row["YMin"]), float(row["YMax"])
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        bw =  xmax - xmin
        bh =  ymax - ymin
        with txt_out.open("a") as t:                 
            t.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Format YOLO pentru /train salvat in {labels_directory}")


annotations_file = pathlib.Path(f"/workspace/fine-tune/validation-annotations.csv")

labels_directory = images_directory.parent / "labels/val"
labels_directory.mkdir(parents=True, exist_ok=True)

with annotations_file.open() as f:
    reader = csv.DictReader(f)
    for row in tqdm.tqdm(reader, desc="writing YOLO .txt"):
        img_id  = row["ImageID"]
        txt_out = labels_directory / f"{img_id}.txt"
        xmin, xmax = float(row["XMin"]), float(row["XMax"])
        ymin, ymax = float(row["YMin"]), float(row["YMax"])
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        bw =  xmax - xmin
        bh =  ymax - ymin
        with txt_out.open("a") as t:                 
            t.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"format YOLO pentru /val salvat in {labels_directory}")
