import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bbox-file', type=str)
parser.add_argument('--object-id', type=str)
parser.add_argument('--range', type=str)

args = parser.parse_args()
index1 = int(args.range.split('-')[0])
index2 = int(args.range.split('-')[1])

annotations = pd.read_csv(args.bbox_file)
annotations_final = annotations[annotations['LabelName'].isin([args.object_id])]
annotations_final = annotations_final.drop_duplicates(subset=['ImageID'])


if args.bbox_file.split('-')[0] == 'train':
    annotations_final[index1:index2].to_csv('train-annotations.csv')

elif args.bbox_file.split('-')[0] == 'validation':
    annotations_final.to_csv('validation-annotations.csv')



