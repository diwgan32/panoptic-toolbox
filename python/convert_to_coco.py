import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home/ubuntu/RawDatasets/panoptic/',
                    help='path to the ANU IKEA assembly video dataset')

args = parser.parse_args()


def process_helper(sequence):
    seq_name = sequence
    data_path = args.dataset_path
    hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'
    for i in range(10):
        hd_vid_name = f"hd_00_{i:02d}.mp4"
        hd_vid_path = os.path.join(data_path, seq_name, "hdVideos", hd_vid_name)
        if (not os.path.exists(hd_vid_path)):
            print(f"Path {hd_vid_path} does not exist")
        cap = cv2.VideoCapture(hd_vid_path)
        if (not cap.isOpened()):
            print(f"Could not open {hd_vid_path}")
        

def process(sequences):
    for sequence in sequences:
        process_helper(sequence)

if __name__ == "__main__":
    f = open(os.path.join(args.dataset_path, 'sequences'), 'r')
    sequences = [x.strip() for x in f.readlines()]
    f.close()
    process(sequences)
