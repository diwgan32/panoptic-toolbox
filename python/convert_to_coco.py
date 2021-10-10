import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home/ubuntu/RawDatasets/panoptic/',
                    help='path to the ANU IKEA assembly video dataset')

args = parser.parse_args()


def vis_keypoints(frame, joints2d):
    for i in range(joints2d.shape[0]):
        if (np.isnan(joints2d[i][0]) or np.isnan(joints2d[i][1])):
            continue
        frame = cv2.circle(frame, (int(joints2d[i][0]), int(joints2d[i][1])), 5, (0, 0, 0), 2)

    return frame

def project_3D_points(cam_mat, pts3D):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2
    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)
    assert len(proj_pts.shape) == 2
    return proj_pts

def get_camera_info(calibration_file):
    # Load camera calibration parameters
    with open(calibration_file) as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

    # Convert data into numpy arrays for convenience
    for k,cam in cameras.items():    
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1))

    return cameras


def process_helper(sequence):
    seq_name = sequence
    data_path = args.dataset_path
    hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/hd/'
    cameras = get_camera_info(os.path.join(data_path, seq_name, f"calibration_{seq_name}.json"))
    for i in range(10):
        hd_vid_name = f"hd_00_{i:02d}.mp4"
        hd_vid_path = os.path.join(data_path, seq_name, "hdVideos", hd_vid_name)
        if (not os.path.exists(hd_vid_path)):
            print(f"Path {hd_vid_path} does not exist")
        
        cap = cv2.VideoCapture(hd_vid_path)
        if (not cap.isOpened()):
            print(f"Could not open {hd_vid_path}")
        
        cam = cameras[(0, i)]       
        hd_idx = 0
        while True:
            ret, frame = cap.read()
            if (not ret):
                break
 
            skel_json_fname = hd_skel_json_path+'body3DScene_{0:08d}.json'.format(hd_idx)
            if (not os.path.exists(skel_json_fname)):
                hd_idx += 1
                continue
            with open(skel_json_fname) as dfile:
                bframe = json.load(dfile)

            if (len(bframe["bodies"]) == 0):
                hd_idx += 1
                continue

            for body in bframe['bodies']:
                skel = np.array(body['joints19']).reshape((-1,4)).transpose()

                joint_world = skel[0:3]
                joints_cam = (np.dot(cam['R'], joint_world) + cam['t']).T
                joints_img = panutils.projectPoints(joints_cam,
                      cam['K'], np.eye(3), np.zeros(3), 
                      cam['distCoef'])
                frame = vis_keypoints(frame, joints_img)
            
            cv2.imwrite(f"{random.randint(1, 100)}.jpg", frame)
            input("? ")
            hd_idx += 1

def process(sequences):
    for sequence in sequences:
        process_helper(sequence)

if __name__ == "__main__":
    f = open(os.path.join(args.dataset_path, 'sequences'), 'r')
    sequences = [x.strip() for x in f.readlines()]
    f.close()
    process(sequences)
