import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import random
import panutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home/ubuntu/RawDatasets/panoptic/',
                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--output_path', type=str, default='/home/ubuntu/RawDatasets/panoptic_processed/',
                    help='path to the ANU IKEA assembly video dataset')

args = parser.parse_args()

def reproject_to_3d(im_coords, K, z):
    im_coords = np.stack([im_coords[:,0], im_coords[:,1]],axis=1)
    im_coords = np.hstack((im_coords, np.ones((im_coords.shape[0],1))))
    projected = np.dot(np.linalg.inv(K), im_coords.T).T
    projected[:, 0] = np.multiply(projected[:, 0], z)
    projected[:, 1] = np.multiply(projected[:, 1], z)
    projected[:, 2] = np.multiply(projected[:, 2], z)
    return projected

def vis_keypoints(frame, joints2d):
    for i in range(joints2d.shape[0]):
        if (np.isnan(joints2d[i][0]) or np.isnan(joints2d[i][1])):
            continue
        frame = cv2.circle(frame, (int(joints2d[i][0]), int(joints2d[i][1])), 3, (0, 0, 0), 1)

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

def get_bbox(uv, frame_shape):
    x = min(uv[:, 0]) - 10
    y = min(uv[:, 1]) - 10

    x_max = min(max(uv[:, 0]) + 10, frame_shape[1])
    y_max = min(max(uv[:, 1]) + 10, frame_shape[0])

    return [
        float(max(0, x)), float(max(0, y)), float(x_max - x), float(y_max - y)
    ]

def process_helper(sequence, machine_num):
    seq_name = sequence
    data_path = args.dataset_path
    hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/hd/'
    cameras = get_camera_info(os.path.join(data_path, seq_name, f"calibration_{seq_name}.json"))

    output = {
        "images": [],
        "annotations": [],
        "categories": [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person'
        }]
    }
    image_idx = 0
    annotation_idx = 0

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
        frame_idx = 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

            detected_groundtruth = []
            for body in bframe['bodies']:
                skel = np.array(body['joints19']).reshape((-1,4)).transpose()

                joint_world = skel[0:3]
                valid = skel[3,:]>0.1

                joints_cam = (np.dot(cam['R'], joint_world) + cam['t']).T
                joints_img = panutils.projectPoints(joints_cam.T,
                      cam['K'], np.eye(3), np.zeros((3, 19)), 
                      cam['distCoef']).T
                
                valid = np.logical_and(valid, np.logical_not(np.logical_or(joints_img[:, 0] > frame.shape[1], joints_img[:, 1] > frame.shape[0])))
                joints_img = (joints_img.T * valid).T
                
                if (np.all(joints_img == np.zeros((19, 3)))):
                    continue
                
                # Resizing 1920p to 640p
                joints_img *= (1.0/3.0)
                joints_cam_new = reproject_to_3d(joints_img, cam["K"], joints_cam[:, 2])
                joints_img_new = panutils.projectPoints(joints_cam_new.T,
                      cam['K'], np.eye(3), np.zeros((3, 1)),
                      cam['distCoef']).T
                detected_groundtruth.append({
                    "id": annotation_idx,
                    "image_id": image_idx,
                    "category_id": 1,
                    "is_crowd": 0,
                    "joint_cam": joints_cam.tolist(),
                    "bbox": get_bbox(joints_img, frame.shape) # x, y, w, h
                })
                frame = vis_keypoints(cv2.resize(frame, (640, 360)), joints_img_new)

                annotation_idx += 1

            if (len(detected_groundtruth) > 0):
                output_dir = os.path.join(
                    args.output_path,
                    seq_name,
                    f"view_{i:02d}"
                )
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(f"{output_dir}/{frame_idx}.jpg", frame)
                K = cam['K']
                output["images"].append({
                    "id": image_idx,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "file_name": os.path.join(seq_name, f"view_{i:02d}", f"{frame_idx}.jpg"),
                    "camera_param": {
                        "focal": [float(K[0, 0]), float(K[1, 1])],
                        "princpt": [float(K[0, 2]), float(K[1, 2])]
                    }
                })

            if (frame_idx % 1000 == 0):
                print(f"Finished {frame_idx} of {total * 10} on machine {machine_num}")
            frame_idx += 1
            image_idx += 1
            hd_idx += 1

def process(sequences):
    for sequence in sequences:
        process_helper(sequence)

if __name__ == "__main__":
    f = open(os.path.join(args.dataset_path, 'sequences'), 'r')
    sequences = [x.strip() for x in f.readlines()]
    f.close()
    process(sequences)
