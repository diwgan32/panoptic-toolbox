import numpy as np
import json
import matplotlib.pyplot as plt
import process_joints
import angles

NUM_SAMPLES = 27886
START_SAMPLE = 0
# 1290 bend over
DISP_3D = False
#%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'

# Setup paths
data_path = '../'
seq_name = '171204_pose1'

hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'
hd_face_json_path = data_path+seq_name+'/hdFace3d/'
hd_hand_json_path = data_path+seq_name+'/hdHand3d/'
#hd_img_path = data_path+seq_name+'/hdImgs/'

colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

# Load camera calibration parameters (for visualizing cameras)
with open(data_path+seq_name+'/calibration_{0}.json'.format(seq_name)) as cfile:
    calib = json.load(cfile)

# Cameras are identified by a tuple of (panel#,node#)
cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

# Convert data into numpy arrays for convenience
for k,cam in cameras.items():    
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3,1))


# Choose only HD cameras for visualization
hd_cam_idx = zip([0] * 30,range(0,30))
hd_cameras = [cameras[cam].copy() for cam in hd_cam_idx]
all_angles = np.zeros((NUM_SAMPLES, 14))
print("done reading")
for i in range(START_SAMPLE, NUM_SAMPLES):
    # Select HD Image index
    hd_idx = i
    

    # Draw all cameras in black
    # for k,cam in cameras.iteritems():
    #     cc = (-cam['R'].transpose()*cam['t'])
    #     ax.scatter(cc[0], cc[1], cc[2], '.', color=[0,0,0])


   
    if (i % 50 == 0):
        print(i)
    '''
    ## Visualize 3D Body
    '''
    # Edges between joints in the body skeleton
    body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1

    hand_edges = np.array(
        [[0,1],[1,2],[2,3],[3,4],
                    [0,5],[5,6],[6,7],[7,8],
                    [0,9],[9,10],[10,11],[11,12],
                    [0,13],[13,14],[14,15],[15,16],
                    [0,17],[17,18],[18,19],[19,20]]
    )
    # Load the json file with this frame's skeletons
    skel_json_fname = hd_skel_json_path+'body3DScene_{0:08d}.json'.format(hd_idx)
   
    try:
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)
    except:
        all_angles[i] = np.zeros(14).fill(np.nan)
        print(f"Frame {i} not found")
        continue

    


    if (DISP_3D):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev = -90, azim=-90)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        ax.set_xlim([-100, 100])
        ax.set_ylim([-200, 0])
        ax.set_zlim([-100, 100])
         # Draw selected camera subset in blue
        # for cam in hd_cameras:
        #     cc = (-cam['R'].transpose()*cam['t'])
        #     ax.scatter(cc[0], cc[1], cc[2], '.', color=[0,0,1])

        #ax.axis('auto')
    else:
        ax = None

    # Load the json file with this frame's face
    hand_json_fname = hd_hand_json_path+'handRecon3D_hd{0:08d}.json'.format(hd_idx)
    try:
        with open(hand_json_fname) as dfile:
            hframe = json.load(dfile)
        hand_joints = process_joints.extract_hand_end_pos(hframe['people'])
    except:
        hand_joints = {"L": None, "R": None}

    
    
    ids = 0
    if (len(bframe['bodies']) < 1):
        all_angles[i] = np.zeros(14).fill(np.nan)
        continue
    body = bframe['bodies'][ids]
    joints = process_joints.get_all_3d_joints(
        body["joints19"]
    )
    
    trunk_angle = angles._get_trunk_angle_3D(joints)
    neck_angle = angles._get_neck_angle_3d(joints)
    l_shoulder = angles._get_lateral_arm_angle3D(joints, "L")
    r_shoulder = angles._get_lateral_arm_angle3D(joints, "R")
    l_shoulder_planar = angles._get_planar_arm_angle3D(joints, "L")
    r_shoulder_planar = angles._get_planar_arm_angle3D(joints, "R")
    l_elbow = angles._get_elbow_angle3D(joints, "L")
    r_elbow = angles._get_elbow_angle3D(joints, "R")
    neck_twist = angles.get_neck_twist_angle(joints)
    trunk_twist = angles.get_trunk_twist_factor(joints)
    neck_sidebend = angles.get_neck_lateral_angle(joints, ax)
    trunk_sidebend = angles.get_trunk_side_angle_3d(joints)[0]
    l_wrist = angles.get_wrist_angle(joints, hand_joints, "L")
    r_wrist = angles.get_wrist_angle(joints, hand_joints, "R")
    if (DISP_3D):
        print(f"Frame: {i}")
        print(f"Left shoulder: {l_shoulder}")
        print(f"Right shoulder: {r_shoulder}")
        print(f"Left shoulder planar: {l_shoulder_planar}")
        print(f"Right shoulder planar: {r_shoulder_planar}")
        print(f"Trunk angle: {trunk_angle}")
        print(f"Neck angle: {neck_angle}")
        print(f"Neck twist: {neck_twist}")
        print(f"Trunk twist: {trunk_twist}")
        print(f"Neck sidebend: {neck_sidebend}")
        print(f"Left elbow: {l_elbow}")
        print(f"Right elbow: {r_elbow}")
        print(f"Left wrist: {wrist_left}")
        print(f"Right wrist: {wrist_right}")
        print("---")
    all_angles[i] = np.array([
        trunk_angle,
        trunk_twist,
        trunk_sidebend,
        neck_angle,
        neck_twist,
        neck_sidebend,
        l_shoulder,
        r_shoulder,
        l_shoulder_planar,
        r_shoulder_planar,
        l_elbow,
        r_elbow,
        l_wrist,
        r_wrist
    ])
    skel = np.array(body['joints19']).reshape((-1,4)).transpose()
    skel[0, :] /= 1
    skel[1, :] /= 1
    skel[2, :] /= 1
    if (DISP_3D):
        for edge in body_edges:
            ax.plot(skel[0,edge], skel[1,edge], skel[2,edge], color=colors[body['id']])
        plt.show()

np.save("171204_pose1_groundtruth.npy", all_angles)