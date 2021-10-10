import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
from core_constants import *
from utils import *

debug = False

def _normalize(vec):
    return vec / np.linalg.norm(vec)

##########################
### PERSON ORIENTATION ###
##########################
def estimate_orientation(joints):
    if (joints == {}):
        return np.nan
    if (test_validity([joints["LEAR"], joints['LEYE']]) and 
        not test_validity([joints["REAR"], joints["REYE"]])):
        return Orientation.LEFT_PROFILE

    if (test_validity([joints["REAR"], joints['REYE']]) and 
        not test_validity([joints["LEAR"], joints["LEYE"]])):
        return Orientation.RIGHT_PROFILE

    if (not test_validity([joints['LEYE']]) and 
        not test_validity([joints["REYE"]])):

        if (test_validity([joints["LSHOULDER"], joints['LHIP']]) and
            not test_validity([joints['RSHOULDER'], joints['RHIP']])):
            return Orientation.LEFT_PROFILE

        if (test_validity([joints["RSHOULDER"], joints['RHIP']]) and
            not test_validity([joints['LSHOULDER'], joints['LHIP']])):

            return Orientation.RIGHT_PROFILE

        return Orientation.BACK
    if (test_validity([joints["LEAR"], joints['LEYE']]) and
        test_validity([joints["REAR"], joints["REYE"]])):

        return Orientation.FRONT

    return Orientation.UNKNOWN

##########################
######### NECK ###########
##########################

# Function: _get_neck_angle_3d
# ------------------------------------
# Parameters:
#   - joints: A mapping from the wrnch joint key name to the joint position.
#             Assumes the given joint information is 3D
# Returns:
#   - angle: The actual neck angle. Returns none if nothing is found
def _get_neck_angle_3d(joints, ax=None):
    if (joints == {}):
        return np.nan
    if (not test_validity([
        joints["REYE"],
        joints["LEYE"],
        joints["RHIP"],
        joints["LHIP"],
        joints["RSHOULDER"],
        joints["LSHOULDER"]
        ])):
        return np.nan
    head = (joints["REYE"] + joints["LEYE"]) / 2.0
    pelvis = (joints["RHIP"] + joints["LHIP"]) / 2.0
    neck = (joints["RSHOULDER"] + joints["LSHOULDER"]) / 2.0
    chest_vec = _normalize(pelvis - neck)
    head_vec = _normalize(head - neck)
    angle = np.arccos(np.dot(head_vec, chest_vec)) * 180.0 / np.pi
    if (not(ax is None)):
        p = joints["NECK"]

        ax.quiver(
            p[0],
            p[1],
            p[2],
            head_vec[0],
            head_vec[1],
            head_vec[2],
            arrow_length_ratio=.01,
            color='b'
        )

        ax.quiver(
            p[0],
            p[1],
            p[2],
            chest_vec[0],
            chest_vec[1],
            chest_vec[2],
            arrow_length_ratio=.01,
            color='b'
        )


    return 180 - (angle + NECK_COMPENSATION_FACTOR_3D)


# Function: _get_neck_angle_2d
# ------------------------------------
# Parameters:
#   - joints: A mapping from the wrnch joint key name to the joint position.
#   - Size of the frame. Needed to accurately compute joint angles
#   - frame: Used for debug purposes to display the joint angles picked up
#            on the camera frame.
# Returns:
#   - angle: The actual neck angle. Returns none if nothing is found
def _get_neck_angle_2d(joints, frame=None):
    if (joints == {}):
        return np.nan
    if (not test_validity([joints["HEAD"], joints["NECK"], joints["PELV"]])):
        return np.nan
    head = None
    if (not test_validity([joints["LEAR"]]) and not test_validity([joints["REAR"]])):
        return np.nan
    elif (not test_validity([joints["LEAR"]]) and test_validity([joints["REAR"]])):
        if (test_validity([joints["NOSE"]])):
            head = joints["REAR"]
        else:
            head = joints["REAR"]
    elif (test_validity([joints["LEAR"]]) and not test_validity([joints["REAR"]])):
        if (test_validity([joints["NOSE"]])):
            head = joints["LEAR"]
        else:
            head = joints["LEAR"]
    elif (test_validity([joints["LEAR"]]) and test_validity([joints["REAR"]])):
        head = (joints["LEAR"] + joints["REAR"]) / 2.0
    chest_vec = _normalize(joints["PELV"] - joints["NECK"])
    head_vec = _normalize(head - joints["NECK"])
    angle = np.arccos(np.dot(head_vec, chest_vec)) * 180.0 / np.pi
    if (not (frame is None)):
        startPoint = joints["NECK"]
        endPoint = startPoint + head_vec * 300
        endPoint2 = startPoint + chest_vec * 300
        frame = cv2.circle(frame, (tuple(head.astype(np.int))), 10, (0, 255, 0), -1)
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint.astype(np.int)), (255, 0, 0), 1)
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint2.astype(np.int)), (0, 255, 0), 1)

    return max(180 - angle - NECK_COMPENSATION_FACTOR_2D, 0)


# Function: get_neck_twist_angle
# ------------------------------------
# Parameters:
#   - joints: A mapping from the wrnch joint key name to the joint position.
#             Expects 3D joints
#   - frame: Used for debug purposes to display the joint angles picked up
#            on the camera frame.
# Returns:
#   - angle: An estimate of how twisted the neck is in degrees

def get_neck_twist_angle(joints):
    if (joints == {}):
        return np.nan
    if (not test_validity([joints["RSHOULDER"], joints["LSHOULDER"], joints["REAR"], joints["LEAR"]])):
        return np.nan
    ear_vector = _normalize(joints["LEAR"] - joints["REAR"])
    shoulder_vector = _normalize(joints["LSHOULDER"] - joints["RSHOULDER"])
    return np.arccos(np.dot(ear_vector, shoulder_vector)) * 180.0 / np.pi

def _getChestPlaneHelper(shoulder_left, shoulder_right, hip_left, hip_right):
    if (not test_validity([shoulder_left, shoulder_right, hip_left, hip_right])):
        return np.nan
    shoulder_hip_vector_1 = shoulder_right - hip_right
    shoulder_hip_vector_2 = shoulder_left - hip_left
    hip_vector = _normalize(hip_left - hip_right)
    chest_normal_1 = np.cross(hip_vector, shoulder_hip_vector_1)
    chest_normal_2 = np.cross(shoulder_hip_vector_2, -hip_vector)
    return -_normalize((chest_normal_1 + chest_normal_2) / 2.0)

# Function: get_neck_lateral_angle
# ------------------------------------
# Computes the side-bending angle of the neck.
# Parameters:
#   - joints: 3D joints
#   - ax: Used for debug purposes to display 3D lines
# Returns:
#   - angle: An estimate of how side bent the neck is in degrees
def get_neck_lateral_angle(joints, ax=None):
    """
    Input:
      - joints: 3D joints
      - side: either "L" or "R" depending on the side
      - ax: A reference to a matplotlib 3D plot that's used for debug purposes
    Output:
      - angle: Returns the lateral angle of the arms
    """
    if (joints == {}):
        return np.nan
    if (not test_validity([
            joints["LSHOULDER"],
            joints["RSHOULDER"],
            joints["RHIP"],
            joints["LHIP"],
            joints["REYE"],
            joints["LEYE"]
        ])):
        return np.nan
    hip_right = joints["RHIP"]
    hip_left = joints["LHIP"]
    thorax = (joints["LSHOULDER"] + joints["RSHOULDER"])/2
    head =  (joints["REYE"] + joints["LEYE"])/2.0
   
    if (not test_validity([hip_right, hip_left, thorax])):
        return np.nan
    hip_vector = _normalize(joints["RHIP"] - joints["LHIP"])
    shoulder_vector = _normalize(joints["RSHOULDER"] - joints["LSHOULDER"])
    avg_lat_vec_3d = _normalize((hip_vector + shoulder_vector)/2.0)

    neck_point = (joints["RSHOULDER"] + joints["LSHOULDER"])/2.0
    neck_vec_prenorm = head - neck_point
    neck_vec = _normalize(head - neck_point)
    body_vec = _normalize(joints["PELV"] - thorax)
    x = -avg_lat_vec_3d
    y = -body_vec

    z = np.cross(x, y)
   
    basis = np.zeros((3, 3))
    basis[:, 0] = x
    basis[:, 1] = y
    basis[:, 2] = z

    neck_vec_in_basis = np.dot(basis.T, neck_vec)
    neck_vec_in_basis_prenorm = np.dot(basis.T, neck_vec_prenorm)
    if (not (ax is None)):
        p =neck_point

        p2 = x * 100
        p3 = y * 100
        p4 = z * 100
        p5 = neck_vec_in_basis_prenorm
        ax.quiver(p[0],
                  p[1],
                  p[2],
                  p2[0],
                  p2[1],
                  p2[2],
                  arrow_length_ratio=.01,
                  color='black')
        ax.quiver(p[0], p[1], p[2], p3[0], p3[1], p3[2], arrow_length_ratio=.01, color='green')
        ax.quiver(p[0], p[1], p[2], p4[0], p4[1], p4[2], arrow_length_ratio=.01, color='blue')
        #ax.quiver(p[0], p[1], p[2], p5[0], p5[1], p5[2], arrow_length_ratio=.01, color="red")
    #raise_angle = np.arccos(np.dot(elbow_shoulder_onto_hip, body_vec)) * 180.0/np.pi
    raise_angle = (np.arctan2(neck_vec_in_basis[0],neck_vec_in_basis[1]) * 180.0/np.pi)
    return np.abs(raise_angle)


# Function: get_neck_angle
# ------------------------------------
# Parameters:
#   - single_frame_3d_joints: 3D joints
#   - single_frame_2d_joints: 2D joints
#   - frame: Used for debug purposes to annotate the camera recording
# Returns:
#   - angle: An estimate of how twisted the neck is in degrees


def get_neck_angle(single_frame_3d_joints, single_frame_2d_joints, frame=None):
    neckAngle2D = _get_neck_angle_2d(single_frame_2d_joints, frame)
    neckAngle3D = _get_neck_angle_3d(single_frame_3d_joints)
    trunkAngle3D = _get_trunk_angle_3D(single_frame_3d_joints)
    trunkAngle2D = _get_trunk_angle_2D(single_frame_2d_joints)

    if (np.isnan(trunkAngle2D) or trunkAngle2D >= 20):
        return np.nan, AngleTypeUsed.TWO_D

    if (not np.isnan(neckAngle3D) and not np.isnan(trunkAngle3D) and trunkAngle3D < 10):
        return neckAngle3D, AngleTypeUsed.THREE_D

    return neckAngle2D, AngleTypeUsed.TWO_D


##########################
######### TRUNK ##########
##########################

def get_front_facing_angle(joints, ax=None):
    # Measure the angle w.r.t the chest and the x-axis
    if (joints == {}):
        return np.nan
    if (not test_validity([joints["LHIP"], joints["RHIP"], joints["LSHOULDER"], joints["RSHOULDER"],
        joints["HEAD"], joints["NECK"]])):
        return np.nan
    hip_right = joints["RHIP"]
    hip_left = joints["LHIP"]
    shoulder_right = joints["RSHOULDER"]
    shoulder_left = joints["LSHOULDER"]
    chest_vec = _getChestPlaneHelper(shoulder_left, shoulder_right, hip_left, hip_right)
    chest_vec[2] = 0
    head_vec = joints["HEAD"] - joints["NECK"]
    head_vec[2] = 0
    head_vec = _normalize(head_vec)
    facing_side_angle_1 = 180.0/np.pi  * np.arccos(np.dot(_normalize(chest_vec), np.array([1, 0, 0])))
    facing_side_angle_2 = 180.0/np.pi  * np.arccos(np.dot(_normalize(head_vec), np.array([1, 0, 0])))

    facing_side_angle = (facing_side_angle_2 + facing_side_angle_1)/2.0
    if (not(ax is None)):
        p = joints["NECK"]

        ax.quiver(p[0],
                  p[1],
                  p[2],
                  head_vec[0],
                  head_vec[1],
                  head_vec[2],
                  arrow_length_ratio=.01,
                  color='g')
        
    return facing_side_angle

def is_trunk_facing_side(joints, frame=None):
    # This is an angle that's an estimate of if the person is facing the camera
    # If they are, their chest should be roughly aligned with the x-axis, so this angle 
    # should either be very high or very low.
    facing_side_angle = get_front_facing_angle(joints)
    print(str(facing_side_angle))

    # If you can't see them properly then assume they are facing the side
    if (np.isnan(facing_side_angle)):
        return True

    if (facing_side_angle > 160 or facing_side_angle < 20):
        return False

    return True


# Function: _get_trunk_angle_2D
# ------------------------------------
# Parameters:
#   - joints: A mapping from the wrnch joint key name to the joint position.
# Returns:
#   - angle: The actual trunk angle. Returns none if nothing is found
def _get_trunk_angle_2D(joints, frame=None):
    if (joints == {}):
        return np.nan
    if (not test_validity([joints["THRX"], joints["PELV"]])):
        return np.nan
    knee = pick_valid_side(joints["RKNEE"], joints["LKNEE"])
    if (knee is None):
        leg_vec = np.array([0.0, 100.0])
    else:
        leg_vec = knee - joints["PELV"]
    body_vec = joints["THRX"] - joints["PELV"]
  
    if (not (frame is None)):
        startPoint = joints["PELV"]
        endPoint = startPoint + body_vec
        endPoint2 = startPoint + leg_vec

        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint.astype(np.int)), (255, 0, 0), 4)
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint2.astype(np.int)), (0, 255, 0), 4)
    leg_vec = _normalize(leg_vec)
    body_vec = _normalize(body_vec)
    return 180 - np.arccos(np.dot(body_vec, leg_vec)) * 180.0 / np.pi


# Function: get_trunk_angle_3D
# ------------------------------------
# Parameters:
#   - joints: A mapping from the wrnch joint key name to the joint position.
# Returns:
#   - angle: The actual trunk angle. Returns none if nothing is found
def _get_trunk_angle_3D(joints):
    if (joints == {}):
        return np.nan
    if (not test_validity(
        [joints["PELV"], joints["LHIP"], joints["RHIP"], joints["NECK"]])):
        return np.nan
    pelvis_neck_vector = _normalize(joints["NECK"] - joints["PELV"])
    right_leg_vector = np.array([0, -1, 0])
    left_leg_vector = np.array([0, -1, 0])

    left_angle = np.arccos(np.dot(pelvis_neck_vector, left_leg_vector)) * 180.0 / np.pi
    right_angle = np.arccos(np.dot(pelvis_neck_vector, right_leg_vector)) * 180.0 / np.pi

    return (left_angle + right_angle) / 2.0


# Function: get_trunk_side_angle_3d
# ------------------------------------
# Parameters:
#   - joints: A mapping from the wrnch joint key name to the joint position.
#             This function only works with 3D info
# Returns:
#   - angle: Returns the side bending angle of the trunk
#            (angle bent not forward, but sideways)
def get_trunk_side_angle_3d(joints):
    if (joints == {}):
        return np.nan, AngleTypeUsed.THREE_D
    if (not test_validity([joints["LHIP"], joints["RHIP"], joints["LSHOULDER"], joints["RSHOULDER"]])):
        return np.nan, AngleTypeUsed.THREE_D
    hip_vector = _normalize(joints["RHIP"] - joints["LHIP"])
    up_vector = np.array([0, 0, 1])

    neck = (joints["LSHOULDER"] + joints["RSHOULDER"]) / 2.0
    pelvis = (joints["LHIP"] + joints["RHIP"]) / 2.0

    body_vec = _normalize(neck - pelvis)

    angle_1 = np.arccos(np.dot(body_vec, up_vector)) * 180.0 / np.pi
    return angle_1, AngleTypeUsed.THREE_D


# Function: get_trunk_angle
# ------------------------------------
# Parameters:
#   - single_frame_3d_joints: 3D joints
#   - single_frame_2d_joints: 2D joints
#   - frame: Used for debug purposes to annotate the camera recording
# Returns:
#   - angle: Returns the side bending angle of the trunk
#            (angle bent not forward, but sideways)


def get_trunk_angle(single_frame_3d_joints, single_frame_2d_joints, orientation, frame=None, ax=None):
    trunkAngle3D = _get_trunk_angle_3D(single_frame_3d_joints)
    trunkAngle2D = _get_trunk_angle_2D(single_frame_2d_joints, frame)
    if (orientation != Orientation.FRONT):
        return trunkAngle2D, AngleTypeUsed.TWO_D

    if (not np.isnan(trunkAngle3D)):
        return trunkAngle3D, AngleTypeUsed.THREE_D
    return trunkAngle2D, AngleTypeUsed.TWO_D


# Function: get_trunk_twist_factor
# ------------------------------------
# Parameters:
#   - joints: 3D joints
#   - ax: Used for debug purposes to annotate the 3d plt
# Returns:
#   - factor: Returns the cross product between the shoulder vector and the hip vector
def get_trunk_twist_factor(joints, ax=None):
    if (joints == {}):
        return np.nan
    hip_right = joints["RHIP"]
    hip_left = joints["LHIP"]
    shoulder_right = joints["RSHOULDER"]
    shoulder_left = joints["LSHOULDER"]
    if (not test_validity([hip_right, hip_left, shoulder_right, shoulder_left])):
        return np.nan
    hip_vector = _normalize(hip_right - hip_left)
    shoulder_vector = _normalize(shoulder_right - shoulder_left)
    hip_twist = np.linalg.norm(np.cross(hip_vector, shoulder_vector))

    return np.arccos(np.dot(hip_vector, shoulder_vector)) * 180.0 / np.pi


##########################
########## LEGS ##########
##########################

# Function: _getLegAngle3D
# ------------------------------------
# Parameters:
#   - joints: 3D joints
#   - ax: Used for debug purposes
# Returns:
#   - angle: Returns the leg bend angle. Wrnch 3d estimates overshoot
#            the angle by some amount so that needs to be compensated for


def _getLegAngle3D(joints, side, ax=None):
    if (joints == {}):
        return np.nan
    hip = joints[side + "HIP"]
    knee = joints[side + "KNEE"]
    ankle = joints[side + "ANKLE"]
    if (not test_validity([hip, knee, ankle])):
        return np.nan
    femur = _normalize(hip - knee)
    calf = _normalize(ankle - knee)
    angle = np.arccos(np.dot(calf, femur)) * 180.0 / np.pi

    return max(180 - angle - LEG_COMPENSATION_FACTOR, 0)


# Function: _getLegAngle2D
# ------------------------------------
# Parameters:
#   - joints: 2D joints
#   - frame: Used for debug purposes
# Returns:
#   - angle: Same as above but uses 2D


def _getLegAngle2D(joints, side, frame=None):
    if (joints == {}):
        return np.nan
    hip = joints[side + "HIP"]
    knee = joints[side + "KNEE"]
    ankle = joints[side + "ANKLE"]
    if (not test_validity([hip, knee, ankle])):
        return np.nan
    femur = _normalize(hip - knee)
    calf = _normalize(ankle - knee)
    angle = np.arccos(np.dot(calf, femur)) * 180.0 / np.pi

    if (not (frame is None)):
        startPoint = knee
        endPoint = knee + femur * 100
        endPoint2 = knee + calf * 100

        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint.astype(np.int)), (255, 0, 0), 1)
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint2.astype(np.int)), (0, 255, 0), 1)

    return 180 - angle


def getLegAngle(single_frame_3d_joints, single_frame_2d_joints, side, frame=None, ax=None):
    legAngle3D = _getLegAngle3D(single_frame_3d_joints, side)
    legAngle2D = _getLegAngle2D(single_frame_2d_joints, side, frame)
    if (not np.isnan(legAngle3D)):
        return legAngle3D, AngleTypeUsed.THREE_D
    return legAngle2D, AngleTypeUsed.TWO_D


##########################
####### Upper Arm ########
##########################


# Function: _get_planar_arm_angle3D
# ------------------------------------
# Parameters:
#   - joints: 3D joints
#   - side: either "L" or "R" depending on the side
#   - ax: A reference to a matplotlib 3D plot that's used for debug purposes
# Returns:
#   - angle: Returns the planar angle of the arm (used to compute
#            whether the arm is abducted in the RULA/REBA assessment)
def _get_planar_arm_angle3D(joints, side, ax=None):
    """
    Input:
      - joints: 3D joints
      - side: either "L" or "R" depending on the side
      - ax: A reference to a matplotlib 3D plot that's used for debug purposes
    Output:
      - angle: Returns the lateral angle of the arms
    """
    if (joints == {}):
        return np.nan
    if (not test_validity([
            joints[side+"SHOULDER"],
            joints["LSHOULDER"],
            joints["RSHOULDER"],
            joints["RHIP"],
            joints["LHIP"]
        ])):
        return np.nan
    hip_right = joints["RHIP"]
    hip_left = joints["LHIP"]
    elbow = joints[side + "ELBOW"]
    shoulder = joints[side + "SHOULDER"]
    thorax = (joints["LSHOULDER"] + joints["RSHOULDER"])/2
    if (side == "L"):
        other_side = "R"
    else:
        other_side = "L"
    if (not test_validity([hip_right, hip_left, elbow, shoulder, thorax])):
        return np.nan
    hip_vector = _normalize(joints[side+"HIP"] - joints[other_side+"HIP"])
    shoulder_vector = _normalize(joints[side+"SHOULDER"] - joints[other_side+"SHOULDER"])


    avg_lat_vec_3d = _normalize((hip_vector + shoulder_vector)/2.0)
    body_vec = _normalize(joints["PELV"] - thorax)
    x = body_vec
    z = avg_lat_vec_3d

    y = np.cross(z, x)
   
    basis = np.zeros((3, 3))
    basis[:, 0] = x
    basis[:, 1] = y
    basis[:, 2] = z
    elbow_shoulder_prenorm = elbow - shoulder
    elbow_shoulder = _normalize(elbow - shoulder)

    elbow_shoulder_in_basis = np.dot(basis.T, elbow_shoulder)
    elbow_shoulder_in_basis_prenorm = np.dot(basis.T, elbow_shoulder_prenorm)
    pelvis = (hip_right + hip_left) / 2.0
    
    
    # chest_plane = np.cross(body_vec, hip_vector)
    # elbow_shoulder_onto_hip = _normalize(elbow_shoulder - np.dot(elbow_shoulder, hip_vector) * chest_plane)

    if (not (ax is None)):
        p = joints[side + "SHOULDER"]
        p2 = x * 100
        p3 = y * 100
        p4 = z * 100
        ax.quiver(p[0],
                  p[1],
                  p[2],
                  p2[0],
                  p2[1],
                  p2[2],
                  arrow_length_ratio=.01,
                  color='black')
        ax.quiver(p[0], p[1], p[2], p3[0], p3[1], p3[2], arrow_length_ratio=.01, color='green')
        ax.quiver(p[0], p[1], p[2], p4[0], p4[1], p4[2], arrow_length_ratio=.01, color='blue')
    #raise_angle = np.arccos(np.dot(elbow_shoulder_onto_hip, body_vec)) * 180.0/np.pi
    # print(elbow_shoulder_in_basis_prenorm)
    raise_angle = (np.arctan2(elbow_shoulder_in_basis[2],elbow_shoulder_in_basis[0]) * 180.0/np.pi)
    return raise_angle


# TODO: REMOVE THIS, HERE FOR DEBUG PURPOSES
def get_planar_arm_angle_right(hip_left, hip_right, wrist_right, shoulder_right):
    if (not test_validity([hip_left, hip_right, wrist_right, shoulder_right])):
        return np.nan
    wrist_shoulder = wrist_right - shoulder_right
    hip_vector = _normalize(hip_right - hip_left)
    shoulder_hip_vector = shoulder_right - hip_right
    side_of_body_right = hip_right - shoulder_right

    chest_plane_normal = np.cross(shoulder_hip_vector, hip_vector)
    chest_plane_normal = chest_plane_normal / np.linalg.norm(chest_plane_normal)

    wrist_shoulder_proj_onto_chest = wrist_shoulder - np.dot(wrist_shoulder, chest_plane_normal) * chest_plane_normal
    norm_of_proj = np.linalg.norm(wrist_shoulder_proj_onto_chest)
    wrist_shoulder_proj_onto_chest = wrist_shoulder_proj_onto_chest / np.linalg.norm(wrist_shoulder_proj_onto_chest)

    angle = np.arccos(
        np.dot(wrist_shoulder_proj_onto_chest, side_of_body_right) /
        (np.linalg.norm(side_of_body_right))) * 180.0 / np.pi
    return angle

def get_planar_arm_angle(single_frame_3d_joints, single_frame_2d_joints, side, frame=None, ax=None):
    """
    Public facing function to return the planar arm angle. Decides whether to use
    3D or 2D joints
    Input:
      - single_frame_3d_joints: 3D joints
      - single_frame_2d_joints: 2D joints
      - side: either "L" or "R" depending on the side. Throw exception if neither
      - frame: Used for debug purposes to annotate the camera recording
      - ax: Used for debug purposes to annotate the
    Output:
      - angle: Returns the planar angle of the arm (used to compute
               whether the arm is abducted)
    """

    # Always return 3D for now
    if (side != "L" and side != "R"):
        raise ValueError("Side not valid value")

    return _get_planar_arm_angle3D(single_frame_3d_joints, side, ax), AngleTypeUsed.THREE_D

def _get_lateral_arm_angle3D(joints, side, ax=None):
    """
    Input:
      - joints: 3D joints
      - side: either "L" or "R" depending on the side
      - ax: A reference to a matplotlib 3D plot that's used for debug purposes
    Output:
      - angle: Returns the lateral angle of the arms
    """
    if (joints == {}):
        return np.nan
    if (not test_validity([
            joints[side+"SHOULDER"],
            joints["LSHOULDER"],
            joints["RSHOULDER"],
            joints["RHIP"],
            joints["LHIP"]
        ])):
        return np.nan
    hip_right = joints["RHIP"]
    hip_left = joints["LHIP"]
    elbow = joints[side + "ELBOW"]
    shoulder = joints[side + "SHOULDER"]
    thorax = (joints["LSHOULDER"] + joints["RSHOULDER"])/2
    if (side == "L"):
        other_side = "R"
    else:
        other_side = "L"
    if (not test_validity([hip_right, hip_left, elbow, shoulder, thorax])):
        return np.nan
    hip_vector = _normalize(joints["RHIP"] - joints["LHIP"])
    shoulder_vector = _normalize(joints["RSHOULDER"] - joints["LSHOULDER"])


    hip_vector_2d = np.array([hip_vector[0], hip_vector[2]])
    shoulder_vector_2d = np.array([shoulder_vector[0], shoulder_vector[2]])

    avg_lat_vec = _normalize((hip_vector_2d + shoulder_vector_2d)/2.0)
    avg_lat_vec_3d = _normalize((hip_vector + shoulder_vector)/2.0)

    elbow_shoulder_prenorm = elbow - shoulder
    elbow_shoulder = _normalize(elbow - shoulder)

    elbow_shoulder_onto_hip = elbow_shoulder
    elbow_shoulder_onto_hip_prenorm = elbow_shoulder - np.dot(elbow_shoulder, avg_lat_vec_3d) * avg_lat_vec_3d
    proj_norm = np.linalg.norm(elbow_shoulder_onto_hip_prenorm)
    #print(f"Proj norm: {proj_norm}")
    
    elbow_shoulder_onto_hip = _normalize(elbow_shoulder - np.dot(elbow_shoulder, avg_lat_vec_3d) * avg_lat_vec_3d)
    body_vec = _normalize(joints["PELV"] - thorax)
    body_vec_onto_hip = _normalize(body_vec - np.dot(body_vec, avg_lat_vec_3d) * avg_lat_vec_3d)

    x = body_vec
    z = avg_lat_vec_3d

    y = np.cross(z, x)
   
    basis = np.zeros((3, 3))
    basis[:, 0] = x
    basis[:, 1] = y
    basis[:, 2] = z

    elbow_shoulder_in_basis = np.dot(basis.T, elbow_shoulder)
    elbow_shoulder_in_basis_prenorm = np.dot(basis.T, elbow_shoulder_prenorm)
    pelvis = (hip_right + hip_left) / 2.0
    
    
    # chest_plane = np.cross(body_vec, hip_vector)
    # elbow_shoulder_onto_hip = _normalize(elbow_shoulder - np.dot(elbow_shoulder, hip_vector) * chest_plane)

    if (not (ax is None)):
        p = joints[side + "SHOULDER"]
        p2 = x * 100
        p3 = y * 100
        p4 = z * 100
        ax.quiver(p[0],
                  p[1],
                  p[2],
                  p2[0],
                  p2[1],
                  p2[2],
                  arrow_length_ratio=.01,
                  color='black')
        ax.quiver(p[0], p[1], p[2], p3[0], p3[1], p3[2], arrow_length_ratio=.01, color='green')
        ax.quiver(p[0], p[1], p[2], p4[0], p4[1], p4[2], arrow_length_ratio=.01, color='blue')
    #raise_angle = np.arccos(np.dot(elbow_shoulder_onto_hip, body_vec)) * 180.0/np.pi
    raise_angle = (np.arctan2(elbow_shoulder_in_basis[1],elbow_shoulder_in_basis[0]) * 180.0/np.pi)
    return np.abs(raise_angle)
    #return np.arccos(np.dot(j1, j2)) * 180.0/np.pi


def _get_lateral_arm_angle2D(joints, side, frame=None):
    """
    Same as _get_lateral_arm_angle3D but for 2d
    """
    if (joints == {}):
        return np.nan
    hip = joints[side + "HIP"]
    elbow = joints[side + "ELBOW"]
    shoulder = joints[side + "SHOULDER"]
    if (not test_validity([hip, elbow, shoulder])):
        return np.nan
    elbow_shoulder = _normalize(elbow - shoulder)
    body_vec = _normalize(hip - shoulder)

    raise_angle = np.arccos(np.dot(elbow_shoulder, body_vec)) * 180.0 / np.pi
    return raise_angle


def get_lateral_arm_angle(single_frame_3d_joints, single_frame_2d_joints, orientation, side, frame=None, ax=None):
    """
    This is the public facing function the risk assessment routines use
    Input:
      - single_frame_3d_joints: 3D joints
      - single_frame_2d_joints: 2D joints
      - side: either "L" or "R" depending on the side. Throw exception if neither
      - frame: Used for debug purposes to annotate the camera recording
      - ax: Used for debug purposes to annotate the
    Output:
      - angle: Returns the lateral angle of the arms
    """
    if (side != "L" and side != "R"):
        raise ValueError("Side not valid value")
    armAngle3D = _get_lateral_arm_angle3D(single_frame_3d_joints, side, ax)
    armAngle2D = _get_lateral_arm_angle2D(single_frame_2d_joints, side, frame)
    trunkAngle, angleType = get_trunk_angle(single_frame_3d_joints, single_frame_2d_joints, orientation)

    if np.isnan(armAngle3D) or np.isnan(trunkAngle):
        return armAngle2D, AngleTypeUsed.TWO_D

    # Only return 3D if trunk is sufficiently not bent
    if (trunkAngle < 50):
        return armAngle3D, AngleTypeUsed.THREE_D

    return armAngle2D, AngleTypeUsed.TWO_D


def get_shoulder_raised_ratio(joints, avg_neck_vec_size):
    """
    This function is used to compute a heuristic that we can use
    to tell if a subject's shoulders are raised. This heuristic works
    computing the ratio between the average length of the
    neck to head vector and the distance between the head and
    the shoulders. If the shoulder to head distance is roughly
    similar to the head to neck distance (avg_neck_vec_size), then we say the shoulder is raised.
    The exact parameter that defines the ratio for which we'd say the
    distances are "roughly similar" can be found in assessment_constants.py
    Input:
      - joints: 3D joints
      - avg_neck_vec_size: Average size of neck vector.
      - frame: Used for debug purposes to annotate the
    Output:
      - ratio: Returns the ratio between the distances of the shoulders
               and head and the neck and head
    """
    if (joints == {}):
        return np.nan
    head = joints["HEAD"]
    shoulder_left = joints["LSHOULDER"]
    shoulder_right = joints["RSHOULDER"]
    hip_left = joints["LHIP"]
    hip_right = joints["RHIP"]
    if (not test_validity([head, shoulder_left, shoulder_right, hip_left, hip_right])):
        return np.nan

    dist1 = np.linalg.norm(head - shoulder_left)
    dist2 = np.linalg.norm(head - shoulder_right)
    ratio = (dist1 + dist2) / (2 * avg_neck_vec_size)
    return ratio

def get_neck_vector_length(joints):
    """
    This helper function is used to compute the size of the
    neck vector, which comes in handy in our heuristic for
    telling if the shoulders are raised
    Input:
      - joints: 3D joints
    Output:
      - angle: Returns the lateral angle of the arms
    """
    if (joints == {}):
        return np.nan
    head = joints["HEAD"]
    shoulder_left = joints["LSHOULDER"]
    shoulder_right = joints["RSHOULDER"]
    if (not test_validity([head, shoulder_left, shoulder_right])):
        return np.nan
    neck = (shoulder_left + shoulder_right) / 2.0
    return np.linalg.norm(head - neck)


# TODO: EXPERIMENTAL NOT USED, MAYBE DELETE
def getShoulderRaisedDistExperimental(joints, frame=None):
    if (joints == {}):
        return np.nan, np.nan

    ear_left = joints["LEAR"]
    ear_right = joints["REAR"]
    shoulder_left = joints["LSHOULDER"]
    shoulder_right = joints["RSHOULDER"]
    left_dist = None
    right_dist = None
    if (test_validity([ear_left, shoulder_left])):
        left_dist = np.linalg.norm(ear_left - shoulder_left)
    if (test_validity([ear_right, shoulder_right])):
        right_dist = np.linalg.norm(ear_right - shoulder_right)

    if (not (frame is None)):
        startPoint = joints["LEAR"]
        endPoint = shoulder_left
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint.astype(np.int)), (255, 0, 0), 1)
    return left_dist, right_dist


##########################
####### LOWER ARM ########
##########################

def _get_elbow_angle3D(joints, side, ax=None):
    """
    Input:
      - joints: 3D joints
      - side: Either L or R depending on the side
      - ax: A reference to a matplotlib 3D plot that's used for debug purposes
    Output:
      - angle: Returns the 3D angle of the elbow on the specified side
    """
    if (joints == {}):
        return np.nan
    wrist = joints[side + "WRIST"]
    elbow = joints[side + "ELBOW"]
    shoulder = joints[side + "SHOULDER"]
    if (not test_validity([wrist, elbow, shoulder])):
        return np.nan

    forearm_vector = _normalize(wrist - elbow)
    bicep_vector = _normalize(shoulder - elbow)

    angle = np.arccos(np.dot(forearm_vector, bicep_vector))
    if (not (ax is None)):
        p = joints[side + "ELBOW"]
        p2 = forearm_vector
        ax.quiver(p[0],
                  p[1],
                  p[2],
                  bicep_vector[0],
                  bicep_vector[1],
                  bicep_vector[2],
                  arrow_length_ratio=.01,
                  color='r')
        ax.quiver(p[0], p[1], p[2], p2[0], p2[1], p2[2], arrow_length_ratio=.01, color='r')

    return max((180.0 - (angle * 180.0 / np.pi) - ELBOW_COMPENSATION_FACTOR), 0)

def _get_elbow_angle2D(joints, side, frame=None):
    """
    Input:
      - joints: 2D joints
      - side: Either L or R depending on the side
      - frame: A reference to a matplotlib 3D plot that's used for debug purposes
    Output:
      - angle: Returns the 2D angle of the elbow on the specified side
    """
    if (joints == {}):
        return np.nan
    wrist = joints[side + "WRIST"]
    elbow = joints[side + "ELBOW"]
    shoulder = joints[side + "SHOULDER"]

    if (not test_validity([wrist, elbow, shoulder])):
        return np.nan

    forearm_vector = _normalize(wrist - elbow)
    bicep_vector = _normalize(shoulder - elbow)

    angle = np.arccos(np.dot(forearm_vector, bicep_vector))
    if (not (frame is None)):
        startPoint = elbow
        endPoint = wrist
        endPoint2 = shoulder
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint.astype(np.int)), (0, 0, 255), 2)
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint2.astype(np.int)), (0, 255, 0), 2)
    return 180.0 - (angle * 180.0 / np.pi)


def get_elbow_angle(single_frame_3d_joints, single_frame_2d_joints, orientation, side, frame=None, ax=None):
    """
    Gets elbow angle, falls back to 2D if we think wrnch's angle is not accurate enough
    Input:
      - single_frame_3d_joints: 3D joints
      - single_frame_2d_joints: 2D joints
      - side: either "L" or "R" depending on the side. Throw exception if neither
      - frame: Used for debug purposes to annotate the camera recording
      - ax: Used for debug purposes to annotate the 3d plt
    Output:
      - angle: Returns the angle of the elnow on the specified side
    """
    if (side != "L" and side != "R"):
        raise ValueError("Side not valid value")

    angle2D = _get_elbow_angle2D(single_frame_2d_joints, side, frame)
    angle3D = _get_elbow_angle3D(single_frame_3d_joints, side, ax)
    planarArmAngle, angleType = get_planar_arm_angle(single_frame_3d_joints, single_frame_2d_joints, side)

    if (orientation == Orientation.LEFT_PROFILE or
        orientation == Orientation.RIGHT_PROFILE):
        return angle2D, AngleTypeUsed.TWO_D

    # If the shoulder is bent too high wrnch's angle is not reliable, fallback to 2D
    if ((not (planarArmAngle is None) and planarArmAngle >= 30) or np.isnan(angle3D)):
        return np.nan, AngleTypeUsed.TWO_D
    return angle3D, AngleTypeUsed.THREE_D


def get_lower_arm_midline(joints, side, ax=None):
    """
    Input:
      - joints: 3D joints
      - side: Either L or R depending on the side. Throws error if neither
      - ax: 3D Plot for debug purposes
    Output:
      - angle: Returns the midline angle of the lower arm. Make sure trunk is not bent
               too much when using this value because it can become unreliable
    """
    if (joints == {}):
        return np.nan
    shoulder_left = joints["LSHOULDER"]
    shoulder_right = joints["RSHOULDER"]
    hip_left = joints["LHIP"]
    hip_right = joints["RHIP"]
    wrist = joints[side + "WRIST"]
    elbow = joints[side + "ELBOW"]
    if (not test_validity([shoulder_left, shoulder_right, hip_left, hip_right, wrist, elbow])):
        return np.nan
    pelvis = (hip_left + hip_right) / 2.0
    neck = (shoulder_left + shoulder_right) / 2.0
    body_vec = _normalize(neck - pelvis)
    chest_normal = _normalize(_getChestPlaneHelper(shoulder_left, shoulder_right, hip_left, hip_right))
    arm_vec = _normalize(wrist - elbow)
    arm_vec = arm_vec - np.dot(arm_vec, body_vec) * body_vec

    if (not (ax is None)):
        p = (shoulder_left + shoulder_right + hip_left + hip_right) / 4.0
        p2 = chest_normal
        ax.quiver(p[0], p[1], p[2], arm_vec[0], arm_vec[1], arm_vec[2], arrow_length_ratio=.01, color='r')
        ax.quiver(p[0], p[1], p[2], p2[0], p2[1], p2[2], arrow_length_ratio=.01, color='r')

    return max(np.arccos(np.dot(arm_vec, chest_normal)) * 180.0 / np.pi - LOWER_ARM_MIDLINE_COMPENSATION_FACTOR_3D, 0)



def get_wrist_angle(joints, handEnd, side, frame=None):
    """
    Input:
      - joints: 2D joints
      - handEnd: position of the end of the hand in 2D
      - side: "L" or "R". Throws exception if neither
    Output:
      - Wrist bend angle
    """
    if (side != "L" and side != "R"):
        raise ValueError("Side not valid value")
    if (joints == {}):
        return np.nan
    elbow = joints[side + "ELBOW"]
    wrist = joints[side + "WRIST"]
    end = handEnd[side]
    if (not test_validity([elbow, wrist, end])):
        return np.nan
    vec1 = _normalize(elbow - wrist)
    vec2 = _normalize(end - wrist)

    if (not (frame is None)):
        startPoint = joints[side + "WRIST"]
        endPoint = end
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint.astype(np.int)), (255, 0, 0), 1)
        endPoint = elbow
        frame = cv2.line(frame, tuple(startPoint.astype(np.int)), tuple(endPoint.astype(np.int)), (0, 255, 0), 1)

    return 180 - np.arccos(np.dot(vec1, vec2)) * 180.0 / np.pi


def is_arm_valid(left_arm_planar, right_arm_planar):
    return left_arm_planar < 30 and right_arm_planar < 30


def test_elbow_planar(left_elbow, right_elbow, left_arm_planar, right_arm_planar):
    if (np.isnan(left_elbow) or np.isnan(right_elbow) or np.isnan(left_arm_planar) or np.isnan(right_arm_planar)):
        return -1
    if (not is_arm_valid(left_arm_planar, right_arm_planar)):
        return -1
    if (left_elbow > 100 or right_elbow > 100):
        return 2
    if (left_elbow < 50 or right_elbow < 50):
        return 2
    return 1
