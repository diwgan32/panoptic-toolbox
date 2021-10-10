import numpy as np

cocoDefs = [
	"NECK",
	"NOSE",
	"PELV",
	"LSHOULDER",
	"LELBOW",
	"LWRIST",
	"LHIP",
	"LKNEE",
	"LANKLE",
	"RSHOULDER",
	"RELBOW",
	"RWRIST",
	"RHIP",
	"RKNEE",
	"RANKLE",
	"LEYE",
	"LEAR",
	"REYE",
	"REAR"
]


def get_joint(data, name):
	i = cocoDefs.index(name)
	if (data[i * 4 + 3] < .1):
		return None
	a = np.array([
		data[i * 4],
		data[i * 4 + 1],
		data[i * 4 + 2]
	])
	return a

def get_all_3d_joints(data):
	joints = {}
	for name in cocoDefs:
		joints[name] = get_joint(data, name)

	return joints

def extract_hand_end_pos(cmu_data):
	# Joint pos's taken from 
	# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#hand-output-format
	if (len(cmu_data) == 0):
		return  {"L": None, "R": None}

	if (not "left_hand" in cmu_data[0]):
		hand_left = None
	else:
		hand_left = cmu_data[0]["left_hand"]["landmarks"]

	if (not "right_hand" in cmu_data[0]):
		hand_right = None
	else:
		hand_right = cmu_data[0]["right_hand"]["landmarks"]

	use_joints = [17, 13, 9, 5, 18, 14, 10, 6, 19, 15, 11, 7, 20, 16, 12, 8]
	#use_joints = [16, 15, 14]
	hand_end = {"L": None, "R": None}
	if (hand_left is None or hand_right is None):
		return  {"L": None, "R": None}
	c = 0
	for i in use_joints:
		if (hand_end["R"] is None):
			hand_end["R"] = np.zeros(3)
		if (hand_end["L"] is None):
			hand_end["L"] = np.zeros(3)

		if (not hand_right is None):
			hand_end["R"] += np.array([hand_right[i * 3], hand_right[i * 3 + 1], hand_right[i * 3 + 2]])
		if (not hand_left is None):
			hand_end["L"] += np.array([hand_left[i * 3], hand_left[i * 3 + 1], hand_left[i * 3 + 2]])
		c += 1
	hand_end["R"] /= c
	hand_end["L"] /= c
	return hand_end