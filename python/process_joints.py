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
