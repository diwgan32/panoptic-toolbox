import numpy as np

if __name__ == "__main__":
	ground_truth_fn = '../171204_pose1_groundtruth.npy'
	estimate_fn = '../12_estimations.npy'

	ground_truth = np.load(ground_truth_fn)
	estimate = np.load(estimate_fn)
	joint_mapping = {
		"TRUNK": 0,
		"NECK": 1,
		"LSHOULDER": 2,
		"RSHOULDER": 3,
		"LELBOW": 4,
		"RELBOW": 5
	}
	
	for joint in list(joint_mapping.keys()):
		joint_num = joint_mapping[joint]
		num_nan = np.sum(np.isnan(estimate[:27886, joint_num]))
		mae_trunk = \
			np.sum(
				np.abs(
					np.nan_to_num(
						ground_truth[:, joint_num] - estimate[:27886, joint_num], nan=0.0
					)
				)
			) / ground_truth.shape[0]
				
		print(f"{joint} percent missing: {int((num_nan/27886.0) * 100)}%")
		print(f"{joint} error: {mae_trunk}")
		print("")