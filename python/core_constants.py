from enum import IntEnum


class AngleTypeUsed(IntEnum):
    TWO_D = 1
    THREE_D = 2
    NONE = 3

class Orientation(IntEnum):
	FRONT = 1
	LEFT_PROFILE = 2
	RIGHT_PROFILE = 3
	BACK = 4
	UNKNOWN = 5


NECK_COMPENSATION_FACTOR_3D = 35
NECK_COMPENSATION_FACTOR_2D = 12
LOWER_ARM_MIDLINE_COMPENSATION_FACTOR_3D = 30
ELBOW_COMPENSATION_FACTOR = 0
LEG_COMPENSATION_FACTOR = 10

HIP_TWIST_FACTOR = .006

# The max # of consecutive NaNs we will interpolate data with. Rest will be zero-ed out
MAX_NAN_WINDOW = 9
# The moving average window
SMOOTHING_WINDOW = 5
