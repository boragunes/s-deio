from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 4096

# bias patch selection towards high gradient regions?
_C.CENTROID_SEL_STRAT = "RANDOM"

# DEVO specific?
_C.GRADIENT_BIAS = False
# Select between random, gradient, scorer
_C.PATCH_SELECTOR = "scorer"
# Eval mode of patch selector (random, topk, multinomial)
_C.SCORER_EVAL_MODE = "multi"
_C.SCORER_EVAL_USE_GRID = True
# Normalizer (only evs): norm, standard
_C.NORM = "std"

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = "DAMPED_LINEAR"
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

# Loop closure
_C.LOOP_CLOSURE = False
_C.BACKEND_THRESH = 64.0
_C.MAX_EDGE_AGE = 1000
_C.GLOBAL_OPT_FREQ = 15

# Classic loop closure
_C.CLASSIC_LOOP_CLOSURE = False
_C.LOOP_CLOSE_WINDOW_SIZE = 3
_C.LOOP_RETR_THRESH = 0.04

_C.VI_INIT_VAR_G = 2
_C.VI_INIT_NORM = 0
_C.VI_WARM_UP_N = 6
_C.VI_WARM_UP_T = 0

# IMU loosening / reliability controls
_C.IMU_GAP_THRESHOLD = 0.025  # seconds before loosening due to sparse IMU
_C.IMU_LOOSE_WITH_LOW_MOTION = False
_C.IMU_LOOSE_LOW_VEL_THRESH = 0.01  # m/s threshold to consider "stuck"
_C.IMU_LOOSE_LOW_VEL_DURATION = 0.1  # seconds of sustained low motion
_C.IMU_LOOSE_LOW_VEL_EXIT_RATIO = 2.0  # hysteresis multiplier for exit speed
_C.IMU_LOOSE_WITH_LOW_ACCEL = False
_C.IMU_LOOSE_LOW_ACCEL_THRESH = 2.0  # m/s^2 threshold to flag constant velocity
_C.IMU_LOOSE_LOW_ACCEL_DURATION = 0.01
_C.IMU_LOOSE_LOW_ACCEL_EXIT_RATIO = 2.0
_C.IMU_LOOSE_LOW_ACCEL_MIN_SPEED = 0.7  # only consider when platform is moving

# patch depth logging (disabled by default)
_C.LOG_PATCH_DEPTHS = False
_C.PATCH_DEPTH_LOG_FILE = ""
_C.PATCH_DEPTH_FILTER_IDS = []
_C.PATCH_DEPTH_FILTER_PIXELS = []
_C.PATCH_DEPTH_PIXEL_TOL = 2.0

cfg = _C
