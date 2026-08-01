"""
Collection of constants for cameras / robots / etc
in kitchen environments
"""

# default free cameras for different kitchen layouts
LAYOUT_CAMS = {
    0: dict(
        lookat=[2.26593463, -1.00037131, 1.38769295],
        distance=3.0505089839567323,
        azimuth=90.71563812375285,
        elevation=-12.63948837207208,
    ),
    1: dict(
        lookat=[2.66147999, -1.00162429, 1.2425155],
        distance=3.7958766287746255,
        azimuth=89.75784013699234,
        elevation=-15.177406642875091,
    ),
    2: dict(
        lookat=[3.02344359, -1.48874618, 1.2412914],
        distance=3.6684844368165512,
        azimuth=51.67880851867874,
        elevation=-13.302619131542388,
    ),
    4: dict(
        lookat=[1.6, -1.0, 1.0],
        distance=5,
        azimuth=89.70301806083651,
        elevation=-18.02177994296577,
    ),
}

DEFAULT_LAYOUT_CAM = {
    "lookat": [2.25, -1, 1.05312667],
    "distance": 5,
    "azimuth": 89.70301806083651,
    "elevation": -18.02177994296577,
}

CAM_CONFIGS = dict(
    DEFAULT=dict(
        robot0_frontview=dict(
            pos=[-0.50, 0, 0.95],
            quat=[
                0.6088936924934387,
                0.3814677894115448,
                -0.3673907518386841,
                -0.5905545353889465,
            ],
            camera_attribs=dict(fovy="60"),
            parent_body="mobilebase0_support",
        ),
        # This is the camera that is used for the robot's "left view" in the experiment
        robot0_leftview=dict(
            pos=[0.4, 0.55, 0.5],  # x -> robot forward
            quat=[0.0, 0.0, 0.7, 1.0],  # good angle
            camera_attribs=dict(fovy="85"),
            parent_body="mobilebase0_support",
        ),
        robot0_rightview=dict(
            pos=[0.4, -0.55, 0.5],  # x -> robot forward
            quat=[0.0, 0.0, -0.7, 1.0],  # good angle
            camera_attribs=dict(fovy="85"),
            parent_body="mobilebase0_support",
        ),
    ),
    ### Add robot specific configs here ####
    PandaMobile=dict(),
    GR1FixedLowerBody=dict(),
)


def deep_update(d, u):
    """
    Copied from https://stackoverflow.com/a/3233356
    """
    import collections

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_robot_cam_configs(robot):
    from copy import deepcopy

    default_configs = deepcopy(CAM_CONFIGS["DEFAULT"])
    robot_specific_configs = deepcopy(CAM_CONFIGS.get(robot, {}))
    return deep_update(default_configs, robot_specific_configs)
