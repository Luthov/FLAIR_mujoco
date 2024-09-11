from .base_controller import load_controller_config
from .operational_space_controller import OperationalSpaceController
from .joint_position_controller import JointPositionController
from .ik_pybullet import InverseKinematicsController
from .differential_ik import DiffIKController


CONTROLLER_INFO = {
    "JOINT_POSITION": "Joint Position",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSE": "Operational Space Control (Position + Orientation)",
    "IK_POSE": "Inverse Kinematics (Position + Orientation)",
    "DIFFIK_POSE": "Differential IK (Position + Orientation)",
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()
