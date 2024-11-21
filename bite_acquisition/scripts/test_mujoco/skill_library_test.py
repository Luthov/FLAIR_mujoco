import rospy
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import PoseStamped

import sys
sys.path.insert(0, '/home/luthov_ubuntu/School/FYP/feeding_ws/src/feeding/task_planner/src/FLAIR_mujoco/bite_acquisition/scripts')
from skill_library_mujoco import SkillLibrary

if __name__ == "__main__":
    rospy.init_node('mujoco_controller')

    # r = Rotation.from_euler('xyz', [-180, -90, 0], degrees=True)
    # quat = r.as_quat()

    # transfer_pose = PoseStamped()
    # transfer_pose.pose.position.x = 0.4
    # transfer_pose.pose.position.y = 0.0
    # transfer_pose.pose.position.z = 0.4

    # transfer_pose.pose.orientation.x = quat[0]
    # transfer_pose.pose.orientation.y = quat[1]
    # transfer_pose.pose.orientation.z = quat[2]
    # transfer_pose.pose.orientation.w = quat[3]

    push_keypoints = [[0.3507, 0.0512, 0.0373 + 0.1], [0.4507, -0.0512, 0.0373 + 0.1]]
    cut_keypoints = [0.3507, 0.0512, 0.0373 + 0.1]
    scoop_keypoint_1 = [0.55, 0.0, 0.05]
    scoop_keypoint_2 = [0.45, 0.13, 0.05]
    scoop_rice_keypoint = [0.4, -0.25, 0.025 + 0.05]
    scoop_chicken_keypoint = [0.4, 0, 0.025 + 0.05]
    scoop_egg_keypoint = [0.4, 0.25, 0.025 + 0.05]

    skill_library = SkillLibrary()

    # skill_library.pushing_skill_mujoco(push_keypoints)
    # skill_library.cutting_skill_mujoco(cut_keypoints, cutting_angle=np.pi/4)

    
    # Bite size range is -1.0 (smallest) to 1.0 (biggest)
    # skill_library.scooping_skill_mujoco(scoop_keypoint_1, bite_size=0.0)
    # TODO: Luke: ideally minimize rotation of gripper acquiring food during bite transfer
    # can try using cartesian path planning: https://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html#cartesian-paths
    # skill_library.scooping_skill_mujoco(scoop_rice_keypoint, bite_size=0.0)
    # skill_library.transfer_to_mouth(transfer_pose)

    # skill_library.scooping_skill_mujoco(scoop_rice_keypoint, bite_size=1.0)
    # skill_library.transfer_to_mouth(transfer_pose)

    # skill_library.scooping_skill_mujoco(scoop_rice_keypoint, bite_size=-1.0)
    # skill_library.transfer_to_mouth(transfer_pose)

    bite_size = 0.0
    while True:
        input(f"bite_size={bite_size}")
        skill_library.scooping_skill_mujoco(scoop_chicken_keypoint, bite_size=bite_size)
        if bite_size == 1.0:
            break
        bite_size += 0.2
    # skill_library.transfer_to_mouth()

    # skill_library.scooping_skill_mujoco(scoop_egg_keypoint, bite_size=0.0)
    # skill_library.transfer_to_mouth()

    # skill_library.scooping_skill_mujoco(scoop_keypoint_2, bite_size=1.0)
    # skill_library.transfer_to_mouth()
    # mouth_pose = np.array([0.70, 0.0, 0.545])

    # transfer_pose = PoseStamped()
    # transfer_pose.pose.position.x = mouth_pose[0]
    # transfer_pose.pose.position.y = mouth_pose[1]
    # transfer_pose.pose.position.z = mouth_pose[2]

    # input("Transfer to mouth")
    # skill_library.transfer_to_mouth(transfer_pose)

