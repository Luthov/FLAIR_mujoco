import rospy
import numpy as np
from skill_library_mujoco import SkillLibrary

if __name__ == "__main__":
    rospy.init_node('mujoco_controller')
    push_keypoints = [[0.3507, 0.0512, 0.0373 + 0.1], [0.4507, -0.0512, 0.0373 + 0.1]]
    cut_keypoints = [0.3507, 0.0512, 0.0373 + 0.1]
    skill_library = SkillLibrary()

    # skill_library.pushing_skill_mujoco(push_keypoints)
    skill_library.cutting_skill_mujoco(cut_keypoints, cutting_angle=np.pi/4)