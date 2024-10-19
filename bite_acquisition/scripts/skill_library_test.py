import rospy
from skill_library_mujoco import SkillLibrary

if __name__ == "__main__":
    rospy.init_node('mujoco_controller')
    keypoints = [[0.3507, 0.0512, 0.0373 + 0.1], [0.4507, -0.0512, 0.0373 + 0.1]]

    skill_library = SkillLibrary()

    skill_library.pushing_skill_mujoco(keypoints)