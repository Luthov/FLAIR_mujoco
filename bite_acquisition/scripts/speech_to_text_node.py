#!/usr/bin/env python
import rospy

from std_msgs.msg import String
from speech_to_text.speech_to_text import get_user_preference

def publish_user_preference():
    rospy.init_node('speech_to_text_node', anonymous=True)
    pub_user_preference = rospy.Publisher('user_preference', String, queue_size=1)
    rate = rospy.Rate(10)  # 10hz

    while not rospy.is_shutdown():
        user_preference = get_user_preference()
        rospy.loginfo(f'[Speech-To-Text Node] User preference: {user_preference}')
        pub_user_preference.publish(user_preference)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_user_preference()
    except rospy.ROSInterruptException:
        pass