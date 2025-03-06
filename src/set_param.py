#!/usr/bin/env python3

import rospy
from mavros_msgs.srv import ParamSet
from mavros_msgs.msg import ParamValue

def set_param(param_id, int_value, real_value):
    rospy.wait_for_service('/mavros/param/set')
    try:
        set_service = rospy.ServiceProxy('/mavros/param/set', ParamSet)
        param_value = ParamValue(integer=int_value, real=real_value)
        response = set_service(param_id, param_value)
        if response.success:
            rospy.loginfo(f"Successfully set {param_id} to {int_value}")
        else:
            rospy.logwarn(f"Failed to set {param_id}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    rospy.init_node('set_param_node', anonymous=True)
    set_param("COM_OBS_AVOID", 0, 0.0)
