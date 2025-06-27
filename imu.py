import util
from main import Component, ConfigField
import rospy
from sensor_msgs.msg import Imu

class ImuComponent(Component):
    quaternion : util.Quaternion = util.Quaternion()
    euler : util.V3 = util.V3()
    
    async def init(self):
        await super().init()
        rospy.Subscriber("/wit/imu", Imu, self.imuCallback)
        
    def imuCallback(self, msg : Imu):
        orientation = msg.orientation
        self.quaternion = util.Quaternion(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        self.euler = self.quaternion.toEulerAnglesDegrees()
        pass 
    
    pass