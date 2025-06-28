import util
from main import Component, ConfigField
import rospy
import asyncio
from sensor_msgs.msg import Imu

class ImuComponent(Component):

    debug : ConfigField[bool] = ConfigField()
    
    quaternion : util.Quaternion = util.Quaternion()
    euler : util.V3 = util.V3()
    
    async def init(self):
        await super().init()
        rospy.Subscriber("/unilidar/imu", Imu, self.imuCallback)
        
        if self.debug:
            asyncio.create_task(self.debugLoop())
            
        
    def imuCallback(self, msg : Imu):
        orientation = msg.orientation
        self.quaternion = util.Quaternion(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        self.euler = self.quaternion.toEulerAnglesDegrees()
        
    async def debugLoop(self):
        
        while True:
            await asyncio.sleep(0.1)
            self.logger.info(f"yaw={self.euler.z}")
        pass
        
    def getYaw(self) -> float:
        """获取当前偏航角（度）"""
        return self.euler.z
    
    def getPitch(self) -> float:
        """获取当前俯仰角（度）"""
        return self.euler.y
    
    def getRoll(self) -> float:
        """获取当前横滚角（度）"""
        return self.euler.x
    
    def getEulerAngles(self) -> util.V3:
        """获取欧拉角（度）"""
        return util.V3(self.euler.x, self.euler.y, self.euler.z)
    
    def getQuaternion(self) -> util.Quaternion:
        """获取四元数"""
        return util.Quaternion(self.quaternion.x, self.quaternion.y, self.quaternion.z, self.quaternion.w)
    
    def getYawRadians(self) -> float:
        """获取当前偏航角（弧度）"""
        return util.degreesToRadians(self.euler.z)
    
    def getPitchRadians(self) -> float:
        """获取当前俯仰角（弧度）"""
        return util.degreesToRadians(self.euler.y)
    
    def getRollRadians(self) -> float:
        """获取当前横滚角（弧度）"""
        return util.degreesToRadians(self.euler.x)
        
    pass