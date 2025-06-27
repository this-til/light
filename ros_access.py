
import roslaunch
import roslaunch.parent
import roslaunch.rlutil

import rospy

from main import Component, ConfigField

class RosAccessComponent (Component):
    
    baseLaunch : ConfigField[str] = ConfigField()
    mapLaunch : ConfigField[str] = ConfigField()
    
    async def awakeInit(self):
        await super().awakeInit()

        self.baseLaunchFile = roslaunch.parent.ROSLaunchParent(
            roslaunch.rlutil.get_or_generate_uuid(None, False),
            [self.baseLaunch]
        )
        
        self.mapLaunchFile =  roslaunch.parent.ROSLaunchParent(
            roslaunch.rlutil.get_or_generate_uuid(None, False),
            [self.mapLaunch]
        )
        
        self.baseLaunchFile.start()
    
        
    def getPriority(self) -> int:
        return 1 << 24
    
    pass