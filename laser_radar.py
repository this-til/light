import roslaunch
import roslaunch.parent
import roslaunch.rlutil

from main import Component, ConfigField


class LaserRadarComponent(Component):
    roscarMappingLaunchPath: ConfigField[str] = ConfigField()

    roscarMappingLaunchFile = None

    async def awakeInit(self):
        await super().awakeInit()

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)

        self.roscarMappingLaunchFile = roslaunch.parent.ROSLaunchParent(uuid, [self.roscarMappingLaunchPath])

    def start(self):
        self.roscarMappingLaunchFile.start()

    def stop(self):
        self.roscarMappingLaunchFile.shutdown()