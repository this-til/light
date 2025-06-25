import asyncio

import roslaunch
import roslaunch.parent
import roslaunch.rlutil

import rospy
from sensor_msgs.msg import LaserScan
from util import Broadcaster

from main import Component, ConfigField


class LaserRadarComponent(Component):
    roscarMappingLaunchPath: ConfigField[str] = ConfigField()

    roscarMappingLaunchFile = None

    source: Broadcaster[list[int]] = Broadcaster()

    async def awakeInit(self):
        await super().awakeInit()

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)

        self.roscarMappingLaunchFile = roslaunch.parent.ROSLaunchParent(uuid, [self.roscarMappingLaunchPath])

        rospy.Subscriber("/scan", LaserScan, self.laserRadarCallback)

    def laserRadarCallback(self, msg):
        self.source.publish_nowait(msg.ranges)
        pass

    async def readAheadTestLoop(self):
        queue = await self.source.subscribe(asyncio.Queue(maxsize=1))
        while True:
            r = await queue.get()
            self.logger.info(f"前方测距 {r[180]}")
            await asyncio.sleep(0.1)

    def start(self):
        self.roscarMappingLaunchFile.start()

    def stop(self):
        self.roscarMappingLaunchFile.shutdown()
