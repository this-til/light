from __future__ import annotations

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
        
        asyncio.create_task(self.readAheadTestLoop())

    def laserRadarCallback(self, msg):
        self.source.publish_nowait(msg.ranges)
        pass

    async def readAheadTestLoop(self):
        queue = await self.source.subscribe(asyncio.Queue(maxsize=1))
        while True:
            r = await queue.get()
            #self.logger.info(f"0:{r[0]}, 90:{r[90]}, 180:{r[180]}, 270:{r[270]}")
            #r_180 = r[180]
            #if r_180 == 31.0:
            #    continue
            #self.logger.info(f"180:{r_180}")
            
            r_0 = r[0]
            if r_0 == 31.0:
                continue
            self.logger.info(f"0:{r_0}")
        

    def start(self):
        self.roscarMappingLaunchFile.start()

    def stop(self):
        self.roscarMappingLaunchFile.shutdown()
