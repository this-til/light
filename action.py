from __future__ import annotations

import asyncio

import actionlib
import cv2
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from main import Component


class ActionComponent(Component):
    actionClient = None

    async def awakeInit(self):
        await super().awakeInit()

        self.actionClient = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.actionClient.wait_for_server()

    async def actionNav(self, x_axle=1, y_axle=0, x=0, y=0, z=0, w=1.0):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        # X轴
        goal.target_pose.pose.position.x = x_axle
        # Y轴
        goal.target_pose.pose.position.y = y_axle
        # Z轴
        goal.target_pose.pose.position.z = 0.0

        # 朝向
        goal.target_pose.pose.orientation.x = x
        goal.target_pose.pose.orientation.y = y
        goal.target_pose.pose.orientation.z = z
        goal.target_pose.pose.orientation.w = w
        self.actionClient.send_goal(goal)
        rospy.loginfo("开始导航")

        # self.actionClient.wait_for_result()
        await asyncio.get_event_loop().run_in_executor(None, self.actionClient.wait_for_result, ())

        return self.actionClient.get_state()

    async def exitCabin(self):
        '''
        让小车出舱
        :return:
        '''

        await self.main.exclusiveServerReportComponent.openRollingDoor()

        self.main.laserRadarComponent.start()

        completed: bool = False

        for retryCount in range(3):
            if await self.actionNav(1, 0, 0, 0, 0, 1) == actionlib.GoalStatus.SUCCEEDED:
                completed = True
                break

        if not completed:
            raise Exception("the exitCabin is failed")

        await self.main.exclusiveServerReportComponent.setRollingDoor(False)

        pass

    async def returnVoyage(self):
        '''
        执行反航操作
        '''

        pass

    async def inCabin(self):
        '''
        在舱门前矫正进入
        :return:
        '''

        await self.main.exclusiveServerReportComponent.openRollingDoor()

        queue: asyncio.Queue[cv2.typing.MatLike] = await self.main.orbbecCameraComponent.source.subscribe(
            asyncio.Queue(maxsize=1))

        try:
            mat: cv2.typing.MatLike = await  queue.get()



        finally:
            await self.main.orbbecCameraComponent.source.unsubscribe(queue)

        await self.main.exclusiveServerReportComponent.setRollingDoor(False)

        pass
