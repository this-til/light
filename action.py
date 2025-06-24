from __future__ import annotations

import asyncio

import actionlib
import cv2
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

import util
from main import Component


class ActionComponent(Component):
    actionClient = None

    async def awakeInit(self):
        await super().awakeInit()

        self.actionClient = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.actionClient.wait_for_server()

        await asyncio.create_task(self.instructionLoop())

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
        self.logger.info("开始导航")

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

    async def searchForTheTarget(self):
        queue: asyncio.Queue[cv2.typing.MatLike] = await self.main.orbbecCameraComponent.source.subscribe(
            asyncio.Queue(maxsize=1))

        self.main.motionComponent.disableSpeedAttenuation()

        try:
            self.main.motionComponent.setVelocity(linear_x=0.3)

            

        finally:
            self.main.motionComponent.enableSpeedAttenuation()
            await self.main.orbbecCameraComponent.source.unsubscribe(queue)

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

        await self.calibration()

        await self.main.exclusiveServerReportComponent.openRollingDoor()

        # TODO 前进到基站

        await self.main.exclusiveServerReportComponent.setRollingDoor(False)

        pass

    async def calibration(self):
        await self.main.exclusiveServerReportComponent.openRollingDoor()

        queue: asyncio.Queue[cv2.typing.MatLike] = await self.main.orbbecCameraComponent.source.subscribe(
            asyncio.Queue(maxsize=1))

        try:
            max_iterations = 20  # 最大调整次数
            tolerance = 10  # 像素容差

            for iteration in range(max_iterations):
                mat: cv2.typing.MatLike = await queue.get()
                crosshair: (float, float) | None = await asyncio.get_event_loop().run_in_executor(
                    None, util.findCrosshair, mat
                )

                if crosshair is None:
                    self.logger.warning(f"校准第{iteration + 1}次: 未找到十字准星")
                    await asyncio.sleep(0.5)
                    continue

                # 获取图像中心位置
                image_height, image_width = mat.shape[:2]
                center_x = image_width / 2
                center_y = image_height / 2

                # 计算X轴偏差（只关注垂直于车身的标识）
                crosshair_x, crosshair_y = crosshair
                offset_x = crosshair_x - center_x

                self.logger.info(f"校准第{iteration + 1}次: 十字准星X位置{crosshair_x:.1f}, "
                                 f"图像中心X位置{center_x:.1f}, "
                                 f"X轴偏差{offset_x:.1f}")

                # 检查X轴是否已经足够接近中心
                if abs(offset_x) <= tolerance:
                    self.logger.info("校准完成：十字准星X轴已接近图像中心")
                    break

                # 根据X轴偏差调整车辆位置
                await self._adjustVehiclePosition(offset_x)

                # 等待运动完成
                await asyncio.sleep(1.0)

            else:
                self.logger.error(f"校准未能在{max_iterations}次迭代内完成")
                raise Exception(f"校准失败：在{max_iterations}次迭代内未能完成十字准星对中")

        finally:
            await self.main.orbbecCameraComponent.source.unsubscribe(queue)

    async def _adjustVehiclePosition(self, offset_x: float):
        """
        根据十字准星X轴偏差调整车辆位置（只调整垂直于车身的方向）
        :param offset_x: X轴偏差（正值表示十字准星在图像右侧）
        """
        # 校准时使用较小的速度，提高精度
        calibration_speed = 0.05  # 校准速度范围：0.03-0.1

        # 根据偏差大小动态调整速度
        if abs(offset_x) > 100:
            calibration_speed = 0.08  # 偏差大时使用较快速度
        elif abs(offset_x) > 50:
            calibration_speed = 0.05  # 中等偏差使用中等速度
        else:
            calibration_speed = 0.03  # 偏差小时使用最慢速度

        # 计算移动时间（基于偏差大小）
        base_time = 0.3  # 基础移动时间
        move_time = max(0.1, min(1.0, abs(offset_x) / 100 * base_time))

        self.logger.info(
            f"调整车辆位置: X轴偏差{offset_x:.1f}像素, 使用速度{calibration_speed:.2f}m/s, 移动时间{move_time:.2f}s")

        # 使用运动控制调整位置
        if hasattr(self.main, 'motionComponent'):
            # 先停止当前运动
            self.main.motionComponent.stopMotion()

            # 根据X轴偏差方向移动
            if abs(offset_x) > 5:  # 只有偏差大于5像素才移动
                if offset_x > 0:
                    # 十字准星在右侧，车需要向右移动（负Y方向）
                    self.main.motionComponent.setVelocity(linear_y=-calibration_speed)
                else:
                    # 十字准星在左侧，车需要向左移动（正Y方向）
                    self.main.motionComponent.setVelocity(linear_y=calibration_speed)

                self.main.motionComponent.publishVelocity()
                await asyncio.sleep(move_time)
                self.main.motionComponent.stopMotion()
                self.main.motionComponent.publishVelocity()
        else:
            self.logger.error("运动控制组件不可用，无法调整车辆位置")
            raise Exception("运动控制组件不可用，校准失败")

    async def instructionLoop(self):
        queue = await self.main.KeyComponent.keyEvent.subscribe(asyncio.Queue(maxsize=1))

        while True:

            try:
                key = await queue.get()

                if key == "open":
                    await self.main.exclusiveServerReportComponent.openRollingDoor()

                if key == "close":
                    await self.main.exclusiveServerReportComponent.closeRollingDoor()

                if key == "calibration":
                    await self.calibration()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"instructionLoop Exception: {str(e)}")
