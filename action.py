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

        asyncio.create_task(self.instructionLoop())

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
            self.main.motionComponent.setVelocity(linear_x=0.2)



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


    async def getCrosshairPosition(self, queue: asyncio.Queue[cv2.typing.MatLike]) -> (float, float) | None:
        """
        获取十字准星位置，进行3次检测并取平均值以提高准确性
        :param queue: 图像队列
        :return: 十字准星位置 (x, y) 或 None
        """
        valid_positions = []
        
        for attempt in range(3):
            try:
                mat: cv2.typing.MatLike = await queue.get()
                crosshair: (float, float) | None = await asyncio.get_event_loop().run_in_executor(
                    None, util.findCrosshair, mat
                )
                
                if crosshair is not None:
                    valid_positions.append(crosshair)
                    self.logger.debug(f"第{attempt + 1}次检测: 十字准星位置({crosshair[0]:.1f}, {crosshair[1]:.1f})")
                else:
                    self.logger.debug(f"第{attempt + 1}次检测: 未找到十字准星")
                
                # 短暂延迟以获取不同的帧
                if attempt < 2:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.warning(f"第{attempt + 1}次检测异常: {str(e)}")
        
        if not valid_positions:
            self.logger.warning("3次检测均未找到十字准星")
            return None
        
        # 计算平均位置
        avg_x = sum(pos[0] for pos in valid_positions) / len(valid_positions)
        avg_y = sum(pos[1] for pos in valid_positions) / len(valid_positions)
        
        self.logger.info(f"位置检测完成: 有效检测{len(valid_positions)}次, 平均位置({avg_x:.1f}, {avg_y:.1f})")
        
        return (avg_x, avg_y)


    async def calibration(self):
        await self.main.exclusiveServerReportComponent.openRollingDoor()

        self.main.motionComponent.disableSpeedAttenuation()

        queue: asyncio.Queue[cv2.typing.MatLike] = await self.main.orbbecCameraComponent.source.subscribe(
            asyncio.Queue(maxsize=1))

        try:
            max_iterations = 40  # 最大调整次数
            tolerance = 3  # 像素容差

            for iteration in range(max_iterations):
                crosshair: (float, float) | None = await self.getCrosshairPosition(queue)

                if crosshair is None:
                    self.logger.warning(f"校准第{iteration + 1}次: 未找到十字准星")
                    await asyncio.sleep(0.5)
                    continue

                center_x = 350  # 图像中心320 物理中心360

                # 计算X轴偏差（只关注垂直于车身的标识）
                crosshair_x, crosshair_y = crosshair
                offset_x = crosshair_x - center_x

                self.logger.info(f"校准第{iteration + 1}次: 十字准星X位置{crosshair_x:.1f}, X轴偏差{offset_x:.1f}")

                # 检查X轴是否已经足够接近中心
                if abs(offset_x) <= tolerance:
                    self.logger.info("校准完成：十字准星X轴已接近图像中心")
                    break

                # 根据X轴偏差调整车辆位置
                await self._adjustVehiclePosition(offset_x)

            else:
                raise Exception(f"校准失败：在{max_iterations}次迭代内未能完成十字准星对中")

        finally:
            self.main.motionComponent.enableSpeedAttenuation()
            await self.main.orbbecCameraComponent.source.unsubscribe(queue)
            
    async def _adjustVehiclePosition(self, offset_x: float):
        """
        根据十字准星X轴偏差调整车辆位置（只调整垂直于车身的方向）
        :param offset_x: X轴偏差（正值表示十字准星在图像右侧）
        """

        if abs(offset_x) > 100:
            calibration_speed = 0.03  # 偏差大时使用较快速度
        elif abs(offset_x) > 50:
            calibration_speed = 0.04  # 中等偏差使用中等速度
        else:
            calibration_speed = 0.05  # 偏差小时使用最慢速度

        self.logger.info(f"调整车辆位置: X轴偏差{offset_x:.1f}像素, 使用速度{calibration_speed:.2f}")

        # 先停止当前运动
        self.main.motionComponent.stopMotion()

        # 根据X轴偏差方向移动
        if abs(offset_x) > 3:  # 只有偏差大于5像素才移动
            if offset_x > 0:
                # 十字准星在右侧，车需要向右移动（负Y方向）
                self.main.motionComponent.setVelocity(linear_y=-calibration_speed)
            else:
                # 十字准星在左侧，车需要向左移动（正Y方向）
                self.main.motionComponent.setVelocity(linear_y=calibration_speed)
            await asyncio.sleep(0.2)
            self.main.motionComponent.stopMotion()

    async def calibrationByAngle(self):
        """
        选择调整姿势函数
        参照calibration实现，只通过旋转z轴来让十字准星落在中心
        """
        await self.main.exclusiveServerReportComponent.openRollingDoor()

        self.main.motionComponent.disableSpeedAttenuation()

        queue: asyncio.Queue[cv2.typing.MatLike] = await self.main.orbbecCameraComponent.source.subscribe(
            asyncio.Queue(maxsize=1))

        try:
            max_iterations = 40  # 最大调整次数
            center_tolerance = 3  # 中心容差（像素）
            
            self.logger.info("开始姿势调整（仅旋转z轴）...")

            for iteration in range(max_iterations):
                crosshair: (float, float) | None = await self.getCrosshairPosition(queue)

                if crosshair is None:
                    self.logger.warning(f"姿势调整第{iteration + 1}次: 未找到十字准星")
                    await asyncio.sleep(0.5)
                    continue

                # 图像中心
                center_x = 350  # 图像中心320 物理中心360
        
                crosshair_x, crosshair_y = crosshair
                offset_x = crosshair_x - center_x
                
                self.logger.info(f"角度校准第{iteration + 1}次: 十字准星X位置{crosshair_x:.1f}, X轴偏差{offset_x:.1f}")
                
                # 检查是否已经足够接近中心
                if abs(offset_x) <= center_tolerance:
                    self.logger.info("姿势调整完成：十字准星已接近图像中心")
                    break

                # 计算需要旋转的角度方向
                await self._rotateToCenter(offset_x)

                # 等待运动完成
                await asyncio.sleep(1.0)

            else:
                raise Exception(f"姿势调整失败：在{max_iterations}次迭代内未能将十字准星调整到中心")

        finally:
            self.main.motionComponent.enableSpeedAttenuation()
            await self.main.orbbecCameraComponent.source.unsubscribe(queue)

    async def _rotateToCenter(self, offset_x: float):
        """
        通过旋转z轴让十字准星移向中心
        参照calibration的调整模式，但只使用角度调整
        :param crosshair: 十字准星位置
        :param center: 图像中心位置
        """
        if abs(offset_x) > 100:
            calibration_speed = 0.5  # 偏差大时使用较快速度
        elif abs(offset_x) > 50:
            calibration_speed = 0.4  # 中等偏差使用中等速度
        else:
            calibration_speed = 0.3  # 偏差小时使用最慢速度
        
        self.logger.info(f"调整车辆位置: X轴偏差{offset_x:.1f}像素, 使用角速度{calibration_speed:.2f}")

        self.main.motionComponent.stopMotion()
        if abs(offset_x) > 3:
            if offset_x > 0: 
                self.main.motionComponent.setVelocity(angular_z=-calibration_speed)
            else:
                self.main.motionComponent.setVelocity(angular_z=calibration_speed)
            await asyncio.sleep(0.2)
            self.main.motionComponent.stopMotion()


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

                if key == "calibrationByAngle":
                    await self.calibrationByAngle()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"instructionLoop Exception: {str(e)}")
