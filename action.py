from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Tuple, Optional

import actionlib
import cv2
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

import util
from main import Component

import numpy as np


@dataclass
class DepthResult:
    """深度检测结果数据类"""
    valid: bool
    average_distance: Optional[float] = None
    min_distance: Optional[float] = None
    max_distance: Optional[float] = None
    valid_pixels: int = 0
    total_pixels: int = 0
    region_shape: Optional[Tuple[int, int]] = None
    coverage_ratio: float = 0.0

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

    async def searchForTheTarget(self, timeout: float = 30.0):
        """
        搜索目标，带超时机制
        :param timeout: 搜索超时时间（秒），默认30秒
        """
        queue: asyncio.Queue[
            cv2.typing.MatLike] = await self.main.orbbecCameraComponent.brightnessNormalizationSource.subscribe(
            asyncio.Queue(maxsize=1))

        self.main.motionComponent.disableSpeedAttenuation()

        self.logger.info(f"开始搜索目标，超时时间: {timeout}秒")
        start_time = asyncio.get_event_loop().time()

        try:
            # 开始前进搜索
            self.main.motionComponent.setVelocity(linear_x=0.2)

            # 搜索循环，带超时检查
            while True:
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - start_time

                # 检查是否超时
                if elapsed_time >= timeout:
                    self.logger.warning(f"搜索目标超时: {timeout}秒内未找到目标")
                    raise Exception(f"搜索目标超时: {timeout}秒内未找到目标")

                # 尝试获取十字准星位置
                try:
                    # 设置较短的超时时间来获取位置，避免阻塞太久
                    crosshair = await asyncio.wait_for(
                        self.getCrosshairPosition(queue),
                        timeout=2.0
                    )

                    if crosshair is not None:
                        self.logger.info(f"找到目标！十字准星位置: ({crosshair[0]:.1f}, {crosshair[1]:.1f})")
                        self.logger.info(f"搜索完成，耗时: {elapsed_time:.1f}秒")
                        return crosshair  # 成功找到目标

                except asyncio.TimeoutError:
                    # 获取位置超时，继续搜索
                    self.logger.debug(f"获取位置超时，继续搜索... 已搜索{elapsed_time:.1f}秒")

                # 短暂延迟后继续搜索
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            self.logger.info("搜索目标被取消")
            raise
        except Exception as e:
            self.logger.error(f"搜索目标异常: {str(e)}")
            raise
        finally:
            # 停止运动
            self.main.motionComponent.stopMotion()
            self.main.motionComponent.enableSpeedAttenuation()
            await self.main.orbbecCameraComponent.brightnessNormalizationSource.unsubscribe(queue)
            self.logger.info("搜索目标结束，已停止运动")

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

        queue: asyncio.Queue[
            cv2.typing.MatLike] = await self.main.orbbecCameraComponent.brightnessNormalizationSource.subscribe(
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
            await self.main.orbbecCameraComponent.brightnessNormalizationSource.unsubscribe(queue)

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

        queue: asyncio.Queue[
            cv2.typing.MatLike] = await self.main.orbbecCameraComponent.brightnessNormalizationSource.subscribe(
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
            await self.main.orbbecCameraComponent.brightnessNormalizationSource.unsubscribe(queue)

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

    def depthImageCalculateDistance(self, image: cv2.typing.MatLike) -> DepthResult:
        """
        计算深度图像中心区域的平均距离
        :param image: 深度图像
        :return: 包含距离信息的DepthResult对象
        """
        region_size = 0.2  # 20% 的中心区域

        height, width = image.shape
        
        # 计算中心区域的边界
        x_start = int(width * (0.5 - region_size / 2))
        x_end = int(width * (0.5 + region_size / 2))
        y_start = int(height * (0.5 - region_size / 2))
        y_end = int(height * (0.5 + region_size / 2))

        # 提取中心区域
        center_region = image[y_start:y_end, x_start:x_end]

        # 创建有效深度值的掩码（排除0和无穷大值）
        valid_mask = (center_region > 0) & (center_region < np.inf) & ~np.isnan(center_region)

        # 初始化结果对象
        result = DepthResult(
            valid=False,
            total_pixels=center_region.size,
            region_shape=center_region.shape
        )

        if np.any(valid_mask):
            valid_depths = center_region[valid_mask]
            average_depth = np.mean(valid_depths)
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            valid_pixel_count = np.sum(valid_mask)
            
            # 转换深度值到米（如果需要的话，深度值可能是毫米）
            # 假设深度值已经是毫米，转换为米
            if average_depth > 100:  # 如果深度值很大，可能是毫米单位
                average_depth = average_depth / 1000.0
                min_depth = min_depth / 1000.0
                max_depth = max_depth / 1000.0
            
            # 更新结果对象
            result.valid = True
            result.average_distance = float(average_depth)
            result.min_distance = float(min_depth)
            result.max_distance = float(max_depth)
            result.valid_pixels = int(valid_pixel_count)
            result.coverage_ratio = float(valid_pixel_count / center_region.size)
            
            self.logger.info(f"深度统计 - 区域大小: {center_region.shape}, "
                           f"有效像素: {valid_pixel_count}/{center_region.size} ({result.coverage_ratio:.1%}), "
                           f"平均距离: {average_depth:.3f}m, "
                           f"最近距离: {min_depth:.3f}m, "
                           f"最远距离: {max_depth:.3f}m")
        else:
            self.logger.warning(f"中心区域({center_region.shape})未检测到有效深度数据")
            
        return result

    async def testDepthDistance(self, duration: float = 10.0):
        """
        测试深度距离计算，输出指定时间内的图像距离
        :param duration: 测试持续时间（秒），默认10秒
        """
        self.logger.info(f"开始深度距离测试，持续时间: {duration}秒")
        
        # 订阅深度图像数据流
        depth_queue: asyncio.Queue[cv2.typing.MatLike] = await self.main.orbbecCameraComponent.depth.subscribe(
            asyncio.Queue(maxsize=1))
        
        start_time = asyncio.get_event_loop().time()
        measurement_count = 0
        valid_measurements = []  # 存储有效的测量结果
        
        try:
            while True:
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - start_time
                
                # 检查是否已经达到测试时间
                if elapsed_time >= duration:
                    self.logger.info(f"深度距离测试完成，总计测量{measurement_count}次，耗时{elapsed_time:.1f}秒")
                    break
                
                try:
                    # 获取深度图像，设置2秒超时
                    depth_image = await asyncio.wait_for(depth_queue.get(), timeout=2.0)
                    
                    if depth_image is not None:
                        measurement_count += 1
                        self.logger.info(f"第{measurement_count}次测量 (时间: {elapsed_time:.1f}s):")
                        
                        # 调用深度距离计算函数并使用返回值
                        depth_result = await asyncio.get_event_loop().run_in_executor(
                            None, self.depthImageCalculateDistance, depth_image
                        )
                        
                        # 使用返回的深度信息
                        if depth_result.valid:
                            # 保存有效测量结果用于统计
                            valid_measurements.append(depth_result)
                            
                            self.logger.info(f"测量结果 - 平均距离: {depth_result.average_distance:.3f}m, "
                                           f"距离范围: {depth_result.min_distance:.3f}m - {depth_result.max_distance:.3f}m, "
                                           f"数据覆盖率: {depth_result.coverage_ratio:.1%}")
                        else:
                            self.logger.warning(f"第{measurement_count}次测量无有效数据")
                    else:
                        self.logger.warning("获取到空的深度图像")
                        
                except asyncio.TimeoutError:
                    self.logger.warning(f"获取深度图像超时 (时间: {elapsed_time:.1f}s)")
                
                # 等待1秒后进行下次测量
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            self.logger.info("深度距离测试被取消")
            raise
        except Exception as e:
            self.logger.error(f"深度距离测试异常: {str(e)}")
            raise
        finally:
            await self.main.orbbecCameraComponent.depth.unsubscribe(depth_queue)
            
            # 输出测试统计信息
            if valid_measurements:
                avg_distances = [m.average_distance for m in valid_measurements]
                min_distances = [m.min_distance for m in valid_measurements]
                max_distances = [m.max_distance for m in valid_measurements]
                coverage_ratios = [m.coverage_ratio for m in valid_measurements]
                
                self.logger.info("=== 深度距离测试统计 ===")
                self.logger.info(f"总测量次数: {measurement_count}, 有效测量: {len(valid_measurements)} ({len(valid_measurements)/measurement_count:.1%})")
                self.logger.info(f"平均距离统计 - 均值: {np.mean(avg_distances):.3f}m, "
                               f"最小: {np.min(avg_distances):.3f}m, "
                               f"最大: {np.max(avg_distances):.3f}m, "
                               f"标准差: {np.std(avg_distances):.3f}m")
                self.logger.info(f"最近距离统计 - 均值: {np.mean(min_distances):.3f}m, "
                               f"最小: {np.min(min_distances):.3f}m, "
                               f"最大: {np.max(min_distances):.3f}m")
                self.logger.info(f"最远距离统计 - 均值: {np.mean(max_distances):.3f}m, "
                               f"最小: {np.min(max_distances):.3f}m, "
                               f"最大: {np.max(max_distances):.3f}m")
                self.logger.info(f"数据覆盖率 - 均值: {np.mean(coverage_ratios):.1%}, "
                               f"最低: {np.min(coverage_ratios):.1%}, "
                               f"最高: {np.max(coverage_ratios):.1%}")
                self.logger.info("========================")
            else:
                self.logger.warning("测试期间未获得任何有效的深度数据")
                
            self.logger.info("深度距离测试结束")

    async def moveToTargetDistance(self, target_distance: float, speed: float = 0.1, timeout: float = 30.0):
        """
        向前行驶到深度摄像头检测到的指定距离，到达后停车
        :param target_distance: 目标距离（米）
        :param speed: 行驶速度，默认0.1m/s
        :param timeout: 超时时间（秒），默认30秒
        """
        self.logger.info(f"开始向前行驶到目标距离: {target_distance:.3f}m (速度: {speed:.2f}m/s)")
        
        # 订阅深度图像数据流
        depth_queue: asyncio.Queue[cv2.typing.MatLike] = await self.main.orbbecCameraComponent.depth.subscribe(
            asyncio.Queue(maxsize=1))
        
        # 禁用速度衰减以保持稳定的前进速度
        self.main.motionComponent.disableSpeedAttenuation()
        
        start_time = asyncio.get_event_loop().time()
        measurement_count = 0
        last_distance = None
        
        try:
            while True:
                current_time = asyncio.get_event_loop().time()
                elapsed_time = current_time - start_time
                
                # 检查是否超时
                if elapsed_time >= timeout:
                    self.logger.warning(f"行驶到目标距离超时: {timeout}秒内未到达目标")
                    raise Exception(f"行驶到目标距离超时: {timeout}秒内未到达目标距离{target_distance:.3f}m")
                
                try:
                    # 获取深度图像
                    depth_image = await asyncio.wait_for(depth_queue.get(), timeout=2.0)
                    
                    if depth_image is not None:
                        measurement_count += 1
                        
                        # 获取当前距离
                        depth_result = await asyncio.get_event_loop().run_in_executor(
                            None, self.depthImageCalculateDistance, depth_image
                        )
                        
                        if depth_result.valid:
                            current_distance = depth_result.average_distance
                            last_distance = current_distance
                            distance_diff = current_distance - target_distance
                            
                            self.logger.info(f"第{measurement_count}次测量 (时间: {elapsed_time:.1f}s): "
                                           f"当前距离: {current_distance:.3f}m, "
                                           f"目标距离: {target_distance:.3f}m, "
                                           f"差距: {distance_diff:+.3f}m, "
                                           f"覆盖率: {depth_result.coverage_ratio:.1%}")
                            
                            # 检查是否到达目标距离
                            if current_distance <= target_distance:
                                self.logger.info(f"到达目标距离！当前距离: {current_distance:.3f}m <= 目标距离: {target_distance:.3f}m")
                                self.main.motionComponent.stopMotion()
                                break
                            
                            # 距离控制逻辑 - 只有当距离大于目标时才前进
                            if distance_diff > 0:
                                # 当前距离大于目标距离，需要继续前进
                                self.main.motionComponent.setVelocity(linear_x=speed)
                            else:
                                # 当前距离小于等于目标距离，停车
                                self.main.motionComponent.stopMotion()
                                self.logger.info(f"已到达目标距离，停车")
                                break
                        else:
                            self.logger.warning(f"无法获取有效的距离数据 (覆盖率: {depth_result.coverage_ratio:.1%})，继续前进...")
                            self.main.motionComponent.setVelocity(linear_x=speed * 0.5)  # 降低速度
                    else:
                        self.logger.warning("获取到空的深度图像")
                        
                except asyncio.TimeoutError:
                    self.logger.warning(f"获取深度图像超时 (时间: {elapsed_time:.1f}s)")
                    # 超时时继续前进，但降低速度
                    self.main.motionComponent.setVelocity(linear_x=speed * 0.3)
                
                # 短暂延迟
                await asyncio.sleep(0.2)
                
        except asyncio.CancelledError:
            self.logger.info("行驶到目标距离被取消")
            raise
        except Exception as e:
            self.logger.error(f"行驶到目标距离异常: {str(e)}")
            raise
        finally:
            # 停止运动
            self.main.motionComponent.stopMotion()
            self.main.motionComponent.enableSpeedAttenuation()
            await self.main.orbbecCameraComponent.depth.unsubscribe(depth_queue)
            
            if last_distance is not None:
                self.logger.info(f"行驶结束 - 最终距离: {last_distance:.3f}m, "
                               f"目标距离: {target_distance:.3f}m, "
                               f"总测量次数: {measurement_count}, "
                               f"总耗时: {elapsed_time:.1f}秒")
            else:
                self.logger.info("行驶结束 - 未获取到有效距离数据")

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

                if key == "searchForTheTarget":
                    await self.searchForTheTarget()

                if key == "testDepthDistance":
                    await self.testDepthDistance()

                if key == "moveToDistance1m":
                    await self.moveToTargetDistance(1.0)  # 行驶到1米距离

                if key == "moveToDistance2m":
                    await self.moveToTargetDistance(2.0)  # 行驶到2米距离

                if key == "moveToDistance0.5m":
                    await self.moveToTargetDistance(0.5)  # 行驶到0.5米距离

                # 支持自定义距离格式：moveToDistance:1.5 (行驶到1.5米)
                if key.startswith("moveToDistance:"):
                    try:
                        distance_str = key.split(":")[1]
                        target_distance = float(distance_str)
                        if 0.1 <= target_distance <= 10.0:  # 限制合理的距离范围
                            await self.moveToTargetDistance(target_distance)
                        else:
                            self.logger.warning(f"目标距离超出范围: {target_distance}m (有效范围: 0.1m - 10.0m)")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的距离格式: {key}, 错误: {str(e)}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"instructionLoop Exception: {str(e)}")
