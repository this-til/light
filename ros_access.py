
import roslaunch
import roslaunch.parent
import roslaunch.rlutil

import rospy
import rosgraph

import logging
import logging.config
import asyncio
import rospy
import cv2
import subprocess
import time
import os
import signal

import util
from typing import Generic, TypeVar


from main import Component, ConfigField


class RosAccessComponent (Component):
    
    baseLaunch : ConfigField[str] = ConfigField()
    mapLaunch : ConfigField[str] = ConfigField()
    
    roscore_process = None
    
    def is_ros_master_running(self):
        """检查 ROS Master 是否正在运行"""
        try:
            # 使用 rosgraph 检查 master 是否可用
            master = rosgraph.Master('/rostopic')
            master.getPid()
            return True
        except Exception:
            return False
    
    def start_ros_master(self):
        """启动 ROS Master (roscore)"""
        try:
            print("ROS Master 未运行，正在启动 roscore...")
            
            # 启动 roscore 进程
            self.roscore_process = subprocess.Popen(
                ['roscore'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # 等待 ROS Master 启动
            max_wait_time = 30  # 最大等待30秒
            wait_interval = 0.5  # 每0.5秒检查一次
            waited_time = 0
            
            while waited_time < max_wait_time:
                if self.is_ros_master_running():
                    print(f"ROS Master 启动成功！用时 {waited_time:.1f} 秒")
                    return True
                    
                time.sleep(wait_interval)
                waited_time += wait_interval
                print(f"等待 ROS Master 启动... ({waited_time:.1f}s)")
            
            print("ROS Master 启动超时！")
            return False
            
        except Exception as e:
            print(f"启动 ROS Master 时出错: {e}")
            return False
    
    def stop_ros_master(self):
        """停止 ROS Master"""
        if self.roscore_process:
            try:
                print("正在停止 ROS Master...")
                if os.name == 'nt':  # Windows
                    self.roscore_process.terminate()
                else:  # Unix/Linux
                    os.killpg(os.getpgid(self.roscore_process.pid), signal.SIGTERM)
                
                self.roscore_process.wait(timeout=5)
                print("ROS Master 已停止")
            except Exception as e:
                print(f"停止 ROS Master 时出错: {e}")
            finally:
                self.roscore_process = None
    
    async def awakeInit(self):
        await super().awakeInit()
        
        # 首先检查 ROS Master 是否运行，如果没有则启动它
        if not self.is_ros_master_running():
            print("检测到 ROS Master 未运行")
            if not self.start_ros_master():
                raise Exception("无法启动 ROS Master，请检查 ROS 环境配置")
        else:
            print("ROS Master 已经在运行")
        
        # 检查 rospy 是否已经启动，如果未启动则主动启动
        try:
            # 尝试获取 ROS 时间来检查是否已经初始化
            rospy.get_rostime()
            print("ROS 节点已经初始化")
        except rospy.exceptions.ROSInitException:
            # rospy 未初始化，主动启动
            print("ROS 节点未初始化，正在启动...")
            rospy.init_node("car_python")
            print("ROS 节点初始化完成")
        except Exception as e:
            # 其他异常，说明可能未初始化或有其他问题
            print(f"检查 ROS 状态时出现异常，尝试初始化: {e}")
            try:
                rospy.init_node("car_python")
                print("ROS 节点初始化完成")
            except rospy.exceptions.ROSException as ros_e:
                print(f"ROS 节点可能已经初始化: {ros_e}")
            
        # 确保 ROS 节点正常运行
        if not rospy.is_shutdown():
            rospy.loginfo("ROS 节点运行正常")
    

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
    
    async def release(self):
        """组件释放，清理资源，停止 ROS Master（如果是由此组件启动的）"""
        await super().release()
        
        try:
            # 停止 launch 文件
            if hasattr(self, 'baseLaunchFile') and self.baseLaunchFile:
                self.baseLaunchFile.shutdown()
                self.logger.info("基础 launch 文件已停止")
            
            if hasattr(self, 'mapLaunchFile') and self.mapLaunchFile:
                self.mapLaunchFile.shutdown()
                self.logger.info("地图 launch 文件已停止")
            
            # 如果 ROS Master 是由此组件启动的，则停止它
            if self.roscore_process:
                self.stop_ros_master()
                self.logger.info("ROS Master 已停止")
                
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")
    
    pass