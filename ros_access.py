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
import threading

import util
from typing import Generic, TypeVar


from main import Component, ConfigField


class RosAccessComponent (Component):
    
    baseLaunch : ConfigField[str] = ConfigField()
    mapLaunch : ConfigField[str] = ConfigField()
    
    roscoreProcess = None
    
    def __init__(self):
        super().__init__()
        # 初始化日志处理器
        self.log_handler = self.LogHandler()
    
    class LogHandler(logging.Handler):
        """智能日志处理器，在输入时缓存日志"""
        
        def __init__(self):
            super().__init__()
            self.pending_logs = []
            self.is_inputting = False
            self.lock = threading.Lock()
            self.input_component = None
            
        def set_input_state(self, is_inputting: bool):
            """设置输入状态"""
            with self.lock:
                self.is_inputting = is_inputting
                if not is_inputting and self.pending_logs:
                    # 输入结束，输出所有缓存的日志
                    for log_msg in self.pending_logs:
                        print(log_msg)
                    self.pending_logs.clear()
                    # 重新显示提示符
                    if self.input_component and hasattr(self.input_component, 'display_prompt_and_input'):
                        try:
                            self.input_component.display_prompt_and_input()
                        except:
                            pass  # 如果显示失败就忽略
        
        def add_pending_log(self, log_msg: str) -> bool:
            """添加待处理日志，返回是否被缓存"""
            with self.lock:
                if self.is_inputting:
                    self.pending_logs.append(log_msg)
                    # 限制缓存大小，避免内存过度使用
                    if len(self.pending_logs) > 100:
                        self.pending_logs.pop(0)
                    return True
                return False
        
        def set_input_component(self, component):
            """设置输入组件引用"""
            self.input_component = component
        
        def emit(self, record):
            try:
                formatted_msg = self.format(record)
                
                # 检查是否在输入状态，如果是则缓存日志
                if not self.add_pending_log(formatted_msg):
                    # 不在输入状态，直接输出
                    print(formatted_msg)
                    
            except Exception as e:
                # 发生异常时直接输出，避免日志丢失
                try:
                    print(self.format(record))
                except:
                    pass

    def isRosMasterRunning(self):
        """检查 ROS Master 是否正在运行"""
        try:
            # 使用 rosgraph 检查 master 是否可用
            master = rosgraph.Master('/rostopic')
            master.getPid()
            return True
        except Exception:
            return False
    
    def startRosMaster(self):
        """启动 ROS Master (roscore)"""
        try:
            self.logger.info("ROS Master 未运行，正在启动 roscore...")
            
            # 启动 roscore 进程
            self.roscoreProcess = subprocess.Popen(
                ['roscore'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # 等待 ROS Master 启动
            maxWaitTime = 30  # 最大等待30秒
            waitInterval = 0.5  # 每0.5秒检查一次
            waitedTime = 0
            
            while waitedTime < maxWaitTime:
                if self.isRosMasterRunning():
                    self.logger.info(f"ROS Master 启动成功！用时 {waitedTime:.1f} 秒")
                    return True
                    
                time.sleep(waitInterval)
                waitedTime += waitInterval
                self.logger.debug(f"等待 ROS Master 启动... ({waitedTime:.1f}s)")
            
            self.logger.error("ROS Master 启动超时！")
            return False
            
        except Exception as e:
            self.logger.error(f"启动 ROS Master 时出错: {e}")
            return False
    
    def stopRosMaster(self):
        """停止 ROS Master"""
        if self.roscoreProcess:
            try:
                self.logger.info("正在停止 ROS Master...")
                if os.name == 'nt':  # Windows
                    self.roscoreProcess.terminate()
                else:  # Unix/Linux
                    os.killpg(os.getpgid(self.roscoreProcess.pid), signal.SIGTERM)
                
                self.roscoreProcess.wait(timeout=5)
                self.logger.info("ROS Master 已停止")
            except Exception as e:
                self.logger.error(f"停止 ROS Master 时出错: {e}")
            finally:
                self.roscoreProcess = None
    
    async def awakeInit(self):
        await super().awakeInit()
        
        # 首先检查 ROS Master 是否运行，如果没有则启动它
        if not self.isRosMasterRunning():
            self.logger.info("检测到 ROS Master 未运行")
            if not self.startRosMaster():
                raise Exception("无法启动 ROS Master，请检查 ROS 环境配置")
        else:
            self.logger.info("ROS Master 已经在运行")
        
        # 检查 rospy 是否已经启动，如果未启动则主动启动
        try:
            # 尝试获取 ROS 时间来检查是否已经初始化
            rospy.get_rostime()
            self.logger.info("ROS 节点已经初始化")
        except rospy.exceptions.ROSInitException:
            # rospy 未初始化，主动启动
            self.logger.info("ROS 节点未初始化，正在启动...")
            rospy.init_node("car_python")
            self.logger.info("ROS 节点初始化完成")
        except Exception as e:
            # 其他异常，说明可能未初始化或有其他问题
            self.logger.info(f"检查 ROS 状态时出现异常，尝试初始化: {e}")
            try:
                rospy.init_node("car_python")
                self.logger.info("ROS 节点初始化完成")
            except rospy.exceptions.ROSException as rosE:
                self.logger.warning(f"ROS 节点可能已经初始化: {rosE}")
        
            
        # 确保 ROS 节点正常运行
        if not rospy.is_shutdown():
            self.logger.info("ROS 节点运行正常")
        
        # 配置日志处理器
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        self.log_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        
        # 移除现有的处理器，避免重复
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel("DEBUG")
        
        self.logger.info("日志处理器配置完成")

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
            if self.roscoreProcess:
                self.stopRosMaster()
                self.logger.info("ROS Master 已停止")
                
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")
    
    pass