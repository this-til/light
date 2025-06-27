
import roslaunch
import roslaunch.parent
import roslaunch.rlutil

import rospy

import logging
import logging.config
import asyncio
import rospy
import cv2

import util
from typing import Generic, TypeVar


from main import Component, ConfigField

class ROSLogHandler(logging.Handler):
    """将 Python logging 重定向到 ROS 日志系统"""
    
    def emit(self, record):
        ## 映射日志级别到 ROS 对应函数
        #log_functions = {
        #    logging.DEBUG: rospy.logdebug,
        #    logging.INFO: rospy.loginfo,
        #    logging.WARNING: rospy.logwarn,
        #    logging.ERROR: rospy.logerr,
        #    logging.CRITICAL: rospy.logfatal
        #}
        #
        ## 获取匹配的 ROS 日志函数（默认使用 logerr）
        #log_func = log_functions.get(record.levelno, rospy.logerr)
        #
        ## 格式化日志消息并发送到 ROS
        #message = self.format(record)
        #log_func(message)
        
        try:
            print(self.format(record))
        except Exception as e:
            pass
        

class RosAccessComponent (Component):
    
    baseLaunch : ConfigField[str] = ConfigField()
    mapLaunch : ConfigField[str] = ConfigField()
    
    async def awakeInit(self):
        await super().awakeInit()
        
            
        rospy.init_node("car_python")
        
        handler = ROSLogHandler()
        
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        log_level = rospy.get_param("~log_level", "DEBUG")
        root_logger.setLevel(log_level)
        

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