#!/usr/bin/python3

import logging
import asyncio
import numpy as np
import cv2
import detection
import time
import hkws
from ctypes import cdll, CDLL
from typing import *
from hkws.cm_camera_adpt import CameraAdapter
from hkws.config import Config
from asyncio.subprocess import PIPE
from util import Broadcaster, FFmpegPushFrame, ByteFFmpegPull
from enum import IntEnum, unique

logger = logging.getLogger(__name__)

ip = "192.168.117.100"
rtsp_port = "554"
user = "admin"
password = "qWERTYUIOP"

width, height = 2560, 1440
fps = 25

# 音频参数
FREQUENCY = 16000  # 采样率16kHz
SAMPLE_SIZE = -16  # 16位有符号PCM
CHANNELS = 1  # 单声道
BUFFER_SIZE = 4096  # 每次读取的数据块大小，须为2的倍数


cameraRtspUrl = f"rtsp://{user}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101"

source: Broadcaster[cv2.typing.MatLike] = Broadcaster()
audioSource: Broadcaster[bytes] = Broadcaster()
out: Broadcaster[cv2.typing.MatLike] = Broadcaster()
identifyKeyframe: Broadcaster[detection.Result] = Broadcaster()

cap: cv2.VideoCapture

pushRtspUrl = "rtsp://localhost:8554/stream1"

cnf = Config()
cnf.User = user
cnf.Password = password
cnf.IP = ip
cnf.SDKPath = "/home/elf/HCNetSDKV6.1.9.45_build20220902_ArmLinux64_ZH/MakeAll/HCNetSDKCom"


@unique
class DeviceCommand(IntEnum):
    """
    设备控制指令枚举（值映射协议指令码）
    每个枚举项的第一个值为协议定义值，注释为功能说明
    """

    # 电源控制类指令
    LIGHT_PWRON = 2  # 接通灯光电源
    WIPER_PWRON = 3  # 接通雨刷开关
    FAN_PWRON = 4  # 接通风扇开关
    HEATER_PWRON = 5  # 接通加热器开关
    AUX_PWRON1 = 6  # 辅助设备开关1
    AUX_PWRON2 = 7  # 辅助设备开关2

    # 光学控制类指令
    ZOOM_IN = 11  # 焦距变大(倍率变大)
    ZOOM_OUT = 12  # 焦距变小(倍率变小)
    FOCUS_NEAR = 13  # 焦点前调
    FOCUS_FAR = 14  # 焦点后调
    IRIS_OPEN = 15  # 光圈扩大
    IRIS_CLOSE = 16  # 光圈缩小

    # 云台基础运动指令
    TILT_UP = 21  # 云台上仰
    TILT_DOWN = 22  # 云台下俯
    PAN_LEFT = 23  # 云台左转
    PAN_RIGHT = 24  # 云台右转
    UP_LEFT = 25  # 云台上仰+左转
    UP_RIGHT = 26  # 云台上仰+右转
    DOWN_LEFT = 27  # 云台下俯+左转
    DOWN_RIGHT = 28  # 云台下俯+右转
    PAN_AUTO = 29  # 云台自动扫描模式

    # 复合运动指令（云台+光学组合）
    TILT_DOWN_ZOOM_IN = 58  # 下俯+焦距变大
    TILT_DOWN_ZOOM_OUT = 59  # 下俯+焦距变小
    PAN_LEFT_ZOOM_IN = 60  # 左转+焦距变大
    PAN_LEFT_ZOOM_OUT = 61  # 左转+焦距变小
    PAN_RIGHT_ZOOM_IN = 62  # 右转+焦距变大
    PAN_RIGHT_ZOOM_OUT = 63  # 右转+焦距变小

    # 三维复合运动指令
    UP_LEFT_ZOOM_IN = 64  # 上仰左转+焦距变大
    UP_LEFT_ZOOM_OUT = 65  # 上仰左转+焦距变小
    UP_RIGHT_ZOOM_IN = 66  # 上仰右转+焦距变大
    UP_RIGHT_ZOOM_OUT = 67  # 上仰右转+焦距变小
    DOWN_LEFT_ZOOM_IN = 68  # 下俯左转+焦距变大
    DOWN_LEFT_ZOOM_OUT = 69  # 下俯左转+焦距变小
    DOWN_RIGHT_ZOOM_IN = 70  # 下俯右转+焦距变大
    DOWN_RIGHT_ZOOM_OUT = 71  # 下俯右转+焦距变小
    TILT_UP_ZOOM_IN = 72  # 上仰+焦距变大
    TILT_UP_ZOOM_OUT = 73  # 上仰+焦距变小


class Camera(CameraAdapter):
    userId: int = -1

    libCache: dict[str, CDLL] = {}
    funcCache: dict[str, Optional[Any]] = {}

    def loadDll(self, dllPath: str) -> CDLL | None:
        if dllPath in self.libCache:
            return self.libCache[dllPath]
        try:
            lib = cdll.LoadLibrary(dllPath)
            self.libCache[dllPath] = lib
        except Exception as e:
            logging.exception(f"库加载失败: {dllPath} - {str(e)}")

        return None

    def loadFunc(self, funcName: str) -> object | None:

        if funcName in self.funcCache:
            return self.funcCache[funcName]

        func: Optional[Any] = None

        for soPath in self.so_list:
            lib: CDLL | None = self.loadDll(soPath)
            if lib is None:
                continue

            try:
                func = getattr(lib, funcName)
                self.funcCache[funcName] = func
            except AttributeError:
                continue

        if func is None:
            logging.debug(f"{funcName}() 函数不存在")
            return None

        return func

    def call_cpp(self, func_name: str, *args) -> object:
        func : Optional[Any]  | None = self.loadFunc(func_name)

        if func is None:
            return None

        try:
            return func(*args)
        except Exception as e:
            logging.warning(f"{func_name}() 函数执行失败: - {str(e)}")
            del self.funcCache[func_name]
            return None

    def ptzControl(self, channel: DeviceCommand, action: int):
        """
        :param channel: 通道号
        :param action: 控制动作
        :return:
        """
        self.call_cpp("NET_DVR_PTZControl", self.userId, channel.value, action)
        pass

    pass


camera = None


async def releaseCap():
    global cap
    if cap and cap.isOpened():
        await asyncio.get_event_loop().run_in_executor(None, cap.release)
    cap = None
    pass


async def readFrames():
    global cap

    while True:

        try:

            logger.info("尝试连接摄像头...")

            # gst_pipeline = f"rtspsrc location={pushRtspUrl} latency=0 ! rtph264depay ! h264parse ! nvh264dec ! videoconvert ! appsink"
            # cap = await asyncio.get_event_loop().run_in_executor(None, cv2.VideoCapture, gst_pipeline, cv2.CAP_GSTREAMER)

            cap = await asyncio.get_event_loop().run_in_executor(
                None, cv2.VideoCapture, cameraRtspUrl
            )

            if not cap.isOpened():
                logger.warning("连接失败，5秒后重试...")
                await asyncio.sleep(5)
                continue

            while True:
                ret: bool
                frame: cv2.typing.MatLike

                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    None, cap.read
                )

                if not ret:
                    logger.error("读取帧失败，释放资源并重连...")
                    raise Exception("读取帧失败...")

                h, w = frame.shape[:2]
                if w < h:
                    frame = await asyncio.get_event_loop().run_in_executor(
                        None, cv2.rotate, frame, cv2.ROTATE_90_CLOCKWISE
                    )

                await source.publish(frame)

        except asyncio.CancelledError:
            logger.info("任务被取消，执行清理...")
            await releaseCap()
            raise

        except Exception as e:
            logger.exception(f"发生未处理异常:  {str(e)}")
            await releaseCap()
            logger.info("5秒后尝试重新连接摄像头...")
            await asyncio.sleep(5)

        pass


async def extractAudio():

    ffmpeg_command = [
        "ffmpeg",
        "-i",
        cameraRtspUrl,  # 输入RTSP流
        "-vn",  # 忽略视频
        "-acodec",
        "pcm_s16le",  # 输出PCM格式
        "-ar",
        str(FREQUENCY),  # 采样率
        "-ac",
        str(CHANNELS),  # 声道数
        "-f",
        "s16le",  # 输出格式为s16le
        "-loglevel",
        "quiet",  # 屏蔽FFmpeg日志
        "pipe:1",  # 输出到标准输出
    ]

    async def publishToAudio(b: bytes):
        await audioSource.publish(b)

    await ByteFFmpegPull(
        ffmpeg_command, BUFFER_SIZE, publishToAudio, "cameraAudio"
    ).loop()

    pass


async def pushFrames():
    await FFmpegPushFrame(
        width,
        height,
        fps,
        pushRtspUrl,
        await out.subscribe(asyncio.Queue(maxsize=16)),
        __name__,
    ).loop()
    pass


async def handleFrames():

    framesQueue: asyncio.Queue[cv2.typing.MatLike] = asyncio.Queue(maxsize=16)
    await source.subscribe(framesQueue)

    res: detection.Result | None = None
    task: asyncio.Future[detection.Result] | None = None

    while True:
        try:

            sourceFrame: cv2.typing.MatLike = await framesQueue.get()

            if res is None:
                res = detection.Result(sourceFrame, {})

            if task is None or task.done():

                if task is not None:
                    res = task.result()
                    await identifyKeyframe.publish(res)

                def _runDetection(
                    inputImage: cv2.typing.MatLike, useModel: list[detection.Model]
                ):
                    # start_time = time.perf_counter()
                    res = detection.runDetection(inputImage, useModel)
                    # end_time = time.perf_counter()
                    # duration_ms = (end_time - start_time) * 1000
                    # logger.info(f"inference 耗时: {duration_ms:.3f}ms")
                    return res

                task = asyncio.get_event_loop().run_in_executor(
                    None,
                    _runDetection,
                    sourceFrame,
                    detection.models,
                )

            _res = detection.Result(sourceFrame, res.cellMap)
            await out.publish(await _res.drawOutputImageAsunc())

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"处理帧时发生异常: {str(e)}")

            res = None
            if task is not None:
                task.cancel()
                task = None
        pass

    pass


async def initCamera():
    logger.info("初始化摄像头模块")

    global camera

    camera = Camera()
    camera.userId = camera.common_start(cnf)

    camera.ptzControl(DeviceCommand.TILT_UP, 1)

    asyncio.create_task(readFrames())
    asyncio.create_task(extractAudio())
    asyncio.create_task(handleFrames())
    asyncio.create_task(pushFrames())


async def releaseCamera():
    logger.info("释放摄像头资源")
    if camera is not None:
        camera.logout(camera.userId)
        camera.sdk_clean()
