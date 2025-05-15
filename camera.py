#!/usr/bin/python3

import logging
import asyncio
import numpy as np
import cv2
import detection
import time
import hkws
import ctypes
import sys
import util
from ctypes import *
from typing import *
from asyncio.subprocess import PIPE
from util import Broadcaster, FFmpegPushFrame, ByteFFmpegPull
from enum import IntEnum, unique

logger = logging.getLogger(__name__)

ip = "192.168.117.100"
rtsp_port = "554"
port = 8000
user = "admin"
password = "qWERTYUIOP"
SDKPath = "/home/elf/HCNetSDKV6.1.9.45_build20220902_ArmLinux64_ZH/MakeAll/"

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


pushRtspUrl = "rtsp://localhost:8554/stream1"


# region HKWS_SDK





class CameraRealPlayData:
    camera: "Camera"
    dataType: int
    data: bytes

    def __init__(self, camera: "Camera", dataType: int, data: bytes):
        self.camera = camera
        self.dataType = dataType
        self.data = data
        pass

    pass


class CameraVoiceData:
    camera: "Camera"
    data: bytes

    def __init__(self, camera: "Camera", data: bytes):
        self.camera = camera
        self.data = data


RealPlayCallBackType = CFUNCTYPE(None, c_int, c_int, c_void_p, c_int, c_void_p)
VoiceDataCallBackType = CFUNCTYPE(None, c_int, c_void_p, c_int, c_byte, c_void_p)


class Camera:

    user: str
    password: str
    ip: str
    port: int

    soList: list[str] = []

    userId: int = -1
    realHandle: int = -1
    voiceHandle: int = -1
    audioDecoderHandle : int = -1

    libCache: dict[str, CDLL] = {}
    funcCache: dict[str, Optional[Any]] = {}

    lastCell: str = ""

    realPlayBroadcaster: Broadcaster[CameraRealPlayData] = Broadcaster()
    voiceBroadcaster: Broadcaster[CameraVoiceData] = Broadcaster()

    def __init__(self, user: str, password: str, ip: str, port: int):
        self.user = user
        self.password = password
        self.ip = ip
        self.port = port

    def addSo(self, soPath: str):
        if soPath not in self.soList:
            self.soList.append(soPath)

    def addSoFromDir(self, dirPath: str):
        import os

        for root, dirs, files in os.walk(dirPath):
            for file in files:
                if file.endswith(".so"):
                    soPath = os.path.join(root, file)
                    self.addSo(soPath)

    def loadDll(self, dllPath: str) -> CDLL | None:
        if dllPath in self.libCache:
            return self.libCache[dllPath]

        lib = None
        try:
            lib = cdll.LoadLibrary(dllPath)
            self.libCache[dllPath] = lib
        except Exception as e:
            logging.exception(f"库加载失败: {dllPath} - {str(e)}")

        return lib

    def loadFunc(self, funcName: str) -> object | None:

        if funcName in self.funcCache:
            return self.funcCache[funcName]

        func: Optional[Any] = None

        for soPath in self.soList:
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

    def callCpp(self, funcName: str, *args) -> object:
        func: Optional[Any] | None = self.loadFunc(funcName)

        if func is None:
            return None

        self.lastCell = funcName

        try:
            return func(*args)
        except Exception as e:
            logging.warning(f"{funcName}() 函数执行失败: - {str(e)}")
            del self.funcCache[funcName]
            return None

    def initSdk(self):
        if not self.callCpp("NET_DVR_Init"):
            self.raiseLastError("NET_DVR_Init() has error")

    def initG711Decoder(self):
        self. audioDecoderHandle = self.callCpp("NET_DVR_InitG711Decoder") # type: ignore
        if self.audioDecoderHandle == None:
            self.audioDecoderHandle = -1
            self.raiseLastError()
            
    def releaseG711Decoder(self):
        if not self.callCpp("NET_DVR_ReleaseG711Decoder"):
            self.raiseLastError()

    def activateDevice(self):
        activate: NET_DVR_ACTIVATECFG = NET_DVR_ACTIVATECFG()
        activate.dwSize = sizeof(activate)
        util.fillBuffer(activate, "sPassword", bytes(password, "ascii"))

        if not self.callCpp(
            "NET_DVR_ActivateDevice", bytes(ip, "ascii"), self.port, byref(activate)
        ):
            self.raiseLastError("NET_DVR_ActivateDevice() has error")

    def setConnectTime(self, time: int = 5000, retry: int = 4):
        if not self.callCpp("NET_DVR_SetConnectTime", time, retry):
            self.raiseLastError("NET_DVR_SetConnectTime() has error")

    def setReconnect(self, time: int = 10000, enable: bool = True):
        if not self.callCpp("NET_DVR_SetReconnect", time, enable):
            self.raiseLastError("NET_DVR_SetReconnect() has error")

    def login(self):
        userInfo: NET_DVR_USER_LOGIN_INFO = NET_DVR_USER_LOGIN_INFO()
        userInfo.bUseAsynLogin = 0
        util.fillBuffer(userInfo, "sDeviceAddress", bytes(ip, "ascii"))
        userInfo.wPort = self.port
        util.fillBuffer(userInfo, "sUserName", bytes(self.user, "ascii"))
        util.fillBuffer(userInfo, "sPassword", bytes(self.password, "ascii"))

        deviceInfo: NET_DVR_DEVICEINFO_V40 = NET_DVR_DEVICEINFO_V40()

        self.userId = self.callCpp("NET_DVR_Login_V40", byref(userInfo), byref(deviceInfo))  # type: ignore
        if self.userId == -1:
            self.raiseLastError("NET_DVR_Login_V40() has error")

    def logout(self):
        if not self.callCpp("NET_DVR_Logout", self.userId):
            self.raiseLastError("NET_DVR_Logout() has error")

    def realPlay(self):
        req: NET_DVR_PREVIEWINFO = NET_DVR_PREVIEWINFO()

        req.hPlayWnd = None
        req.lChannel = 1  # 预览通道号
        req.dwStreamType = (
            0  # 码流类型：0-主码流，1-子码流，2-三码流，3-虚拟码流，以此类推
        )
        req.dwLinkMode = 0  # 连接方式：0-TCP方式，1-UDP方式，2-多播方式，3-RTP方式，4-RTP/RTSP，5-RTP/HTTP,6-HRUDP（可靠传输）
        req.bBlocked = 0  # 0-非阻塞 1-阻塞

        self.realHandle = self.callCpp("NET_DVR_RealPlay_V40", self.userId, byref(req), None, None)  # type: ignore
        if self.realHandle < 0:
            self.raiseLastError()

    def setRealDataCallBack(self):
        if not self.callCpp(
            "NET_DVR_SetRealDataCallBack",
            self.realHandle,
            self.generateRealPlayCallBack(),
            self.userId,
        ):
            self.raiseLastError()

    def setStandardDataCallBack(self):
        if not self.callCpp(
            "NET_DVR_SetStandardDataCallBack",
            self.realHandle,
            self.generateRealPlayCallBack(),
            self.userId,
        ):
            self.raiseLastError()

    def stopPreview(self):
        if not self.callCpp("NET_DVR_StopRealPlay"):
            self.raiseLastError()

    def startVoiceCom(self):
        self.voiceHandle = self.callCpp("NET_DVR_StartVoiceCom", self.userId, self.generateVoiceDataCallBack(), None)  # type: ignore
        if self.voiceHandle == -1:
            self.raiseLastError()
        pass

    def startVoiceComMr(self):
        self.voiceHandle = self.callCpp("NET_DVR_StartVoiceCom_MR", self.userId, self.generateVoiceDataCallBack(), None)  # type: ignore
        if self.voiceHandle == -1:
            self.raiseLastError()
        pass

    def stopVoiceCom(self):
        if not self.callCpp("NET_DVR_StopVoiceCom", self.voiceHandle):
            self.raiseLastError()

    def sdkClean(self):
        if not self.callCpp("NET_DVR_Cleanup"):
            self.raiseLastError()

    def getLastError(self) -> int:
        return int(self.callCpp("NET_DVR_GetLastError"))  # type: ignore

    def logLastError(self, message: str):
        errorCode = self.getLastError()
        logger.error(f"{message}, the errorCode is {errorCode}")

    def raiseLastError(self, message: str | None = None):
        if message is None:
            message = f"{self.lastCell}() has error"

        errorCode = self.getLastError()
        raise CameraException(message, errorCode)

    def ptzControlOther(self, channel: int, command: DeviceCommand, action: int):
        """
        :param channel: 通道号
        :param action: 控制动作
        :return:
        """
        if not self.callCpp(
            "NET_DVR_PTZControl_Other", self.userId, channel, command.value, action
        ):
            self.raiseLastError()

    realPlayCallBack: Callable[[int, int, c_void_p, int, c_void_p], None] | None = None

    def generateRealPlayCallBack(self):

        if self.realPlayCallBack is not None:
            return self.realPlayCallBack

        def realPlayCallBack(
            lRealHandle: int,
            dwDataType: int,
            pBuffer: c_void_p,
            dwBufSize: int,
            dwUser: c_void_p,
        ):

            data: bytes = ctypes.string_at(pBuffer, dwBufSize)
            self.realPlayBroadcaster.publish_nowait(
                CameraRealPlayData(self, dwDataType, data)
            )
            pass

        self.realPlayCallBack = RealPlayCallBackType(realPlayCallBack)
        return self.realPlayCallBack

    voiceDataCallBack: Callable[[int, c_void_p, int, c_byte, c_void_p], None] | None = None

    def generateVoiceDataCallBack(self):

        if self.voiceDataCallBack is not None:
            return self.voiceDataCallBack

        def voiceDataCallBack(
            lVoiceHandle: int,
            pRecvDataBuffer: c_void_p,
            dwBufSize: int,
            byAudioFlag: c_byte,
            dwUser: c_void_p,
        ):
            
            logger.debug(f"收到语音数据{byAudioFlag} {dwBufSize}") 
            if byAudioFlag != 1:
                return

            data: bytes = ctypes.string_at(pRecvDataBuffer, dwBufSize)
            self.voiceBroadcaster.publish_nowait(CameraVoiceData(self, data))

        self.voiceDataCallBack = VoiceDataCallBackType(voiceDataCallBack)
        return self.voiceDataCallBack


# endregion

camera = None

cap: cv2.VideoCapture = None  # type: ignore


async def releaseCap():
    global cap
    if cap and cap.isOpened():
        await asyncio.get_event_loop().run_in_executor(None, cap.release)
    cap = None  # type: ignore
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

    camera = Camera(user, password, ip, port)
    camera.addSoFromDir(SDKPath)

    camera.initSdk()
    camera.initG711Decoder()
    camera.setConnectTime()
    camera.setReconnect()
    camera.login()
    camera.realPlay()
    camera.setRealDataCallBack()
    camera.startVoiceComMr()

    # camera.ptzControlOther(1, DeviceCommand.TILT_DOWN, 0)

    # asyncio.create_task(readFrames())
    # asyncio.create_task(extractAudio())
    asyncio.create_task(handleFrames())
    asyncio.create_task(pushFrames())


async def releaseCamera():
    logger.info("释放摄像头资源")
    if camera is not None:
        camera.releaseG711Decoder()
        camera.logout()
        camera.sdkClean()
