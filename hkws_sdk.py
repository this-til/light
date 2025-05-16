#!/usr/bin/python3
import sys
import logging
import util

from ctypes import *
from typing import *
from threading import local
from enum import IntEnum, unique

from main import Component, ConfigField

logger = logging.getLogger(__name__)

NET_DVR_SYSHEAD = 1  # 系统头数据
NET_DVR_STREAMDATA = 2  # 流数据（包括复合流或音视频分开的视频流数据）
NET_DVR_AUDIOSTREAMDATA = 3  # 音频数据
NET_DVR_PRIVATE_DATA = 112  # 	私有数据,包括智能信息

h_BOOL = c_bool
h_CHAR = c_char
h_BYTE = c_byte
h_INT = c_int
h_WORD = c_uint16
h_LONG = c_long
h_FLOAT = c_float
h_DWORD = c_ulong  # 64bit:c_ulong    32bit:c_uint32

if not sys.maxsize > 2**32:
    h_DWORD = c_uint32

h_VOID_P = c_void_p
h_HWND = c_void_p  # handle of window
h_CHAR_P = c_ubyte
h_BYTE_P = c_ubyte

PASSWD_LEN = 16


# 设置sdk加载路劲
class NET_DVR_LOCAL_SDK_PATH(Structure):
    _fields_ = [("sPath", h_BYTE * 256), ("byRes", h_BYTE * 128)]


# 登录参数结构体
class NET_DVR_USER_LOGIN_INFO(Structure):
    _fields_ = [
        ("sDeviceAddress", h_BYTE * 129),  # 设备地址，IP或者普通域名
        ("byUseTransport", h_BYTE),  # 是否启用能力透传 0：不启动，默认  1：启动
        ("wPort", h_WORD),  # 设备端口号
        ("sUserName", h_BYTE * 64),  # 登录用户名
        ("sPassword", h_BYTE * 64),  # 登录密码
        # ("fLoginResultCallBack",)  #
        ("bUseAsynLogin", h_BOOL),  # 是否异步登录, 0:否 1:是
        (
            "byProxyType",
            h_BYTE,
        ),  # 代理服务器类型：0- 不使用代理，1- 使用标准代理，2- 使用EHome代理
        # 是否使用UTC时间：
        # 0 - 不进行转换，默认；
        # 1 - 输入输出UTC时间，SDK进行与设备时区的转换；
        # 2 - 输入输出平台本地时间，SDK进行与设备时区的转换
        ("byUseUTCTime", h_BYTE),
        # 登录模式(不同模式具体含义详见“Remarks”说明)：
        # 0- SDK私有协议，
        # 1- ISAPI协议，
        # 2- 自适应（设备支持协议类型未知时使用，一般不建议）
        ("byLoginMode", h_BYTE),
        # ISAPI协议登录时是否启用HTTPS(byLoginMode为1时有效)：
        # 0 - 不启用，
        # 1 - 启用，
        # 2 - 自适应（设备支持协议类型未知时使用，一般不建议）
        ("byHttps", h_BYTE),
        # 代理服务器序号，添加代理服务器信息时相对应的服务器数组下表值
        ("iProxyID", h_LONG),
        # 保留，置为0
        ("byRes3", h_BYTE * 120),
    ]


# 设备参数结构体。
class NET_DVR_DEVICEINFO_V30(Structure):
    _fields_ = [
        ("sSerialNumber", h_BYTE * 48),  # 序列号
        ("byAlarmInPortNum", h_BYTE),  # 模拟报警输入个数
        ("byAlarmOutPortNum", h_BYTE),  # 模拟报警输出个数
        ("byDiskNum", h_BYTE),  # 硬盘个数
        ("byDVRType", h_BYTE),  # 设备类型，详见下文列表
        (
            "byChanNum",
            h_BYTE,
        ),  # 设备模拟通道个数，数字(IP)通道最大个数为byIPChanNum + byHighDChanNum*256
        (
            "byStartChan",
            h_BYTE,
        ),  # 模拟通道的起始通道号，从1开始。数字通道的起始通道号见下面参数byStartDChan
        ("byAudioChanNum", h_BYTE),  # 设备语音对讲通道数
        ("byIPChanNum", h_BYTE),
        # 设备最大数字通道个数，低8位，搞8位见byHighDChanNum. 可以根据ip通道个数是否调用NET_DVR_GetDVRConfig (
        # 配置命令NET_DVR_GET_IPPARACFG_V40)获得模拟和数字通道的相关参数
        ("byZeroChanNum", h_BYTE),  # 零通道编码个数
        (
            "byMainProto",
            h_BYTE,
        ),  # 主码流传输协议类型： 0 - private, 1 - rtsp, 2- 同时支持私有协议和rtsp协议去留（默认采用私有协议取流）
        (
            "bySubProto",
            h_BYTE,
        ),  # 字码流传输协议类型： 0 - private , 1 - rtsp , 2 - 同时支持私有协议和rtsp协议取流 （默认采用私有协议取流）
        # 能力，位与结果为0表示不支持，1
        # 表示支持
        # bySupport & 0x1，表示是否支持智能搜索
        # bySupport & 0x2，表示是否支持备份
        # bySupport & 0x4，表示是否支持压缩参数能力获取
        # bySupport & 0x8, 表示是否支持双网卡
        # bySupport & 0x10, 表示支持远程SADP
        # bySupport & 0x20, 表示支持Raid卡功能
        # bySupport & 0x40, 表示支持IPSAN目录查找
        # bySupport & 0x80, 表示支持rtp over rtsp
        ("bySupport", h_BYTE),
        # 能力集扩充，位与结果为0表示不支持，1
        # 表示支持
        # bySupport1 & 0x1, 表示是否支持snmp
        # v30
        # bySupport1 & 0x2, 表示是否支持区分回放和下载
        # bySupport1 & 0x4, 表示是否支持布防优先级
        # bySupport1 & 0x8, 表示智能设备是否支持布防时间段扩展
        # bySupport1 & 0x10, 表示是否支持多磁盘数（超过33个）
        # bySupport1 & 0x20, 表示是否支持rtsp over http
        # bySupport1 & 0x80, 表示是否支持车牌新报警信息，且还表示是否支持NET_DVR_IPPARACFG_V40配置
        ("bySupport1", h_BYTE),
        # 能力集扩充，位与结果为0表示不支持，1
        # 表示支持
        # bySupport2 & 0x1, 表示解码器是否支持通过URL取流解码
        # bySupport2 & 0x2, 表示是否支持FTPV40
        # bySupport2 & 0x4, 表示是否支持ANR(断网录像)
        # bySupport2 & 0x20, 表示是否支持单独获取设备状态子项
        # bySupport2 & 0x40, 表示是否是码流加密设备
        ("bySupport2", h_BYTE),
        ("wDevType", h_WORD),  # 设备型号，详见下文列表
        # 能力集扩展，位与结果：0 - 不支持，1 - 支持
        # bySupport3 & 0x1, 表示是否支持多码流
        # bySupport3 & 0x4, 表示是否支持按组配置，具体包含通道图像参数、报警输入参数、IP报警输入 / 输出接入参数、用户参数、设备工作状态、JPEG抓图、定时和时间抓图、硬盘盘组管理等
        # bySupport3 & 0x20,表示是否支持通过DDNS域名解析取流
        ("bySupport3", h_BYTE),
        # 是否支持多码流，按位表示，位与结果：0 - 不支持，1 - 支持
        # byMultiStreamProto & 0x1, 表示是否支持码流3
        # byMultiStreamProto & 0x2, 表示是否支持码流4
        # byMultiStreamProto & 0x40, 表示是否支持主码流
        # byMultiStreamProto & 0x80, 表示是否支持子码流
        ("byMultiStreamProto", h_BYTE),
        ("byStartDChan", h_BYTE),  # 起始数字通道号，0表示无数字通道，比如DVR或IPC
        (
            "byStartDTalkChan",
            h_BYTE,
        ),  # 起始数字对讲通道号，区别于模拟对讲通道号，0表示无数字对讲通道
        ("byHighDChanNum", h_BYTE),  # 数字通道个数，高8位
        # 能力集扩展，按位表示，位与结果：0 - 不支持，1 - 支持
        # bySupport4 & 0x01, 表示是否所有码流类型同时支持RTSP和私有协议
        # bySupport4 & 0x10, 表示是否支持域名方式挂载网络硬盘
        ("bySupport4", h_BYTE),
        # 支持语种能力，按位表示，位与结果：0 - 不支持，1 - 支持
        # byLanguageType == 0，表示老设备，不支持该字段
        # byLanguageType & 0x1，表示是否支持中文
        # byLanguageType & 0x2，表示是否支持英文
        ("byLanguageType", h_BYTE),
        ("byVoiceInChanNum", h_BYTE),  # 音频输入通道数
        ("byStartVoiceInChanNo", h_BYTE),  # 音频输入起始通道号，0表示无效
        ("byRes3", h_BYTE * 2),  # 保留，置为0
        ("byMirrorChanNum", h_BYTE),  # 镜像通道个数，录播主机中用于表示导播通道
        ("wStartMirrorChanNo", h_WORD),  # 起始镜像通道号
        ("byRes2", h_BYTE * 2),
    ]  # 保留，置为0


class NET_DVR_DEVICEINFO_V40(Structure):
    _fields_ = [
        ("struDeviceV30", NET_DVR_DEVICEINFO_V30),  # 设备参数
        (
            "bySupportLock",
            h_BYTE,
        ),  # 设备是否支持锁定功能，bySuportLock 为1时，dwSurplusLockTime和byRetryLoginTime有效
        (
            "byRetryLoginTime",
            h_BYTE,
        ),  # 剩余可尝试登陆的次数，用户名，密码错误时，此参数有效
        # 密码安全等级： 0-无效，1-默认密码，2-有效密码，3-风险较高的密码，
        # 当管理员用户的密码为出厂默认密码（12345）或者风险较高的密码时，建议上层客户端提示用户名更改密码
        ("byPasswordLevel", h_BYTE),
        (
            "byProxyType",
            h_BYTE,
        ),  # 代理服务器类型，0-不使用代理，1-使用标准代理，2-使用EHome代理
        # 剩余时间，单位：秒，用户锁定时次参数有效。在锁定期间，用户尝试登陆，不算用户名密码输入对错
        # 设备锁定剩余时间重新恢复到30分钟
        ("dwSurplusLockTime", h_DWORD),
        # 字符编码类型（SDK所有接口返回的字符串编码类型，透传接口除外）：
        # 0 - 无字符编码信息（老设备）
        # 1 - GB2312
        ("byCharEncodeType", h_BYTE),
        # 支持v50版本的设备参数获取，设备名称和设备类型名称长度扩展为64字节
        ("bySupportDev5", h_BYTE),
        # 登录模式（不同的模式具体含义详见"Remarks"说明：0- SDK私有协议，1- ISAPI协议）
        ("byLoginMode", h_BYTE),
        # 保留，置为0
        ("byRes2", h_BYTE * 253),
    ]


class NET_DVR_Login_V40(Structure):
    _fields_ = [
        ("pLoginInfo", NET_DVR_USER_LOGIN_INFO),
        ("lpDeviceInfo", NET_DVR_DEVICEINFO_V40),
    ]


# 设备激活参数结构体
class NET_DVR_ACTIVATECFG(Structure):
    _fields_ = [
        ("dwSize", h_DWORD),
        ("sPassword", h_BYTE * PASSWD_LEN),
        ("byRes", h_BYTE * 108),
    ]


# SDK状态信息结构体
class NET_DVR_SDKSTATE(Structure):
    _fields_ = [
        ("dwTotalLoginNum", h_DWORD),  # 当前注册用户数
        ("dwTotalRealPlayNum", h_DWORD),  # 当前实时预览的路数
        ("dwTotalPlayBackNum", h_DWORD),  # 当前回放或下载的路数
        ("dwTotalAlarmChanNum", h_DWORD),  # 当前建立报警通道的路数
        ("dwTotalFormatNum", h_DWORD),  # 当前硬盘格式化的路数
        ("dwTotalFileSearchNum", h_DWORD),  # 当前文件搜索的路数
        ("dwTotalLogSearchNum", h_DWORD),  # 当前日志搜索的路数
        ("dwTotalSerialNum", h_DWORD),  # 当前建立透明通道的路数
        ("dwTotalUpgradeNum", h_DWORD),  # 当前升级的路数
        ("dwTotalVoiceComNum", h_DWORD),  # 当前语音转发的路数
        ("dwTotalBroadCastNum", h_DWORD),  # 当前语音广播的路数
        (" dwRes", h_DWORD * 10),  # 保留，置为0
    ]


# SDK功能信息结构体
class NET_DVR_SDKABL(Structure):
    _fields_ = [
        ("dwMaxLoginNum", h_DWORD),  # 最大注册用户数
        ("dwMaxRealPlayNum", h_DWORD),  # 最大实时预览的路数
        ("dwMaxPlayBackNum", h_DWORD),  # 最大回放或下载的路数
        ("dwMaxAlarmChanNum", h_DWORD),  # 最大建立报警通道的路数
        ("dwMaxFormatNum", h_DWORD),  # 最大硬盘格式化的路数
        ("dwMaxFileSearchNum", h_DWORD),  # 最大文件搜索的路数
        ("dwMaxLogSearchNum", h_DWORD),  # 最大日志搜索的路数
        ("dwMaxSerialNum", h_DWORD),  # 最大建立透明通道的路数
        ("dwMaxUpgradeNum", h_DWORD),  # 最大升级的路数
        ("dwMaxVoiceComNum", h_DWORD),  # 最大语音转发的路数
        ("dwMaxBroadCastNum", h_DWORD),  # 最大语音广播的路数
        (" dwRes", h_DWORD * 10),  # 保留，置为0
    ]


# 预览参数结构体
class NET_DVR_PREVIEWINFO(Structure):
    _fields_ = [
        # 通道号，目前设备模拟通道号从1开始，数字通道的起始通道号通过
        # NET_DVR_GetDVRConfig(配置命令NET_DVR_GET_IPPARACFG_V40)获取（dwStartDChan）
        ("lChannel", h_LONG),
        # 码流类型：0-主码流，1-子码流，2-三码流，3-虚拟码流，以此类推
        ("dwStreamType", h_DWORD),
        # 连接方式：0-TCP方式，1-UDP方式，2-多播方式，3-RTP方式，4-RTP/RTSP，5-RTP/HTTP,6-HRUDP（可靠传输）
        ("dwLinkMode", h_DWORD),
        # 播放窗口的句柄，为NULL表示不解码显示
        ("hPlayWnd", h_HWND),
        # 0-非阻塞取流，1- 阻塞取流
        # 若设为不阻塞，表示发起与设备的连接就认为连接成功，如果发生码流接收失败、播放失败等
        # 情况以预览异常的方式通知上层。在循环播放的时候可以减短停顿的时间，与NET_DVR_RealPlay
        # 处理一致。
        # 若设为阻塞，表示直到播放操作完成才返回成功与否，网络异常时SDK内部connect失败将会有5s
        # 的超时才能够返回，不适合于轮询取流操作。
        ("bBlocked", h_BOOL),
        # 是否启用录像回传： 0-不启用录像回传，1-启用录像回传。ANR断网补录功能，
        # 客户端和设备之间网络异常恢复之后自动将前端数据同步过来，需要设备支持。
        ("bPassbackRecord", h_BOOL),
        # 延迟预览模式：0-正常预览，1-延迟预览
        ("byPreviewMode", h_BYTE),
        # 流ID，为字母、数字和"_"的组合，IChannel为0xffffffff时启用此参数
        ("byStreamID", h_BYTE * 32),
        # 应用层取流协议：0-私有协议，1-RTSP协议。
        # 主子码流支持的取流协议通过登录返回结构参数NET_DVR_DEVICEINFO_V30的byMainProto、bySubProto值得知。
        # 设备同时支持私协议和RTSP协议时，该参数才有效，默认使用私有协议，可选RTSP协议。
        ("byProtoType", h_BYTE),
        # 保留，置为0
        ("byRes1", h_BYTE),
        # 码流数据编解码类型：0-通用编码数据，1-热成像探测器产生的原始数据
        # （温度数据的加密信息，通过去加密运算，将原始数据算出真实的温度值）
        ("byVideoCodingType", h_BYTE),
        # 播放库播放缓冲区最大缓冲帧数，取值范围：1、6（默认，自适应播放模式）   15:置0时默认为1
        ("dwDisplayBufNum", h_DWORD),
        # NPQ模式：0- 直连模式，1-过流媒体模式
        ("byNPQMode", h_BYTE),
        # 保留，置为0
        ("byRes", h_BYTE * 215),
    ]


class NET_DVR_AUDIODEC_INFO(Structure):
    _fields_ = [
        ("nchans", c_int),  # 声道数
        ("sample_rate", c_int),  # 采样率
        ("aacdec_profile", c_int),  # 编码框架（保留）
        ("reserved", c_int * 16),  # 保留字段
    ]


class NET_DVR_AUDIODEC_PROCESS_PARAM(Structure):
    _fields_ = [
        ("in_buf", c_void_p),  # 输入缓冲区指针
        ("out_buf", c_void_p),  # 输出缓冲区指针
        ("in_data_size", c_uint32),  # 输入数据大小
        ("proc_data_size", c_uint32),  # 已处理数据大小
        ("out_frame_size", c_uint32),  # 输出帧大小
        ("dec_info", NET_DVR_AUDIODEC_INFO),  # 解码信息结构体
        ("g726dec_reset", c_int),  # G726重置开关
        ("g711_type", c_int),  # G711编码类型
        ("reserved", c_int * 16),  # 保留字段
    ]


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


class CameraException(Exception):

    errorCode: int

    def __init__(self, message: str, errorCode: int):
        super().__init__(f"{message} , the errorCode is {errorCode}")
        self.errorCode = errorCode

    pass


# region 库方法加载与调用

libCache: dict[str, CDLL] = {}
funcCache: dict[str, Optional[Any]] = {}
soList: list[str] = []


def addSo(soPath: str):
    if soPath not in soList:
        soList.append(soPath)


def addSoFromDir(dirPath: str):
    import os

    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if file.endswith(".so"):
                soPath = os.path.join(root, file)
                addSo(soPath)


def loadDll(dllPath: str) -> CDLL | None:
    if dllPath in libCache:
        return libCache[dllPath]
    lib = None
    try:
        lib = cdll.LoadLibrary(dllPath)
        libCache[dllPath] = lib
    except Exception as e:
        logging.exception(f"库加载失败: {dllPath} - {str(e)}")
    return lib


def loadFunc(funcName: str) -> object | None:
    if funcName in funcCache:
        return funcCache[funcName]
    func: Optional[Any] = None
    for soPath in soList:
        lib: CDLL | None = loadDll(soPath)
        if lib is None:
            continue
        try:
            func = getattr(lib, funcName)
            funcCache[funcName] = func
        except AttributeError:
            continue
    if func is None:
        logging.debug(f"{funcName}() 函数不存在")
        return None
    return func


lastCell: str = ""


def callCpp(funcName: str, *args) -> object:
    func: Optional[Any] | None = loadFunc(funcName)
    if func is None:
        return None
    global lastCell
    lastCell = funcName
    try:
        return func(*args)
    except Exception as e:
        logging.warning(f"{funcName}() 函数执行失败: - {str(e)}")
        del funcCache[funcName]
        return None


def getLastError() -> int:
    return int(callCpp("NET_DVR_GetLastError"))  # type: ignore


def logLastError(message: str):
    errorCode = getLastError()
    logger.error(f"{message}, the errorCode is {errorCode}")


def raiseLastError(message: str | None = None):
    if message is None:
        message = f"{lastCell}() has error"
    errorCode = getLastError()
    raise CameraException(message, errorCode)


# endregion

# region SDK初始化和注销


def initSdk():
    if not callCpp("NET_DVR_Init"):
        raiseLastError()


def setConnectTime(time: int = 5000, retry: int = 4):
    if not callCpp("NET_DVR_SetConnectTime", time, retry):
        raiseLastError()


def setReconnect(time: int = 10000, enable: bool = True):
    if not callCpp("NET_DVR_SetReconnect", time, enable):
        raiseLastError()


def sdkClean():
    if not callCpp("NET_DVR_Cleanup"):
        raiseLastError()


# endregion

# region 音频解析


def initG711Decoder() -> c_void_p:
    audioDecoderHandle: c_void_p = callCpp("NET_DVR_InitG711Decoder")  # type: ignore
    if audioDecoderHandle == None:
        raiseLastError()
    return audioDecoderHandle


def releaseG711Decoder(audioDecoderHandle: c_void_p):
    if not callCpp("NET_DVR_ReleaseG711Decoder", audioDecoderHandle):
        raiseLastError()


# endregion

# region 激活设备


def activateDevice(ip: str, port: int, password: str):

    activate: NET_DVR_ACTIVATECFG = NET_DVR_ACTIVATECFG()
    activate.dwSize = sizeof(activate)
    util.fillBuffer(activate, "sPassword", bytes(password, "ascii"))

    if not callCpp("NET_DVR_ActivateDevice", bytes(ip, "ascii"), port, byref(activate)):
        raiseLastError()


# endregion


# region 监控登录和注销


def login(ip: str, port: int, user: str, password: str) -> int:
    userInfo: NET_DVR_USER_LOGIN_INFO = NET_DVR_USER_LOGIN_INFO()
    userInfo.bUseAsynLogin = 0
    util.fillBuffer(userInfo, "sDeviceAddress", bytes(ip, "ascii"))
    userInfo.wPort = port
    util.fillBuffer(userInfo, "sUserName", bytes(user, "ascii"))
    util.fillBuffer(userInfo, "sPassword", bytes(password, "ascii"))

    deviceInfo: NET_DVR_DEVICEINFO_V40 = NET_DVR_DEVICEINFO_V40()

    return loginFromInfo(userInfo, deviceInfo)


def loginFromInfo(
    userInfo: NET_DVR_USER_LOGIN_INFO, deviceInfo: NET_DVR_DEVICEINFO_V40
) -> int:

    userId: int = callCpp("NET_DVR_Login_V40", byref(userInfo), byref(deviceInfo))  # type: ignore

    if userId == -1:
        raiseLastError()

    return userId


def logout(userId: int):
    if not callCpp("NET_DVR_Logout", userId):
        raiseLastError("")


# endregion


# region 预览


def realPlay(userId: int):
    req: NET_DVR_PREVIEWINFO = NET_DVR_PREVIEWINFO()

    req.hPlayWnd = None
    req.lChannel = 1  # 预览通道号
    req.dwStreamType = 0  # 码流类型：0-主码流，1-子码流，2-三码流，3-虚拟码流，以此类推
    req.dwLinkMode = 0  # 连接方式：0-TCP方式，1-UDP方式，2-多播方式，3-RTP方式，4-RTP/RTSP，5-RTP/HTTP,6-HRUDP（可靠传输）
    req.bBlocked = 0  # 0-非阻塞 1-阻塞

    return realPlayFromInfo(userId, req)


def realPlayFromInfo(userId: int, req: NET_DVR_PREVIEWINFO) -> int:
    realHandle: int = callCpp("NET_DVR_RealPlay_V40", userId, byref(req), None, None)  # type: ignore
    if realHandle < 0:
        raiseLastError()

    return realHandle


def setRealDataCallBack(userId: int, realHandle: int):
    if not callCpp(
        "NET_DVR_SetRealDataCallBack",
        realHandle,
        RealPlayCallBackType(realPlayCallBack),
        userId,
    ):
        raiseLastError()


def setStandardDataCallBack(userId: int, realHandle: int):
    if not callCpp(
        "NET_DVR_SetStandardDataCallBack",
        realHandle,
        VoiceDataCallBackType(voiceDataCallBack),
        userId,
    ):
        raiseLastError()


def stopPreview(realHandle: int):
    if not callCpp("NET_DVR_StopRealPlay", realHandle):
        raiseLastError()


# endregion

# region 音频


def startVoiceCom(userId: int) -> int:
    voiceHandle: int = callCpp("NET_DVR_StartVoiceCom", userId, VoiceDataCallBackType(voiceDataCallBack), None)  # type: ignore
    if voiceHandle == -1:
        raiseLastError()
    return voiceHandle


def startVoiceComMr(userId: int) -> int:
    voiceHandle: int = callCpp("NET_DVR_StartVoiceCom_MR", userId, VoiceDataCallBackType(voiceDataCallBack), None)  # type: ignore
    if voiceHandle == -1:
        raiseLastError()
    return voiceHandle


def stopVoiceCom(voiceHandle: int):
    if not callCpp("NET_DVR_StopVoiceCom", voiceHandle):
        raiseLastError()


# endregion

# region Bus

voiceOutBuf = local()


def getVoiceOutBuf() -> type[Array[c_char]]:
    if voiceOutBuf.value is None:
        voiceOutBuf.value = create_string_buffer(1024 * 1024)
    return voiceOutBuf.value  # type: ignore


class CameraRealPlayData:
    handle: int
    dataType: int
    data: bytes
    user: c_void_p

    def __init__(
        self, lRealHandle: int, dwDataType: int, data: bytes, dwUser: c_void_p
    ):
        self.handle = lRealHandle
        self.dataType = dwDataType
        self.data = data
        self.user = dwUser


class CameraVoiceData:
    handle: int
    data: bytes
    audioFlag: int
    user: c_void_p

    def __init__(self, handle: int, data: bytes, audioFlag: int, user: c_void_p):
        self.handle = handle
        self.data = data
        self.audioFlag = audioFlag
        self.user = user


realPlayBroadcaster: util.Broadcaster[CameraRealPlayData] = util.Broadcaster()
voiceBroadcaster: util.Broadcaster[CameraVoiceData] = util.Broadcaster()

# endregion

# region 通用回调

RealPlayCallBackType = CFUNCTYPE(None, c_int, c_int, c_void_p, c_int, c_void_p)
VoiceDataCallBackType = CFUNCTYPE(None, c_int, c_void_p, c_int, c_byte, c_void_p)


def realPlayCallBack(
    lRealHandle: int,
    dwDataType: int,
    pBuffer: c_void_p,
    dwBufSize: int,
    dwUser: c_void_p,
):

    # logger.debug(f"realPlayCallBack: {dwDataType}")

    realPlayBroadcaster.publish_nowait(
        CameraRealPlayData(
            lRealHandle, dwDataType, string_at(pBuffer, dwBufSize), dwUser
        )
    )


def voiceDataCallBack(
    lVoiceHandle: int,
    pRecvDataBuffer: c_void_p,
    dwBufSize: int,
    byAudioFlag: int,
    dwUser: c_void_p,
):

    # p: NET_DVR_AUDIODEC_PROCESS_PARAM = NET_DVR_AUDIODEC_PROCESS_PARAM()
    # p.in_buf = pRecvDataBuffer
    # p.in_data_size = dwBufSize

    voiceBroadcaster.publish_nowait(
        CameraVoiceData(
            lVoiceHandle, string_at(pRecvDataBuffer, dwBufSize), byAudioFlag, dwUser
        )
    )


# endregion

# region 云台控制


def ptzControlOther(userId: int, channel: int, command: DeviceCommand, action: int):
    """
    :param channel: 通道号
    :param action: 控制动作
    :return:
    """
    if not callCpp("NET_DVR_PTZControl_Other", userId, channel, command.value, action):
        raiseLastError()


# endregion

SDKPath = "/home/elf/HCNetSDKV6.1.9.45_build20220902_ArmLinux64_ZH/MakeAll/"
addSoFromDir(SDKPath)


class HCNetSdkComponent(Component):
    
    SDKPath : ConfigField[str ] = ConfigField()
    
    async def init(self):
        await super().init()
        addSoFromDir(self.SDKPath)
        initSdk()
        setConnectTime()
        setReconnect()
        

    async def release(self):        
        sdkClean()
        pass
        
    def getPriority(self) -> int:
        return 1 << 8
