#!/usr/bin/python3

import asyncio
import logging
import json
from typing import Generic, TypeVar, Optional, Any, Union
import re
import cv2
import numpy as np

from pyorbbecsdk import FormatConvertFilter, VideoFrame, OBFormat, OBConvertFormat


logger = logging.getLogger(__name__)


class Box:

    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0

    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        pass

    pass


class Color:

    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 0

    def __init__(self, r: int, g: int, b: int, a: int = 255):

        self.r = r
        self.g = g
        self.b = b
        self.a = a

        pass

T = TypeVar("T")  # 定义泛型类型


class Broadcaster(Generic[T]):  # 继承 Generic 标记泛型类型
    def __init__(self):
        self.queues: list[asyncio.Queue[T]] = []  # 明确队列存储类型
        self.lock = asyncio.Lock()

    async def subscribe(
        self, queue: asyncio.Queue[T]
    ) -> asyncio.Queue[T]:  # 订阅的队列类型与泛型一致
        async with self.lock:
            self.queues.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[T]) -> None:
        async with self.lock:
            self.queues.remove(queue)

    async def publish(self, item: T) -> None:  # 发布项类型与泛型一致
        async with self.lock:
            for q in self.queues:
                if q.full():
                    q.get_nowait()
                await q.put(item)


class FFmpegPush:

    pushProcess: asyncio.subprocess.Process | None

    command: list[str]

    width: int
    height: int
    fps: int
    pushRtspUrl: str
    framesQueue: asyncio.Queue[cv2.typing.MatLike]

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        pushRtspUrl: str,
        framesQueue: asyncio.Queue[cv2.typing.MatLike],
        logTag: str,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.pushRtspUrl = pushRtspUrl
        self.framesQueue = framesQueue
        self.logger = logging.getLogger(logTag)

        self.command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            pushRtspUrl,
        ]

    async def pushFrames(self):
        while True:  # 无限循环尝试重启FFmpeg进程
            try:
                logger.info("正在启动FFmpeg推流进程...")
                # 启动FFmpeg进程
                pushProcess = await asyncio.create_subprocess_exec(
                    *self.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # 创建一个任务来监控进程的输出和错误
                while True:  # 主循环处理帧
                    frame = await self.framesQueue.get()

                    # 检查帧尺寸是否匹配预期
                    frame_height, frame_width = frame.shape[:2]
                    if (frame_width, frame_height) != (self.width, self.height):
                        self.logger.warning(
                            f"帧尺寸不匹配(期望{self.width}x{self.height}，实际{frame_width}x{frame_height})，正在调整尺寸..."
                        )
                        try:
                            # 使用双线性插值调整尺寸（可根据需求更换插值算法）
                            frame = cv2.resize(
                                frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
                            )
                        except Exception as resize_err:
                            self.logger.error(
                                f"调整帧尺寸失败: {str(resize_err)}，跳过该帧"
                            )
                            continue

                    # 将帧转换为字节流
                    data = frame.tobytes()

                    # 写入FFmpeg进程的输入管道
                    try:
                        if pushProcess.stdin is not None:
                            pushProcess.stdin.write(data)
                            await pushProcess.stdin.drain()  # 确保数据已写入
                    except (BrokenPipeError, ConnectionResetError) as e:
                        self.logger.error(f"写入FFmpeg进程失败: {e}")
                        raise  # 触发外层异常处理重新启动进程

                    # 检查FFmpeg进程是否仍在运行
                    returncode = pushProcess.returncode
                    if returncode is not None:
                        self.logger.error(
                            f"FFmpeg进程异常退出(代码{returncode})，正在重启..."
                        )
                        raise Exception("FFmpeg进程意外终止")

            except asyncio.CancelledError:
                self.logger.info("推流任务被取消，执行清理...")
                await self.releaseProcess()
                break

            except Exception as e:
                self.logger.error(f"推流任务异常: {str(e)}")
                await self.releaseProcess()
                self.logger.info(f"5秒后重启")
                await asyncio.sleep(5)

    async def releaseProcess(self):

        if self.pushProcess is not None and self.pushProcess.stdin is not None:
            self.pushProcess.stdin.close()
            await self.pushProcess.stdin.wait_closed()

            try:
                await asyncio.wait_for(self.pushProcess.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.pushProcess.kill()
                await self.pushProcess.wait()

            self.pushProcess = None

def getAllTasks() -> list[asyncio.Task]:
    """
    获取当前事件循环中的所有异步任务，并且排除当前调用的
    """
    loop = asyncio.get_event_loop()
    tasks = asyncio.all_tasks(loop)
    current_task = asyncio.current_task(loop)
    return [t for t in tasks if t is not current_task]


async def gracefulShutdown():
    """
    优雅关闭所有异步任务
    """
    tasks = getAllTasks()
    for task in tasks:
        task.cancel()  # 发送取消信号

    for task in tasks:
        try:
            await task  # 等待任务完成
        except asyncio.CancelledError:
            logging.error(f"Task {task.get_name()} was cancelled.")
        except Exception as e:
            logging.error(f"Task {task.get_name()} raised an exception:", e)


def flattenJson(data, parent_key="", sep=".", list_sep="[{}]"):
    items = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            items.update(flattenJson(value, new_key, sep, list_sep))
    elif isinstance(data, (list, tuple)):
        for index, item in enumerate(data):
            new_key = f"{parent_key}{list_sep.format(index)}"
            items.update(flattenJson(item, new_key, sep, list_sep))
    else:
        if parent_key:  # 忽略根元素为非容器的值
            items[parent_key] = data
    return items


def jsonDeepMerge(source, overrides):
    """
    JSON 深度合并（直接修改 source 字典）
    警告：此函数会直接修改传入的 source 参数！
    """
    for key, value in overrides.items():
        # 若双方均为字典，则递归合并
        if isinstance(value, dict) and isinstance(source.get(key), dict):
            jsonDeepMerge(source[key], value)
        # 否则直接覆盖（或添加）键值
        else:
            source[key] = value
    return source  # 返回修改后的 source


def parseKeyPath(key_str: str) -> list[tuple[str, Optional[int]]]:
    """解析路径字符串为键和索引的列表，例如 'a.b[0].c' -> [('a', None), ('b', 0), ('c', None)]"""
    parts = key_str.split(".")
    parsed = []
    pattern = re.compile(r"^(\w+)(?:\[(\d+)\])?$")
    for part in parts:
        match = pattern.match(part)
        if not match:
            raise ValueError(f"Invalid path part: {part}")
        key, index_str = match.groups()
        index = int(index_str) if index_str is not None else None
        parsed.append((key, index))
    return parsed


def getFromJson(key: str, ojson: dict) -> any:
    def parse_segment(seg):
        match = re.match(r"^([^\[\]]*?)((?:\[\d+\])*)$", seg)
        if not match:
            return None, []
        field = match.group(1)
        indices_str = match.group(2)
        indices = list(map(int, re.findall(r"\[(\d+)\]", indices_str)))
        return field, indices

    current = ojson
    segments = key.split(".") if key else []
    for seg in segments:
        field, indices = parse_segment(seg)
        if field is None:
            return None
        # Process field
        if field:
            if not isinstance(current, dict) or field not in current:
                return None
            current = current[field]
        # Process indices
        for index in indices:
            if not isinstance(current, list) or index < 0 or index >= len(current):
                return None
            current = current[index]
    return current


def setFromJson(key: str, value: any, ojson: dict) -> None:
    def parse_segment(seg):
        match = re.match(r"^([^\[\]]*?)((?:\[\d+\])*)$", seg)
        if not match:
            return None, []
        field = match.group(1)
        indices_str = match.group(2)
        indices = list(map(int, re.findall(r"\[(\d+)\]", indices_str)))
        return field, indices

    parent = None
    current = ojson
    key_or_index = None
    is_parent_list = False

    segments = key.split(".") if key else []
    for seg in segments:
        field, indices = parse_segment(seg)
        if field is None:
            return  # Invalid segment format, do nothing

        # Process field
        if field:
            # Ensure current is a dict
            if not isinstance(current, dict):
                # Replace current with a dict
                if parent is not None:
                    if is_parent_list:
                        parent[key_or_index] = {}
                    else:
                        parent[key_or_index] = {}
                current = {}
                # Update parent's reference
                if parent is not None and not is_parent_list:
                    parent[key_or_index] = current
            # Create the field if not exists
            if field not in current:
                current[field] = {}
            # Move down
            parent = current
            current = current[field]
            key_or_index = field
            is_parent_list = False

        # Process indices
        for index in indices:
            # Ensure current is a list
            if not isinstance(current, list):
                # Replace current with a list
                new_list = []
                if parent is not None:
                    if is_parent_list:
                        parent[key_or_index] = new_list
                    else:
                        parent[key_or_index] = new_list
                current = new_list
            # Extend the list if necessary, filling with empty dicts
            while len(current) <= index:
                current.append({})
            # Move down
            parent = current
            current = current[index]
            key_or_index = index
            is_parent_list = True

    # Set the value
    if parent is not None:
        parent[key_or_index] = value
    else:
        # Only possible if key is empty, replace ojson's content if possible
        if isinstance(ojson, dict) and isinstance(value, dict):
            ojson.clear()
            ojson.update(value)


def changeSize(
    inputImage: cv2.typing.MatLike, size: tuple[int, int]
) -> cv2.typing.MatLike:
    if inputImage is None:
        return None

    # 基础版：直接拉伸到目标尺寸（可能变形）
    # return cv2.resize(inputImage, size, interpolation=cv2.INTER_LINEAR)

    # 进阶版：保持比例 + 填充黑边（不变形）
    h, w = inputImage.shape[:2]
    target_w, target_h = size

    # 计算缩放比例并调整
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(inputImage, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算填充位置
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # 添加黑边填充
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def realBox(
    boxes: list[np.ndarray],
    originalSize: tuple[float, float],
    targetSize: tuple[float, float],
) -> list[Box]:
    """
    将模型输出的检测框坐标从预处理后的坐标系转换回原始图像坐标系，并封装为Box对象

    参数:
        boxes: np.ndarray, 形状为(N,4)的数组，每行表示一个检测框 (x1,y1,x2,y2)
               x1,y1为左上角坐标，x2,y2为右下角坐标（基于预处理后的图像坐标系）
        originalImageSize: tuple[float, float], 原始图像尺寸 (height, width)
        targetSize: tuple[float, float], 识别时的目标尺寸 (height, width)
        
    返回:
        List[Box]: 转换后的Box对象列表
    """
    # 获取原始图像尺寸
    orig_height, orig_width = originalSize

    # 预处理后的目标尺寸 (根据实际预处理参数修改，这里假设为640x640)
    processed_height, processed_width = targetSize

    # 计算缩放比例和填充量 --------------------------------------------------------
    scale = min(processed_height / orig_height, processed_width / orig_width)

    # 计算缩放后的新尺寸
    new_height = int(orig_height * scale)
    new_width = int(orig_width * scale)

    # 计算填充区域 (LetterBox的填充方式)
    pad_top = (processed_height - new_height) // 2
    pad_left = (processed_width - new_width) // 2

    # 坐标转换 ------------------------------------------------------------------
    box_list = []
    for box in boxes:
        # 解包坐标值
        x1, y1, x2, y2 = box

        # 去除填充并逆缩放
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        # 确保坐标不超出原始图像范围
        x1 = np.clip(x1, 0, orig_width)
        y1 = np.clip(y1, 0, orig_height)
        x2 = np.clip(x2, 0, orig_width)
        y2 = np.clip(y2, 0, orig_height)

        # 计算宽高
        width = x2 - x1
        height = y2 - y1

        # 创建Box对象（使用Python原生float类型）
        box_obj = Box(x=float(x1), y=float(y1), w=float(width), h=float(height))

        box_list.append(box_obj)

    return box_list


def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    yuyv = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return bgr_image


def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    uyvy = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image


def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image


def determine_convert_format(frame: VideoFrame):
    if frame.get_format() == OBFormat.I420:
        return OBConvertFormat.I420_TO_RGB888
    elif frame.get_format() == OBFormat.MJPG:
        return OBConvertFormat.MJPG_TO_RGB888
    elif frame.get_format() == OBFormat.YUYV:
        return OBConvertFormat.YUYV_TO_RGB888
    elif frame.get_format() == OBFormat.NV21:
        return OBConvertFormat.NV21_TO_RGB888
    elif frame.get_format() == OBFormat.NV12:
        return OBConvertFormat.NV12_TO_RGB888
    elif frame.get_format() == OBFormat.UYVY:
        return OBConvertFormat.UYVY_TO_RGB888
    else:
        return None


def frame_to_rgb_frame(frame: VideoFrame) -> Union[Optional[VideoFrame], Any]:
    if frame.get_format() == OBFormat.RGB:
        return frame
    convert_format = determine_convert_format(frame)
    if convert_format is None:
        print("Unsupported format")
        return None
    print("covert format: {}".format(convert_format))
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        print("Convert {} to RGB failed".format(frame.get_format()))
    return rgb_frame


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image