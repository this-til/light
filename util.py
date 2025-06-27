#!/usr/bin/python3
from __future__ import annotations

import asyncio
import logging
import math
import platform
import re
from ctypes import Structure
from typing import *

import cv2
import numpy as np
from pyorbbecsdk import FormatConvertFilter, VideoFrame, OBFormat, OBConvertFormat

systemPlatFrom = platform.system()

logger = logging.getLogger(__name__)

T = TypeVar("T")  # 定义泛型类型


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

    def normalization(self, width: float, height: float) -> 'Box':
        """
        将坐标归一化到[0, 1]范围内
        """

        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be greater than zero.")

        return Box(
            x=self.x / width,
            y=self.y / height,
            w=self.w / width,
            h=self.h / height,
        )

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


class V2:
    """二维向量类"""
    x: float = 0.0
    y: float = 0.0

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def __add__(self, other: 'V2') -> 'V2':
        return V2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'V2') -> 'V2':
        return V2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'V2':
        return V2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'V2':
        if scalar == 0:
            raise ValueError("除数不能为零")
        return V2(self.x / scalar, self.y / scalar)

    def dot(self, other: 'V2') -> float:
        """点积"""
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        """向量长度"""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> 'V2':
        """归一化"""
        mag = self.magnitude()
        if mag == 0:
            return V2(0, 0)
        return V2(self.x / mag, self.y / mag)

    def distance(self, other: 'V2') -> float:
        """与另一个向量的距离"""
        return (self - other).magnitude()

    def __str__(self) -> str:
        return f"V2({self.x:.3f}, {self.y:.3f})"

    def __repr__(self) -> str:
        return self.__str__()


class V3:
    """三维向量类"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: 'V3') -> 'V3':
        return V3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'V3') -> 'V3':
        return V3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'V3':
        return V3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'V3':
        if scalar == 0:
            raise ValueError("除数不能为零")
        return V3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: 'V3') -> float:
        """点积"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'V3') -> 'V3':
        """叉积"""
        return V3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self) -> float:
        """向量长度"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> 'V3':
        """归一化"""
        mag = self.magnitude()
        if mag == 0:
            return V3(0, 0, 0)
        return V3(self.x / mag, self.y / mag, self.z / mag)

    def distance(self, other: 'V3') -> float:
        """与另一个向量的距离"""
        return (self - other).magnitude()

    def toV2(self) -> V2:
        """转换为二维向量（忽略z分量）"""
        return V2(self.x, self.y)

    def __str__(self) -> str:
        return f"V3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __repr__(self) -> str:
        return self.__str__()


class V4:
    """四维向量类"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __add__(self, other: 'V4') -> 'V4':
        return V4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other: 'V4') -> 'V4':
        return V4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __mul__(self, scalar: float) -> 'V4':
        return V4(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)

    def __truediv__(self, scalar: float) -> 'V4':
        if scalar == 0:
            raise ValueError("除数不能为零")
        return V4(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)

    def dot(self, other: 'V4') -> float:
        """点积"""
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def magnitude(self) -> float:
        """向量长度"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def normalize(self) -> 'V4':
        """归一化"""
        mag = self.magnitude()
        if mag == 0:
            return V4(0, 0, 0, 0)
        return V4(self.x / mag, self.y / mag, self.z / mag, self.w / mag)

    def distance(self, other: 'V4') -> float:
        """与另一个向量的距离"""
        return (self - other).magnitude()

    def toV3(self) -> V3:
        """转换为三维向量（忽略w分量）"""
        return V3(self.x, self.y, self.z)

    def toV2(self) -> V2:
        """转换为二维向量（忽略z和w分量）"""
        return V2(self.x, self.y)

    def __str__(self) -> str:
        return f"V4({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.w:.3f})"

    def __repr__(self) -> str:
        return self.__str__()


class Velocity:
    """速度类 - 封装线性和角速度"""
    linear: V3
    angular: V3
    
    def __init__(self, linear: V3 = None, angular: V3 = None):
        self.linear = linear if linear is not None else V3()
        self.angular = angular if angular is not None else V3()
    
    @staticmethod
    def create(linear_x: float = 0.0, linear_y: float = 0.0, linear_z: float = 0.0,
               angular_x: float = 0.0, angular_y: float = 0.0, angular_z: float = 0.0) -> 'Velocity':
        """创建速度对象"""
        return Velocity(
            linear=V3(linear_x, linear_y, linear_z),
            angular=V3(angular_x, angular_y, angular_z)
        )
    
    def setLinear(self, x: float = None, y: float = None, z: float = None):
        """设置线性速度分量"""
        if x is not None:
            self.linear.x = x
        if y is not None:
            self.linear.y = y
        if z is not None:
            self.linear.z = z
    
    def setAngular(self, x: float = None, y: float = None, z: float = None):
        """设置角速度分量"""
        if x is not None:
            self.angular.x = x
        if y is not None:
            self.angular.y = y
        if z is not None:
            self.angular.z = z
    
    def hasVelocity(self, threshold: float = 0.01) -> bool:
        """检查是否有速度"""
        return (abs(self.linear.x) > threshold or 
                abs(self.linear.y) > threshold or 
                abs(self.linear.z) > threshold or
                abs(self.angular.x) > threshold or 
                abs(self.angular.y) > threshold or 
                abs(self.angular.z) > threshold)
    
    def stop(self):
        """停止所有速度"""
        self.linear = V3()
        self.angular = V3()
    
    def attenuate(self, factor: float = 0.85, threshold: float = 0.01):
        """衰减速度"""
        self.linear = self.linear * factor
        self.angular = self.angular * factor
        
        # 低于阈值时设为0
        if abs(self.linear.x) < threshold:
            self.linear.x = 0.0
        if abs(self.linear.y) < threshold:
            self.linear.y = 0.0
        if abs(self.linear.z) < threshold:
            self.linear.z = 0.0
        if abs(self.angular.x) < threshold:
            self.angular.x = 0.0
        if abs(self.angular.y) < threshold:
            self.angular.y = 0.0
        if abs(self.angular.z) < threshold:
            self.angular.z = 0.0
    
    def __str__(self) -> str:
        return f"Velocity(linear={self.linear}, angular={self.angular})"


class Pose:
    """姿态类 - 封装位置和朝向"""
    position: V3
    orientation: Quaternion
    
    def __init__(self, position: V3 = None, orientation: Quaternion = None):
        self.position = position if position is not None else V3()
        self.orientation = orientation if orientation is not None else Quaternion.identity()
    
    @staticmethod
    def create(x: float = 0.0, y: float = 0.0, z: float = 0.0,
               qx: float = 0.0, qy: float = 0.0, qz: float = 0.0, qw: float = 1.0) -> 'Pose':
        """创建姿态对象"""
        return Pose(
            position=V3(x, y, z),
            orientation=Quaternion(qx, qy, qz, qw)
        )
    
    @staticmethod
    def fromEuler(x: float = 0.0, y: float = 0.0, z: float = 0.0,
                  roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> 'Pose':
        """从欧拉角创建姿态对象"""
        return Pose(
            position=V3(x, y, z),
            orientation=Quaternion.fromEulerAngles(roll, pitch, yaw)
        )
    
    @staticmethod
    def fromEulerDegrees(x: float = 0.0, y: float = 0.0, z: float = 0.0,
                         rollDeg: float = 0.0, pitchDeg: float = 0.0, yawDeg: float = 0.0) -> 'Pose':
        """从欧拉角（度数）创建姿态对象"""
        return Pose(
            position=V3(x, y, z),
            orientation=Quaternion.fromEulerAnglesDegrees(rollDeg, pitchDeg, yawDeg)
        )
    
    def setPosition(self, x: float = None, y: float = None, z: float = None):
        """设置位置分量"""
        if x is not None:
            self.position.x = x
        if y is not None:
            self.position.y = y
        if z is not None:
            self.position.z = z
    
    def __str__(self) -> str:
        return f"Pose(position={self.position}, orientation={self.orientation})"


class Quaternion:
    """四元数类"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    @staticmethod
    def fromEulerAngles(roll: float, pitch: float, yaw: float) -> 'Quaternion':
        """从欧拉角（弧度）创建四元数"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        return Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy
        )
    
    @staticmethod
    def fromEulerAnglesDegrees(rollDeg: float, pitchDeg: float, yawDeg: float) -> 'Quaternion':
        """从欧拉角（度数）创建四元数"""
        return Quaternion.fromEulerAngles(
            degreesToRadians(rollDeg),
            degreesToRadians(pitchDeg),
            degreesToRadians(yawDeg)
        )
    
    def toEulerAngles(self) -> 'V3':
        """转换为欧拉角（弧度）- 返回 (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return V3(roll, pitch, yaw)
    
    def toEulerAnglesDegrees(self) -> 'V3':
        """转换为欧拉角（度数）- 返回 (roll, pitch, yaw)"""
        e= self.toEulerAngles()
        return V3(
            math.degrees(e.x),
            math.degrees(e.y),
            math.degrees(e.z)
        )

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """四元数乘法"""
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        )

    def magnitude(self) -> float:
        """四元数长度"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def normalize(self) -> 'Quaternion':
        """归一化"""
        mag = self.magnitude()
        if mag == 0:
            return Quaternion(0, 0, 0, 1)
        return Quaternion(self.x / mag, self.y / mag, self.z / mag, self.w / mag)

    def conjugate(self) -> 'Quaternion':
        """共轭四元数"""
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def inverse(self) -> 'Quaternion':
        """逆四元数"""
        magSq = self.magnitude() ** 2
        if magSq == 0:
            raise ValueError("无法计算零四元数的逆")
        conj = self.conjugate()
        return Quaternion(conj.x / magSq, conj.y / magSq, conj.z / magSq, conj.w / magSq)

    def rotateVector(self, vector: V3) -> V3:
        """使用四元数旋转三维向量"""
        vecQuat = Quaternion(vector.x, vector.y, vector.z, 0)
        result = self * vecQuat * self.conjugate()
        return V3(result.x, result.y, result.z)

    def toV4(self) -> V4:
        """转换为四维向量"""
        return V4(self.x, self.y, self.z, self.w)

    @staticmethod
    def fromAxisAngle(axis: V3, angle: float) -> 'Quaternion':
        """从轴角创建四元数"""
        normalizedAxis = axis.normalize()
        halfAngle = angle * 0.5
        sinHalfAngle = math.sin(halfAngle)
        cosHalfAngle = math.cos(halfAngle)
        
        return Quaternion(
            normalizedAxis.x * sinHalfAngle,
            normalizedAxis.y * sinHalfAngle,
            normalizedAxis.z * sinHalfAngle,
            cosHalfAngle
        )

    @staticmethod
    def identity() -> 'Quaternion':
        """单位四元数"""
        return Quaternion(0, 0, 0, 1)

    def __str__(self) -> str:
        return f"Quaternion({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.w:.3f})"

    def __repr__(self) -> str:
        return self.__str__()

class Broadcaster(Generic[T]):

    def __init__(self):
        self.queues: list[asyncio.Queue[T]] = []
        self.lock = asyncio.Lock()
        pass

    async def subscribe(self, queue: asyncio.Queue[T]) -> asyncio.Queue[T]:
        async with self.lock:
            logger.debug("Broadcaster.subscribe()")
            self.queues.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[T]) -> None:
        async with self.lock:
            self.queues.remove(queue)

    async def publish(self, item: T) -> None:
        async with self.lock:
            for q in self.queues:
                if q.full():
                    q.get_nowait()
                await q.put(item)

    def publish_nowait(self, item: T) -> None:
        for q in self.queues:
            if q.full():
                q.get_nowait()
            q.put_nowait(item)


class EventBroadcaster:
    events: list[asyncio.Event] = []
    lock = asyncio.Lock()

    async def subscribe(self, event: asyncio.Event) -> asyncio.Event:
        async with self.lock:
            self.events.append(event)
            event.clear()
        return event

    async def unsubscribe(self, event: asyncio.Event) -> None:
        async with self.lock:
            self.events.remove(event)

    async def setEvent(self):
        async with self.lock:
            for event in self.events:
                event.set()

    def setEvent_nowait(self):
        for event in self.events:
            event.set()

    async def clearEvent(self):
        async with self.lock:
            for event in self.events:
                event.clear()


class FFmpeg:
    command: list[str]
    process: asyncio.subprocess.Process = None  # type: ignore

    def __init__(self, command: list[str], logTag: str):
        self.logger = logging.getLogger(logTag)
        self.command = command
        pass

    async def loop(self):

        while True:
            try:
                self.logger.info("正在启动FFmpeg进程...")
                # 启动FFmpeg进程

                self.process = await asyncio.create_subprocess_exec(
                    *self.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                while True:
                    await self.operate()

                    returncode = self.process.returncode
                    if returncode is not None:
                        self.logger.error(
                            f"FFmpeg进程异常退出(代码{returncode})，正在重启..."
                        )
                        raise Exception("FFmpeg进程意外终止")

            except asyncio.CancelledError:
                self.logger.info("任务被取消，执行清理...")
                await self.releaseProcess()
                break

            except Exception as e:
                self.logger.error(f"FFmpeg任务异常: {str(e)}")
                await self.releaseProcess()
                self.logger.info(f"5秒后重启")
                await asyncio.sleep(5)

            pass
        pass

    async def operate(self):
        pass

    async def releaseProcess(self):

        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()
            await self.process.stdin.wait_closed()

            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

            self.process = None  # type: ignore

    pass


class ByteFFmpegPull(FFmpeg):
    handle: Callable[[bytes], Awaitable]
    size: int

    def __init__(
            self,
            command: list[str],
            size: int,
            handle: Callable[[bytes], Awaitable],
            logTag: str,
    ):
        super().__init__(command, logTag)

        self.size = size
        self.handle = handle
        pass

    async def operate(self):
        await self.handle(await self.process.stdout.read(self.size))  # type: ignore
        pass

    pass


class FFmpegPush(FFmpeg):
    getPushDataLambda: Callable[[], Awaitable[bytes | None]]

    def __init__(
            self,
            command: list[str],
            getPushDataLambda: Callable[[], Awaitable[bytes | None]],
            logTag: str,
    ):
        super().__init__(command, logTag)
        self.getPushDataLambda = getPushDataLambda
        pass

    async def operate(self):
        data = await self.getPushDataLambda()

        if data is None:
            await asyncio.sleep(0.1)
            return

        self.process.stdin.write(data)  # type: ignore
        await self.process.stdin.drain()  # type: ignore

    pass


class FFmpegPushFrame(FFmpegPush):
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
        super().__init__(
            [
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
            ],
            self.toByte,
            logTag,
        )

        self.framesQueue = framesQueue

        self.width = width
        self.height = height
        self.fps = fps
        self.pushRtspUrl = pushRtspUrl
        self.framesQueue = framesQueue

        pass

    async def toByte(self) -> bytes | None:
        frame: cv2.typing.MatLike = await self.framesQueue.get()

        # 检查帧尺寸是否匹配预期
        frame_height, frame_width = frame.shape[:2]
        if (frame_width, frame_height) != (self.width, self.height):
            self.logger.warning(
                f"帧尺寸不匹配(期望{self.width}x{self.height}，实际{frame_width}x{frame_height})，正在调整尺寸..."
            )
            try:
                # 使用双线性插值调整尺寸（可根据需求更换插值算法）
                frame = cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            except Exception as resize_err:
                self.logger.error(f"调整帧尺寸失败: {str(resize_err)}，跳过该帧")
                return None
        # 将帧转换为字节流
        return frame.tobytes()


class CircularBuffer:
    """固定大小的循环缓冲区实现"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = bytearray(capacity)
        self.read_pos = 0
        self.write_pos = 0
        self.size = 0

    def write(self, data: bytes) -> int:
        """写入数据，返回实际写入字节数"""
        remaining = self.capacity - self.size
        write_len = min(len(data), remaining)

        # 计算尾部可用空间
        end_space = self.capacity - self.write_pos
        first_part = min(write_len, end_space)
        second_part = write_len - first_part

        # 写入数据
        self.buffer[self.write_pos: self.write_pos + first_part] = data[:first_part]
        if second_part > 0:
            self.buffer[0:second_part] = data[first_part: first_part + second_part]

        self.write_pos = (self.write_pos + write_len) % self.capacity
        self.size += write_len
        return write_len

    def read(self, size: int) -> bytes:
        """读取指定大小的数据，返回空字节表示不足"""
        if self.size < size:
            return b""

        # 计算需要读取的分布
        end_data = self.capacity - self.read_pos
        first_part = min(size, end_data)
        second_part = size - first_part

        # 构建返回数据
        result = bytearray(size)
        result[0:first_part] = self.buffer[self.read_pos: self.read_pos + first_part]
        if second_part > 0:
            result[first_part:size] = self.buffer[0:second_part]

        self.read_pos = (self.read_pos + size) % self.capacity
        self.size -= size
        return bytes(result)

    def available(self) -> int:
        return self.size


def getAllTasks() -> list[asyncio.Task]:
    """
    获取当前事件循环中的所有异步任务，并且排除当前调用的
    """
    loop = asyncio.get_event_loop()
    tasks = asyncio.all_tasks(loop)
    current_task = asyncio.current_task(loop)
    return [t for t in tasks if t is not current_task]


async def gracefulShutdown(tasks: list[asyncio.Task] | None = None) -> None:
    """
    优雅关闭所有异步任务
    """
    tasks = tasks or getAllTasks()
    for task in tasks:
        task.cancel()  # 发送取消信号

    for task in tasks:
        try:
            await task  # 等待任务完成
        except asyncio.CancelledError:
            logging.error(f"Task {task.get_name()} was cancelled.")
        except Exception as e:
            logging.error(f"Task {task.get_name()} raised an exception:", e)


def flattenJson(data, parent_key="", sep=".", list_sep="[{}]") -> dict[str, object]:
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


def getFromJson(key: str, ojson: dict) -> object:
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


def setFromJson(key: str, value: object, ojson: dict) -> None:
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


def splitJsonObjects(json_str):
    """
    将连续的JSON对象字符串分割成独立的JSON对象列表
    示例输入: '{"a":1}{"b":2}' -> ['{"a":1}', '{"b":2}']
    """
    objects = []
    start_index = 0
    brace_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(json_str):
        if in_string:
            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
                escape_next = False
            elif char == '{':
                if brace_count == 0:
                    start_index = i  # 标记对象起始位置
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # 当括号归零时截取完整对象
                    objects.append(json_str[start_index:i + 1])

    return objects


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
    u = frame[height: height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height: height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height: height + height // 2].reshape(height // 2, width)
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
        try:
            image = data.reshape((height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except ValueError:
            print(f"RGB reshape failed: data shape {data.shape}, target ({height}, {width}, 3)")
            return None
    elif color_format == OBFormat.BGR:
        try:
            image = data.reshape((height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except ValueError:
            print(f"BGR reshape failed: data shape {data.shape}, target ({height}, {width}, 3)")
            return None
    elif color_format == OBFormat.YUYV:
        try:
            image = data.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        except ValueError:
            print(f"YUYV reshape failed: data shape {data.shape}, target ({height}, {width}, 2)")
            return None
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
        try:
            image = data.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        except ValueError:
            print(f"UYVY reshape failed: data shape {data.shape}, target ({height}, {width}, 2)")
            return None
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


def isWindows():
    return systemPlatFrom == "Windows"


def isLinux():
    return systemPlatFrom == "Linux"


def fillBuffer(struct_obj: Structure, field_name: str, data):
    buffer = getattr(struct_obj, field_name)
    max_len = len(buffer)
    truncated_data = data[:max_len]
    buffer[: len(truncated_data)] = truncated_data
    if len(data) > max_len:
        logger.warning(f"{field_name} truncated from {len(data)} to {max_len} bytes.")


def fillStr(source: str, values: dict[str, object]) -> str:
    return source.format_map(values)


def HexToDecMa(wHex: int) -> int:
    """
    将16进制数（以整数形式表示）转换为伪十进制形式
    例如：0x1234 -> 1234（整数形式，不是真正的十进制值）
    """
    return (
            (wHex // 4096) * 1000
            + ((wHex % 4096) // 256) * 100
            + ((wHex % 256) // 16) * 10
            + (wHex % 16)
    )


def DEC2HEX_doc(x: int) -> int:
    """
    将伪十进制形式转换回16进制数（整数形式）
    例如：1234 -> 0x1234（返回整数4660）
    """
    return (
            (x // 1000) * 4096
            + ((x % 1000) // 100) * 256
            + ((x % 100) // 10) * 16
            + (x % 10)
    )


def findCrosshair(mat: cv2.typing.MatLike) -> (float, float) | None:
    # 转换为HSV颜色空间（更好识别红色）
    hsv = cv2.cvtColor(mat, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围（两个区间：0°-10° 和 170°-180°）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作（去除噪点，连接断裂部分）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,  # 闭运算：先膨胀后腐蚀
        kernel,
        iterations=3
    )

    # 查找轮廓
    contours, _ = cv2.findContours(
        processed_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # 找到最大轮廓（假设十字准星是最大的红色物体）
    largest_contour = max(contours, key=cv2.contourArea)

    # 计算轮廓的矩并获取中心点
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)


def drawCrosshairCenter(img: cv2.typing.MatLike, center: (float, float), size=30, thickness=3):
    if img is None or center is None:
        return img

    cx, cy = center

    # 绘制醒目的绿色十字标记
    color = (0, 255, 0)  # 绿色
    # 水平线
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    # 垂直线
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)

    # 绘制外圆
    cv2.circle(img, center, size // 2, color, thickness)

    # 绘制中心点
    cv2.circle(img, center, thickness * 2, (0, 0, 255), -1)  # 红色实心点

    # 添加坐标文本
    coord_text = f"({cx}, {cy})"
    cv2.putText(img, coord_text, (cx + size + 5, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, coord_text, (cx + size + 5, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    return img


def brightnessNormalization(img, target_mean=128):
    """
    实现亮度均值化

    参数:
        img: 输入图像 (BGR格式)
        target_mean: 目标亮度均值 (0-255之间, 默认为128)

    返回:
        normalized_img: 亮度均值化后的图像
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算当前图像的亮度均值
    current_mean = np.mean(gray)

    # 计算缩放因子 (避免除以零)
    if current_mean == 0:
        scale = 1.0
    else:
        scale = target_mean / current_mean

    # 将图像转换为浮点类型以便进行数学运算
    img_float = img.astype(np.float32)

    # 应用缩放因子到所有通道
    normalized_img = img_float * scale

    # 确保像素值在0-255范围内
    normalized_img = np.clip(normalized_img, 0, 255)

    # 转换回8位无符号整数
    normalized_img = normalized_img.astype(np.uint8)

    return normalized_img


def eulerToQuaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    将欧拉角转换为四元数
    
    参数:
        roll: 绕X轴的旋转角度（弧度）
        pitch: 绕Y轴的旋转角度（弧度）  
        yaw: 绕Z轴的旋转角度（弧度）
    
    返回:
        tuple: (x, y, z, w) 四元数表示
    """
    # 计算半角的三角函数值
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # 计算四元数分量
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)


def eulerToQuaternionDegrees(rollDeg: float, pitchDeg: float, yawDeg: float) -> Tuple[float, float, float, float]:
    """
    将欧拉角（度数）转换为四元数
    
    参数:
        rollDeg: 绕X轴的旋转角度（度）
        pitchDeg: 绕Y轴的旋转角度（度）
        yawDeg: 绕Z轴的旋转角度（度）
    
    返回:
        tuple: (x, y, z, w) 四元数表示
    """
    # 将度数转换为弧度
    roll = math.radians(rollDeg)
    pitch = math.radians(pitchDeg)
    yaw = math.radians(yawDeg)
    
    return eulerToQuaternion(roll, pitch, yaw)


def quaternionToEuler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    将四元数转换为欧拉角（弧度）
    
    参数:
        x, y, z, w: 四元数分量
    
    返回:
        tuple: (roll, pitch, yaw) 欧拉角（弧度）
    """
    # 计算roll (x轴旋转)
    sinRollCosP = 2 * (w * x + y * z)
    cosRollCosP = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinRollCosP, cosRollCosP)

    # 计算pitch (y轴旋转)
    sinPitch = 2 * (w * y - z * x)
    if abs(sinPitch) >= 1:
        pitch = math.copysign(math.pi / 2, sinPitch)  # 使用90度，避免万向锁
    else:
        pitch = math.asin(sinPitch)

    # 计算yaw (z轴旋转)
    sinYawCosP = 2 * (w * z + x * y)
    cosYawCosP = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(sinYawCosP, cosYawCosP)

    return (roll, pitch, yaw)


def quaternionToEulerDegrees(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    将四元数转换为欧拉角（度数）
    
    参数:
        x, y, z, w: 四元数分量
    
    返回:
        tuple: (rollDeg, pitchDeg, yawDeg) 欧拉角（度）
    """
    roll, pitch, yaw = quaternionToEuler(x, y, z, w)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


# 向量转换函数
def v2ToTuple(v: V2) -> Tuple[float, float]:
    """将 V2 转换为元组"""
    return (v.x, v.y)


def v3ToTuple(v: V3) -> Tuple[float, float, float]:
    """将 V3 转换为元组"""
    return (v.x, v.y, v.z)


def v4ToTuple(v: V4) -> Tuple[float, float, float, float]:
    """将 V4 转换为元组"""
    return (v.x, v.y, v.z, v.w)


def tupleToV2(t: Tuple[float, float]) -> V2:
    """将元组转换为 V2"""
    return V2(t[0], t[1])


def tupleToV3(t: Tuple[float, float, float]) -> V3:
    """将元组转换为 V3"""
    return V3(t[0], t[1], t[2])


def tupleToV4(t: Tuple[float, float, float, float]) -> V4:
    """将元组转换为 V4"""
    return V4(t[0], t[1], t[2], t[3])


def v2ToV3(v: V2, z: float = 0.0) -> V3:
    """将 V2 转换为 V3，添加指定的 z 分量"""
    return V3(v.x, v.y, z)


def v3ToV4(v: V3, w: float = 0.0) -> V4:
    """将 V3 转换为 V4，添加指定的 w 分量"""
    return V4(v.x, v.y, v.z, w)


def v2ToV4(v: V2, z: float = 0.0, w: float = 0.0) -> V4:
    """将 V2 转换为 V4，添加指定的 z 和 w 分量"""
    return V4(v.x, v.y, z, w)


# 四元数转换函数
def quaternionToTuple(q: Quaternion) -> Tuple[float, float, float, float]:
    """将四元数转换为元组 (x, y, z, w)"""
    return (q.x, q.y, q.z, q.w)


def tupleToQuaternion(t: Tuple[float, float, float, float]) -> Quaternion:
    """将元组 (x, y, z, w) 转换为四元数"""
    return Quaternion(t[0], t[1], t[2], t[3])


def quaternionToV4(q: Quaternion) -> V4:
    """将四元数转换为 V4"""
    return q.toV4()


def v4ToQuaternion(v: V4) -> Quaternion:
    """将 V4 转换为四元数"""
    return Quaternion(v.x, v.y, v.z, v.w)


# 角度转换函数
def degreesToRadians(degrees: float) -> float:
    """将度数转换为弧度"""
    return math.radians(degrees)


def radiansToDegrees(radians: float) -> float:
    """将弧度转换为度数"""
    return math.degrees(radians)


def normalizeAngle(angle: float) -> float:
    """将角度标准化到 [-π, π] 范围内（弧度）"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def normalizeAngleDegrees(angle: float) -> float:
    """将角度标准化到 [-180, 180] 范围内（度数）"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


# 向量操作函数
def lerp(start: float, end: float, t: float) -> float:
    """线性插值"""
    return start + (end - start) * t


def lerpV2(start: V2, end: V2, t: float) -> V2:
    """V2 线性插值"""
    return V2(
        lerp(start.x, end.x, t),
        lerp(start.y, end.y, t)
    )


def lerpV3(start: V3, end: V3, t: float) -> V3:
    """V3 线性插值"""
    return V3(
        lerp(start.x, end.x, t),
        lerp(start.y, end.y, t),
        lerp(start.z, end.z, t)
    )


def slerpQuaternion(start: Quaternion, end: Quaternion, t: float) -> Quaternion:
    """四元数球面线性插值"""
    dot = start.x * end.x + start.y * end.y + start.z * end.z + start.w * end.w
    
    # 如果点积为负，取反其中一个四元数以选择更短的路径
    if dot < 0.0:
        end = Quaternion(-end.x, -end.y, -end.z, -end.w)
        dot = -dot
    
    # 如果四元数非常接近，使用线性插值
    if dot > 0.9995:
        result = Quaternion(
            start.x + t * (end.x - start.x),
            start.y + t * (end.y - start.y),
            start.z + t * (end.z - start.z),
            start.w + t * (end.w - start.w)
        )
        return result.normalize()
    
    # 计算球面插值
    theta0 = math.acos(abs(dot))
    sinTheta0 = math.sin(theta0)
    theta = theta0 * t
    sinTheta = math.sin(theta)
    
    s0 = math.cos(theta) - dot * sinTheta / sinTheta0
    s1 = sinTheta / sinTheta0
    
    return Quaternion(
        s0 * start.x + s1 * end.x,
        s0 * start.y + s1 * end.y,
        s0 * start.z + s1 * end.z,
        s0 * start.w + s1 * end.w
    )


def clamp(value: float, minVal: float, maxVal: float) -> float:
    """将值限制在指定范围内"""
    return max(minVal, min(maxVal, value))


def clampV2(v: V2, minVal: V2, maxVal: V2) -> V2:
    """将 V2 的各分量限制在指定范围内"""
    return V2(
        clamp(v.x, minVal.x, maxVal.x),
        clamp(v.y, minVal.y, maxVal.y)
    )


def clampV3(v: V3, minVal: V3, maxVal: V3) -> V3:
    """将 V3 的各分量限制在指定范围内"""
    return V3(
        clamp(v.x, minVal.x, maxVal.x),
        clamp(v.y, minVal.y, maxVal.y),
        clamp(v.z, minVal.z, maxVal.z)
    )