#!/usr/bin/python3

import logging
import asyncio
import numpy as np
import cv2
import detection
import time
from asyncio.subprocess import PIPE
from util import Broadcaster

logger = logging.getLogger(__name__)

ip = "192.168.117.100"
rtsp_port = "554"
user = "admin"
password = "qWERTYUIOP"

width, height = 2560, 1440
fps = 25

cameraRtspUrl = f"rtsp://{user}:{password}@{ip}:{rtsp_port}/Streaming/Channels/101"

source: Broadcaster[cv2.typing.MatLike] = Broadcaster()
out: Broadcaster[cv2.typing.MatLike] = Broadcaster()

cap: cv2.VideoCapture = None

pushRtspUrl = "rtsp://localhost:8554/channels001"

pushProcess: asyncio.subprocess.Process = None


async def releaseCap():
    global cap
    if cap and cap.isOpened():
        await asyncio.get_event_loop().run_in_executor(None, cap.release)
    cap = None
    pass


async def releaseProcess():
    global pushProcess

    if pushProcess is None:
        return

    pushProcess.stdin.close()

    if pushProcess.stdin.is_closing():
        return

    await pushProcess.stdin.wait_closed()

    try:
        await asyncio.wait_for(pushProcess.wait(), timeout=2)
    except asyncio.TimeoutError:
        pushProcess.kill()
        await pushProcess.wait()

    pushProcess = None

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
                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    None, cap.read
                )

                if not ret:
                    logger.error("读取帧失败，释放资源并重连...")
                    raise Exception("读取帧失败...")

                h, w = frame.shape[:2]
                if w < h:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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


async def renderFrames():
    while True:
        try:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"渲染帧时发生异常: {str(e)}")
        pass
    pass


async def pushFrames():
    global pushProcess  # 如果需要访问全局变量，记得声明
    command = [
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

    framesQueue: asyncio.Queue[cv2.typing.MatLike] = asyncio.Queue(maxsize=16)
    await out.subscribe(framesQueue)

    while True:  # 无限循环尝试重启FFmpeg进程
        try:
            logger.info("正在启动FFmpeg推流进程...")
            # 启动FFmpeg进程
            pushProcess = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # 创建一个任务来监控进程的输出和错误
            stderr_reader = asyncio.create_task(pushProcess.stderr.read())
            stdout_reader = asyncio.create_task(pushProcess.stdout.read())

            while True:  # 主循环处理帧
                frame = await framesQueue.get()

                # 检查帧尺寸是否匹配预期
                frame_height, frame_width = frame.shape[:2]
                if (frame_width, frame_height) != (width, height):
                    logger.warning(
                        f"帧尺寸不匹配(期望{width}x{height}，实际{frame_width}x{frame_height})，正在调整尺寸..."
                    )
                    try:
                        # 使用双线性插值调整尺寸（可根据需求更换插值算法）
                        frame = cv2.resize(
                            frame, (width, height), interpolation=cv2.INTER_LINEAR
                        )
                    except Exception as resize_err:
                        logger.error(f"调整帧尺寸失败: {str(resize_err)}，跳过该帧")
                        continue

                # 将帧转换为字节流
                data = frame.tobytes()

                # 写入FFmpeg进程的输入管道
                try:
                    pushProcess.stdin.write(data)
                    await pushProcess.stdin.drain()  # 确保数据已写入
                except (BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"写入FFmpeg进程失败: {e}")
                    raise  # 触发外层异常处理重新启动进程

                # 检查FFmpeg进程是否仍在运行
                returncode = pushProcess.returncode
                if returncode is not None:
                    logger.error(f"FFmpeg进程异常退出(代码{returncode})，正在重启...")
                    raise Exception("FFmpeg进程意外终止")

        except asyncio.CancelledError:
            logger.info("推流任务被取消，执行清理...")
            await releaseProcess()
            break

        except Exception as e:
            logger.error(f"推流任务异常: {str(e)}")
            # 清理资源
            await releaseProcess()
            # 读取并记录FFmpeg的错误输出
            if not stderr_reader.done():
                stderr = await stderr_reader
                logger.error(f"FFmpeg错误输出: {stderr.decode()}")
            if not stdout_reader.done():
                stdout = await stdout_reader
                logger.debug(f"FFmpeg标准输出: {stdout.decode()}")
            # 重试前等待一段时间
            await asyncio.sleep(5)


async def handleFrames():

    framesQueue: asyncio.Queue[cv2.typing.MatLike] = asyncio.Queue(maxsize=16)
    await source.subscribe(framesQueue)

    
    while True:

        try:

            sourceFrame = await framesQueue.get()
            
            start_time = time.perf_counter()
            
            result = detection.runDetection(sourceFrame, [detection.fall_down_model])
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            logger.info(f"runDetection 耗时: {duration_ms:.3f}ms")
            await out.publish(result.drawOutputImage())
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"处理帧时发生异常: {str(e)}")
        pass


async def initCamera():
    logger.info("初始化摄像头模块")
    asyncio.create_task(readFrames())
    asyncio.create_task(handleFrames())
    asyncio.create_task(pushFrames())


async def releaseCamera():
    logger.info("释放摄像头资源")
    await releaseCap()
    await releaseProcess()
