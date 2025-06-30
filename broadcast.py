import pygame as pg
import os
import asyncio
import random
from typing import Dict, Optional
from pathlib import Path

from command import CommandEvent
from main import Component, ConfigField


class BroadcastComponent(Component):
    soundSource: ConfigField[str] = ConfigField()

    # 音频文件表
    audioFiles: Dict[str, str] = {}

    # 当前播放任务
    currentPlayTask: Optional[asyncio.Task] = None

    async def awakeInit(self):
        await super().awakeInit()
        pg.mixer.init()

        # 扫描音频文件并生成表
        await self.scanAudioFiles()
        
        # 启动指令循环
        asyncio.create_task(self.instructionLoop())
        asyncio.create_task(self.playLoop())

    async def scanAudioFiles(self):
        """扫描soundSource目录下的音频文件并生成文件表"""
        if not self.soundSource:
            self.logger.warning("soundSource 配置为空")
            return

        soundPath = Path(self.soundSource)
        if not soundPath.exists():
            self.logger.warning(f"音频目录不存在: {self.soundSource}")
            return

        # 支持的音频格式
        audioExtensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac'}

        self.audioFiles.clear()

        if soundPath.is_file():
            # 如果是单个文件
            if soundPath.suffix.lower() in audioExtensions:
                fileName = soundPath.stem
                self.audioFiles[fileName] = str(soundPath)
                self.logger.info(f"加载音频文件: {fileName}")
        elif soundPath.is_dir():
            # 如果是目录，扫描所有音频文件
            for file in soundPath.iterdir():
                if file.is_file() and file.suffix.lower() in audioExtensions:
                    fileName = file.stem
                    self.audioFiles[fileName] = str(file)
                    self.logger.info(f"发现音频文件: {fileName}")

        self.logger.info(f"总共加载了 {len(self.audioFiles)} 个音频文件")

    async def playAudio(self, fileName: str, loop: int = 0) -> bool:
        """
        异步播放音频文件
        
        Args:
            fileName: 音频文件名（不包含扩展名）
            loop: 循环次数，0表示播放一次，-1表示无限循环
            
        Returns:
            bool: True表示播放完成，False表示被取消或出错
        """
        if fileName not in self.audioFiles:
            self.logger.error(f"音频文件不存在: {fileName}")
            return False

        filePath = self.audioFiles[fileName]

        try:
            # 停止当前播放
            await self.stopAudio()

            # 加载并播放音频
            pg.mixer.music.load(filePath)
            pg.mixer.music.play(loops=loop)

            self.logger.info(f"开始播放音频: {fileName}")

            # 异步等待播放完成
            while pg.mixer.music.get_busy():
                await asyncio.sleep(0.5)

            self.logger.info(f"音频播放完成: {fileName}")
            return True

        except asyncio.CancelledError:
            self.logger.info(f"音频播放被取消: {fileName}")
            pg.mixer.music.stop()
            return False
        except Exception as e:
            self.logger.exception(f"播放音频时发生错误: {str(e)}")
            return False

    async def playAudioAsync(self, fileName: str, loop: int = 0) -> asyncio.Task:
        """
        创建异步播放任务
        
        Args:
            fileName: 音频文件名
            loop: 循环次数
            
        Returns:
            asyncio.Task: 播放任务，可以用于取消
        """
        # 取消当前播放任务
        if self.currentPlayTask and not self.currentPlayTask.done():
            self.currentPlayTask.cancel()

        # 创建新的播放任务
        self.currentPlayTask = asyncio.create_task(self.playAudio(fileName, loop))
        return self.currentPlayTask

    async def stopAudio(self):
        """停止当前音频播放"""
        if self.currentPlayTask and not self.currentPlayTask.done():
            self.currentPlayTask.cancel()
            try:
                await self.currentPlayTask
            except asyncio.CancelledError:
                pass

        if pg.mixer.music.get_busy():
            pg.mixer.music.stop()

    def getAudioList(self) -> Dict[str, str]:
        """获取音频文件列表"""
        return self.audioFiles.copy()

    def hasAudio(self, fileName: str) -> bool:
        """检查是否存在指定名称的音频文件"""
        return fileName in self.audioFiles

    async def playLoop(self):
        queue: asyncio.Queue[CommandEvent] = await self.main.commandComponent.commandEvent.subscribe(asyncio.Queue(maxsize=16))

        while True:
            try:
                commandEvent = await queue.get()

                if commandEvent.key == "Broadcast.File":
                    if commandEvent.value in self.audioFiles:
                        asyncio.create_task(self.playAudioAsync(commandEvent.value))

                if commandEvent.key == "Broadcast.Stop":
                    await self.stopAudio()


            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"播放循环异常:{e} ")
                await asyncio.sleep(5)

    async def _playRandomSequence(self, audio_sequence: list):
        """连续播放随机音频序列"""
        try:
            for i, audio_name in enumerate(audio_sequence, 1):
                self.logger.info(f"连续播放第{i}/{len(audio_sequence)}个: {audio_name}")
                success = await self.playAudio(audio_name)
                if not success:
                    self.logger.warning(f"播放失败，跳过: {audio_name}")
                    continue
                # 短暂停顿
                await asyncio.sleep(0.5)
            self.logger.info("连续随机播放序列完成")
        except asyncio.CancelledError:
            self.logger.info("连续随机播放被取消")
            await self.stopAudio()
        except Exception as e:
            self.logger.exception(f"连续随机播放异常: {str(e)}")

    async def instructionLoop(self):
        """指令循环，处理键盘事件触发的音频播放命令"""
        queue = await self.main.keyComponent.keyEvent.subscribe(asyncio.Queue(maxsize=1))

        while True:
            try:
                key = await queue.get()

                # 播放指定音频文件
                if key.startswith("play:"):
                    audio_name = key.split(":", 1)[1]
                    if self.hasAudio(audio_name):
                        await self.playAudioAsync(audio_name)
                        self.logger.info(f"通过指令播放音频: {audio_name}")
                    else:
                        self.logger.warning(f"指定的音频文件不存在: {audio_name}")

                # 播放并循环指定音频文件
                elif key.startswith("playLoop:"):
                    audio_name = key.split(":", 1)[1]
                    if self.hasAudio(audio_name):
                        await self.playAudioAsync(audio_name, loop=-1)
                        self.logger.info(f"通过指令循环播放音频: {audio_name}")
                    else:
                        self.logger.warning(f"指定的音频文件不存在: {audio_name}")

                # 停止当前播放
                elif key == "stopAudio":
                    await self.stopAudio()
                    self.logger.info("通过指令停止音频播放")

                # 列出所有可用音频文件
                elif key == "listAudio":
                    audio_list = list(self.audioFiles.keys())
                    if audio_list:
                        self.logger.info(f"可用音频文件: {', '.join(audio_list)}")
                    else:
                        self.logger.info("没有可用的音频文件")

                # 重新扫描音频文件
                elif key == "rescanAudio":
                    await self.scanAudioFiles()
                    self.logger.info("重新扫描音频文件完成")

                # 随机播放一个音频文件
                elif key == "playRandom":
                    if self.audioFiles:
                        random_audio = random.choice(list(self.audioFiles.keys()))
                        await self.playAudioAsync(random_audio)
                        self.logger.info(f"随机播放音频: {random_audio}")
                    else:
                        self.logger.warning("没有可用的音频文件进行随机播放")

                # 随机循环播放一个音频文件
                elif key == "playRandomLoop":
                    if self.audioFiles:
                        random_audio = random.choice(list(self.audioFiles.keys()))
                        await self.playAudioAsync(random_audio, loop=-1)
                        self.logger.info(f"随机循环播放音频: {random_audio}")
                    else:
                        self.logger.warning("没有可用的音频文件进行随机循环播放")

                # 连续随机播放指定数量的音频文件
                elif key.startswith("playRandomSequence:"):
                    try:
                        count_str = key.split(":", 1)[1]
                        count = int(count_str)
                        if self.audioFiles and count > 0:
                            audio_list = list(self.audioFiles.keys())
                            selected_audios = random.choices(audio_list, k=min(count, len(audio_list)))
                            asyncio.create_task(self._playRandomSequence(selected_audios))
                            self.logger.info(f"开始连续随机播放 {len(selected_audios)} 个音频: {', '.join(selected_audios)}")
                        else:
                            self.logger.warning(f"无法进行连续随机播放: 音频文件数={len(self.audioFiles)}, 请求数量={count}")
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"无效的连续随机播放格式: {key}, 错误: {str(e)}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.exception(f"instructionLoop Exception: {str(e)}")
