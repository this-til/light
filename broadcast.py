import pygame as pg
import os
import asyncio
from typing import Dict, Optional
from pathlib import Path

from main import Component, ConfigField

class BroadcastComponent(Component):
    
    soundSource : ConfigField[str] = ConfigField()
    
    # 音频文件表
    audioFiles: Dict[str, str] = {}
    
    # 当前播放任务
    currentPlayTask: Optional[asyncio.Task] = None
    
    async def awakeInit(self):
        await super().awakeInit()
        pg.mixer.init()
        
        # 扫描音频文件并生成表
        await self.scanAudioFiles()
        
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
                await asyncio.sleep(0.1)
                
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