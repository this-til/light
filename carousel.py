import pygame
import asyncio
import os
import glob
from pathlib import Path
import time
from typing import List
from main import Component, ConfigField

class Carousel(Component):
    enable: ConfigField[bool] = ConfigField(False)
    images_folder: ConfigField[str] = ConfigField("images")
    interval: ConfigField[float] = ConfigField(3.0)  # 轮播间隔（秒）
    fullscreen: ConfigField[bool] = ConfigField(True)
    width: ConfigField[int] = ConfigField(1920)
    height: ConfigField[int] = ConfigField(1080)
    fade_duration: ConfigField[float] = ConfigField(0.5)  # 淡入淡出时间
    
    def __init__(self):
        super().__init__()
        self.screen = None
        self.clock = None
        self.images: List[pygame.Surface] = []
        self.current_image_index = 0
        self.last_switch_time = 0
        self.running = False
        self.fade_alpha = 255
        
    async def init(self):
        """初始化PyGame和图像轮播"""
        await super().init()
        
        if not self.enable:
            self.logger.info("图像轮播组件未启用")
            return
            
        try:
            # 初始化PyGame
            pygame.init()
            pygame.display.init()
            
            # 设置显示模式
            if self.fullscreen:
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                self.logger.info(f"设置全屏模式：{self.screen.get_size()}")
            else:
                self.screen = pygame.display.set_mode((self.width, self.height))
                self.logger.info(f"设置窗口模式：{self.width}x{self.height}")
                
            pygame.display.set_caption("图像轮播")
            self.clock = pygame.time.Clock()
            
            # 加载图像
            await self.load_images()
            
            if self.images:
                self.running = True
                # 启动轮播循环
                asyncio.create_task(self.carousel_loop())
                self.logger.info(f"图像轮播已启动，共加载 {len(self.images)} 张图片")
            else:
                self.logger.warning("未找到图像文件，轮播未启动")
                
        except Exception as e:
            self.logger.error(f"初始化图像轮播失败: {e}")
            
    async def load_images(self):
        """加载图像文件"""
        self.images.clear()
        
        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        
        images_path = Path(self.images_folder)
        if not images_path.exists():
            # 如果文件夹不存在，创建一个默认的
            images_path.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"图像文件夹 {self.images_folder} 不存在，已创建")
            return
            
        # 查找所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(images_path / ext)))
            image_files.extend(glob.glob(str(images_path / ext.upper())))
            
        if not image_files:
            self.logger.warning(f"在 {self.images_folder} 中未找到图像文件")
            return
            
        # 加载并缩放图像
        screen_size = self.screen.get_size()
        
        for image_file in sorted(image_files):
            try:
                # 加载图像
                image = pygame.image.load(image_file)
                
                # 缩放图像以适应屏幕，保持宽高比
                image = self.scale_image_to_screen(image, screen_size)
                
                self.images.append(image)
                self.logger.debug(f"已加载图像: {os.path.basename(image_file)}")
                
            except Exception as e:
                self.logger.error(f"加载图像 {image_file} 失败: {e}")
                
    def scale_image_to_screen(self, image: pygame.Surface, screen_size: tuple) -> pygame.Surface:
        """缩放图像以适应屏幕，保持宽高比"""
        screen_width, screen_height = screen_size
        image_width, image_height = image.get_size()
        
        # 计算缩放比例
        scale_x = screen_width / image_width
        scale_y = screen_height / image_height
        scale = min(scale_x, scale_y)  # 保持宽高比
        
        # 计算新尺寸
        new_width = int(image_width * scale)
        new_height = int(image_height * scale)
        
        # 缩放图像
        scaled_image = pygame.transform.scale(image, (new_width, new_height))
        
        # 创建居中的Surface
        centered_surface = pygame.Surface(screen_size)
        centered_surface.fill((0, 0, 0))  # 黑色背景
        
        # 计算居中位置
        x = (screen_width - new_width) // 2
        y = (screen_height - new_height) // 2
        
        centered_surface.blit(scaled_image, (x, y))
        
        return centered_surface
        
    async def carousel_loop(self):
        """主轮播循环"""
        self.last_switch_time = time.time()
        
        while self.running:
            try:
                # 处理PyGame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            # 空格键手动切换到下一张
                            self.next_image()
                        elif event.key == pygame.K_LEFT:
                            # 左箭头键切换到上一张
                            self.previous_image()
                        elif event.key == pygame.K_RIGHT:
                            # 右箭头键切换到下一张
                            self.next_image()
                            
                # 检查是否需要自动切换图像
                current_time = time.time()
                if current_time - self.last_switch_time >= self.interval:
                    self.next_image()
                    self.last_switch_time = current_time
                    
                # 绘制当前图像
                if self.images:
                    self.draw_current_image()
                    
                # 限制帧率
                self.clock.tick(60)
                
                # 让出控制权
                await asyncio.sleep(0.016)  # ~60 FPS
                
            except asyncio.CancelledError:
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"轮播循环错误: {e}")
                await asyncio.sleep(1)
                
    def draw_current_image(self):
        """绘制当前图像"""
        if not self.images:
            return
            
        # 清屏
        self.screen.fill((0, 0, 0))
        
        # 绘制当前图像
        current_image = self.images[self.current_image_index]
        self.screen.blit(current_image, (0, 0))
        
        # 更新显示
        pygame.display.flip()
        
    def next_image(self):
        """切换到下一张图像"""
        if self.images:
            self.current_image_index = (self.current_image_index + 1) % len(self.images)
            self.logger.debug(f"切换到图像 {self.current_image_index + 1}/{len(self.images)}")
            
    def previous_image(self):
        """切换到上一张图像"""
        if self.images:
            self.current_image_index = (self.current_image_index - 1) % len(self.images)
            self.logger.debug(f"切换到图像 {self.current_image_index + 1}/{len(self.images)}")
            
    async def release(self):
        """释放资源"""
        await super().release()
        
        self.running = False
        
        if pygame.get_init():
            pygame.quit()
            self.logger.info("PyGame已退出")
            
    def getPriority(self) -> int:
        """设置组件优先级"""
        return 10  # 较低优先级，在其他组件之后初始化