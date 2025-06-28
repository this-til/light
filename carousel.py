import pygame
import asyncio
import os
import glob
from pathlib import Path
import time
from typing import List
from main import Component, ConfigField

class CarouselComponent(Component):
    enable: ConfigField[bool] = ConfigField(False)
    images_folder: ConfigField[str] = ConfigField("images")
    interval: ConfigField[float] = ConfigField(3.0)  # 轮播间隔（秒）
    fullscreen: ConfigField[bool] = ConfigField(True)
    width: ConfigField[int] = ConfigField(400)
    height: ConfigField[int] = ConfigField(800)
    animation_duration: ConfigField[float] = ConfigField(0.5)  # 动画持续时间（秒）

    def __init__(self):
        super().__init__()
        self.screen = None
        self.clock = None
        self.images: List[pygame.Surface] = []
        self.current_image_index = 0
        self.last_switch_time = 0
        self.running = False
        self.fade_alpha = 255
        
        # 动画相关
        self.is_animating = False
        self.animation_progress = 0.0
        self.animation_start_time = 0
        self.next_image_index = 0
        self.animation_direction = 1  # 1为向右，-1为向左
        
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
                self.logger.info(f"配置信息 - 间隔: {self.interval}s, 动画时长: {self.animation_duration}s")
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
        """缩放图像以铺满屏幕，拉伸到屏幕大小"""
        screen_width, screen_height = screen_size
        
        # 直接将图像拉伸到屏幕大小，不保持宽高比
        scaled_image = pygame.transform.scale(image, (screen_width, screen_height))
        
        return scaled_image
        
    async def carousel_loop(self):
        """主轮播循环"""
        self.last_switch_time = time.time()
        
        while self.running:
            try:
                # 更新动画
                current_time = time.time()
                if self.is_animating:
                    self.update_animation(current_time)
                    
                # 检查是否需要自动切换图像
                interval = self.interval if self.interval and self.interval > 0 else 3.0
                if not self.is_animating and current_time - self.last_switch_time >= interval:
                    self.start_transition_to_next()
                    self.last_switch_time = current_time

                # 绘制当前图像
                if self.images:
                    if self.is_animating:
                        self.draw_transition()
                    else:
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
        
    def draw_transition(self):
        """绘制过渡动画"""
        if not self.images:
            return
            
        # 清屏
        self.screen.fill((0, 0, 0))
        
        screen_width = self.screen.get_width()
        
        # 使用缓动函数（ease-in-out）
        eased_progress = self.ease_in_out(self.animation_progress)
        
        # 计算偏移量
        offset = int(screen_width * eased_progress)
        
        # 当前图像位置（向左或向右移出）
        current_x = -offset * self.animation_direction
        
        # 下一张图像位置（从右或从左飞入）
        next_x = screen_width * self.animation_direction - offset * self.animation_direction
        
        # 绘制当前图像
        current_image = self.images[self.current_image_index]
        self.screen.blit(current_image, (current_x, 0))
        
        # 绘制下一张图像
        next_image = self.images[self.next_image_index]
        self.screen.blit(next_image, (next_x, 0))
        
        # 更新显示
        pygame.display.flip()
        
    def ease_in_out(self, t):
        """缓动函数，提供平滑的加速和减速效果"""
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t
            
    def update_animation(self, current_time):
        """更新动画进度"""
        elapsed = current_time - self.animation_start_time
        
        # 安全检查动画持续时间
        duration = self.animation_duration if self.animation_duration and self.animation_duration > 0 else 0.5
        
        self.animation_progress = min(elapsed / duration, 1.0)
        
        if self.animation_progress >= 1.0:
            # 动画完成
            self.is_animating = False
            self.current_image_index = self.next_image_index
            self.animation_progress = 0.0
            
    def start_transition_to_next(self):
        """开始向下一张图片的过渡动画"""
        if not self.images or self.is_animating:
            return
            
        self.next_image_index = (self.current_image_index + 1) % len(self.images)
        self.animation_direction = 1  # 向右
        self.is_animating = True
        self.animation_start_time = time.time()
        self.animation_progress = 0.0
        self.logger.debug(f"开始向右切换到图像 {self.next_image_index + 1}/{len(self.images)}")
        
    def start_transition_to_previous(self):
        """开始向上一张图片的过渡动画"""
        if not self.images or self.is_animating:
            return
            
        self.next_image_index = (self.current_image_index - 1) % len(self.images)
        self.animation_direction = -1  # 向左
        self.is_animating = True
        self.animation_start_time = time.time()
        self.animation_progress = 0.0
        self.logger.debug(f"开始向左切换到图像 {self.next_image_index + 1}/{len(self.images)}")
        

            
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