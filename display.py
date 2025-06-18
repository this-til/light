import asyncio
import main
import os
import webbrowser
import time
import subprocess
import sys
import psutil
from urllib.parse import urlparse

from main import Component, ConfigField

class DisplayComponent(Component):
    targetUrl : ConfigField[str] = ConfigField()

    async def init(self):
        await super().init()
        asyncio.create_task(self.loop())

    def is_browser_running_with_url(self, url):
        """检查是否有浏览器进程已打开指定URL"""
        # 获取域名用于检查
        domain = urlparse(url).netloc

        # 检查常见浏览器进程
        browsers = [
            "chrome", "chromium", "firefox", "brave", "opera",
            "vivaldi", "epiphany", "konqueror", "edge", "waterfox"
        ]

        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                proc_name = proc.info['name'].lower()
                cmdline = proc.info['cmdline'] or []

                # 检查是否是浏览器进程
                if any(browser in proc_name for browser in browsers):
                    # 检查命令行参数中是否包含目标URL或域名
                    if any(url in arg or domain in arg for arg in cmdline):
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return False

    def open_url_fullscreen(self, url):
        """打开URL并全屏显示（如果尚未打开）"""
        # 检查浏览器是否已打开该URL
        if self.is_browser_running_with_url(url):
            print(f"浏览器已打开指定URL: {url}，无需操作")
            return

        # 检查是否安装了xdotool
        if not any(os.access(os.path.join(path, "xdotool"), os.X_OK)
                   for path in os.environ["PATH"].split(os.pathsep)):
            print("未找到xdotool，请先安装：sudo apt install xdotool")
            return

        # 使用默认浏览器打开URL
        webbrowser.open(url)
        print(f"正在打开: {url}")

        # 等待浏览器加载
        time.sleep(3)

        try:
            # 查找最新创建的浏览器窗口
            window_id = subprocess.check_output(
                ["xdotool", "search", "--sync", "--onlyvisible", "--class", "browser"]
            ).decode().split()[-1].strip()

            # 激活窗口并全屏
            subprocess.call(["xdotool", "windowactivate", window_id])
            subprocess.call(["xdotool", "key", "F11"])
            print("浏览器已全屏显示")
        except Exception as e:
            print(f"全屏操作失败: {e}")
            print("请尝试手动按F11全屏")

    async def loop(self):

        while True:
            try:

                self.open_url_fullscreen(self.targetUrl)

                await asyncio.sleep(10)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Error in displayLoop: {e}")

