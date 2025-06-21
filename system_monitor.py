#!/usr/bin/env python3
"""
系统监控脚本 - 用于诊断分段错误
监控内存使用、线程数量、GPU状态等信息
"""

import psutil
import threading
import time
import logging
import asyncio
from datetime import datetime


class SystemMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        self.logger = logging.getLogger("SystemMonitor")
        
    def start_monitoring(self):
        """启动监控"""
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("系统监控已启动")
        
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        self.logger.info("系统监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self._log_system_info()
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"监控错误: {e}")
                
    def _log_system_info(self):
        """记录系统信息"""
        try:
            # 内存信息
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 进程信息
            process = psutil.Process()
            process_memory = process.memory_info()
            num_threads = process.num_threads()
            
            # 线程信息
            thread_count = threading.active_count()
            
            self.logger.info(f"""
=== 系统状态 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===
内存使用: {memory.percent:.1f}% ({memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB)
Swap使用: {swap.percent:.1f}% ({swap.used / 1024**3:.2f}GB / {swap.total / 1024**3:.2f}GB)
CPU使用: {cpu_percent:.1f}%
进程内存: RSS={process_memory.rss / 1024**2:.2f}MB, VMS={process_memory.vms / 1024**2:.2f}MB
进程线程数: {num_threads}
Python线程数: {thread_count}
=======================================
            """)
            
            # 检查内存警告
            if memory.percent > 90:
                self.logger.warning("⚠️  内存使用率超过90%!")
                
            if process_memory.rss / 1024**3 > 2:  # 超过2GB
                self.logger.warning("⚠️  进程内存使用超过2GB!")
                
            if num_threads > 50:
                self.logger.warning("⚠️  线程数量过多!")
                
        except Exception as e:
            self.logger.error(f"获取系统信息失败: {e}")


def setup_crash_handling():
    """设置崩溃处理"""
    import signal
    import sys
    import traceback
    
    def signal_handler(signum, frame):
        print(f"\n💥 收到信号 {signum}, 正在生成崩溃报告...")
        
        # 打印堆栈信息
        print("=== 堆栈跟踪 ===")
        traceback.print_stack(frame)
        
        # 打印线程信息
        print("\n=== 活跃线程 ===")
        for thread_id, frame in sys._current_frames().items():
            print(f"Thread {thread_id}:")
            traceback.print_stack(frame)
            print()
            
        sys.exit(1)
    
    # 注册信号处理器
    signal.signal(signal.SIGSEGV, signal_handler)  # 分段错误
    signal.signal(signal.SIGABRT, signal_handler)  # 中止信号
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置崩溃处理
    setup_crash_handling()
    
    # 启动监控
    monitor = SystemMonitor(interval=10)
    monitor.start_monitoring()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("监控已停止") 