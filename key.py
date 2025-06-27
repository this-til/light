import sys
import select
import tty
import termios
import asyncio
import signal
import os
from collections import deque
from typing import AsyncGenerator, Callable, Optional

import util
from main import Component, ConfigField


# 全局输入状态管理
class InputStateManager:
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.is_inputting = False
            cls._instance.pending_logs = []
            cls._instance.input_component = None
            cls._instance.log_handler = None
        return cls._instance
    
    def set_log_handler(self, handler):
        """设置日志处理器引用"""
        self.log_handler = handler
        if handler:
            handler.set_input_component(self.input_component)
    
    async def set_inputting(self, state: bool):
        async with self._lock:
            self.is_inputting = state
            
            # 同步状态到日志处理器
            if self.log_handler:
                self.log_handler.set_input_state(state)
            
            if not state and self.pending_logs:
                # 输入结束，输出缓存的日志
                for log_msg in self.pending_logs:
                    print(log_msg)
                self.pending_logs.clear()
                # 重新显示提示符
                if self.input_component:
                    self.input_component.display_prompt_and_input()
    
    async def add_pending_log(self, log_msg: str):
        async with self._lock:
            if self.is_inputting:
                self.pending_logs.append(log_msg)
                return True  # 表示日志被缓存
            return False  # 表示可以直接输出
    
    def set_input_component(self, component):
        self.input_component = component
        # 如果已经有日志处理器，也更新它的组件引用
        if self.log_handler:
            self.log_handler.set_input_component(component)


# 全局状态管理器实例
input_state_manager = InputStateManager()


# 尝试导入全局日志处理器
try:
    from ros_access import global_log_handler
    input_state_manager.set_log_handler(global_log_handler)
except ImportError:
    # 如果无法导入，创建一个临时的占位符
    pass


class KeyComponent(Component):
    keyEvent: util.Broadcaster[str] = util.Broadcaster()
    
    def __init__(self):
        super().__init__()
        # 命令历史记录
        self.command_history: deque[str] = deque(maxlen=100)  # 最多保存100条历史
        self.history_index: int = -1  # 当前历史索引，-1表示最新
        self.current_input: str = ""  # 当前输入缓冲
        self.cursor_position: int = 0  # 光标位置
        
        # 终端设置
        self.old_settings = None
        self.terminal_width = 80  # 终端宽度，默认80
        
        # 加载历史记录
        self.load_command_history()

    async def init(self):
        await super().init()
        
        # 注册到输入状态管理器
        input_state_manager.set_input_component(self)
        
        # 尝试重新设置日志处理器连接
        try:
            from ros_access import global_log_handler
            input_state_manager.set_log_handler(global_log_handler)
        except ImportError:
            self.logger.warning("无法连接到全局日志处理器")
        
        # 获取终端宽度
        try:
            self.terminal_width = os.get_terminal_size().columns
        except OSError:
            self.terminal_width = 80
            
        self.logger.info(f"终端宽度: {self.terminal_width}")
        
        asyncio.create_task(self.keyboardListener())

    def load_command_history(self):
        """从文件加载命令历史"""
        try:
            history_file = "command_history.txt"
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        command = line.strip()
                        if command:
                            self.command_history.append(command)
                self.logger.info(f"已加载 {len(self.command_history)} 条历史命令")
        except Exception as e:
            self.logger.warning(f"加载命令历史失败: {str(e)}")

    def save_command_history(self):
        """保存命令历史到文件"""
        try:
            history_file = "command_history.txt"
            with open(history_file, 'w', encoding='utf-8') as f:
                for command in self.command_history:
                    f.write(f"{command}\n")
        except Exception as e:
            self.logger.warning(f"保存命令历史失败: {str(e)}")

    def add_to_history(self, command: str):
        """添加命令到历史记录"""
        command = command.strip()
        if command and (not self.command_history or self.command_history[-1] != command):
            self.command_history.append(command)
            self.save_command_history()
            self.logger.debug(f"添加到历史: {command}")

    def setup_terminal(self):
        """设置终端为原始模式"""
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except Exception as e:
            self.logger.warning(f"设置终端模式失败: {str(e)}")

    def restore_terminal(self):
        """恢复终端设置"""
        try:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except Exception as e:
            self.logger.warning(f"恢复终端设置失败: {str(e)}")

    def clear_current_line(self):
        """清除当前行"""
        sys.stdout.write('\r' + ' ' * self.terminal_width + '\r')
        sys.stdout.flush()

    def display_prompt_and_input(self):
        """显示提示符和当前输入"""
        prompt = "car> "
        display_text = prompt + self.current_input
        
        # 如果显示文本太长，截断显示
        if len(display_text) > self.terminal_width - 1:
            visible_start = max(0, len(display_text) - self.terminal_width + 1)
            display_text = display_text[visible_start:]
            cursor_pos = min(self.cursor_position, len(display_text) - len(prompt))
        else:
            cursor_pos = self.cursor_position
        
        self.clear_current_line()
        sys.stdout.write(display_text)
        
        # 移动光标到正确位置
        if cursor_pos < len(self.current_input):
            sys.stdout.write('\r' + prompt)
            if cursor_pos > 0:
                sys.stdout.write(self.current_input[:cursor_pos])
        
        sys.stdout.flush()

    def navigate_history(self, direction: str):
        """浏览历史记录"""
        if not self.command_history:
            return
            
        if direction == "up":
            if self.history_index == -1:
                # 第一次按上键，保存当前输入
                self.temp_current_input = self.current_input
                self.history_index = len(self.command_history) - 1
            elif self.history_index > 0:
                self.history_index -= 1
            
            if 0 <= self.history_index < len(self.command_history):
                self.current_input = self.command_history[self.history_index]
                self.cursor_position = len(self.current_input)
                
        elif direction == "down":
            if self.history_index == -1:
                return
            elif self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.current_input = self.command_history[self.history_index]
                self.cursor_position = len(self.current_input)
            else:
                # 回到最新输入
                self.history_index = -1
                self.current_input = getattr(self, 'temp_current_input', '')
                self.cursor_position = len(self.current_input)

    async def keyboardListener(self):
        """键盘监听器"""
        self.setup_terminal()
        
        try:
            # 显示欢迎信息
            self.show_welcome_message()
            # 显示初始提示符
            self.display_prompt_and_input()
            
            # 初始状态不启用输入状态，等待用户真正开始输入
            
            while True:
                try:
                    # 使用非阻塞读取
                    ready, _, _ = await asyncio.get_event_loop().run_in_executor(
                        None, select.select, [sys.stdin], [], [], 0.1
                    )
                    
                    if ready:
                        char = await asyncio.get_event_loop().run_in_executor(
                            None, sys.stdin.read, 1
                        )
                        
                        if not char:
                            continue
                            
                        # 处理特殊按键序列
                        if ord(char) == 27:  # ESC序列开始
                            # 读取后续字符
                            try:
                                char2 = await asyncio.get_event_loop().run_in_executor(
                                    None, sys.stdin.read, 1
                                )
                                if char2 == '[':
                                    char3 = await asyncio.get_event_loop().run_in_executor(
                                        None, sys.stdin.read, 1
                                    )
                                    
                                    if char3 == 'A':  # 上箭头
                                        # 如果有历史记录可浏览，启用输入状态
                                        if self.command_history and not input_state_manager.is_inputting:
                                            await input_state_manager.set_inputting(True)
                                        self.navigate_history("up")
                                        self.display_prompt_and_input()
                                        continue
                                    elif char3 == 'B':  # 下箭头
                                        # 如果有历史记录可浏览，启用输入状态
                                        if self.command_history and not input_state_manager.is_inputting:
                                            await input_state_manager.set_inputting(True)
                                        self.navigate_history("down")
                                        self.display_prompt_and_input()
                                        continue
                                    elif char3 == 'C':  # 右箭头
                                        if self.cursor_position < len(self.current_input):
                                            self.cursor_position += 1
                                            self.display_prompt_and_input()
                                        continue
                                    elif char3 == 'D':  # 左箭头
                                        if self.cursor_position > 0:
                                            self.cursor_position -= 1
                                            self.display_prompt_and_input()
                                        continue
                            except:
                                pass
                                
                        elif ord(char) == 13 or ord(char) == 10:  # 回车键
                            command = self.current_input.strip()
                            sys.stdout.write('\n')
                            sys.stdout.flush()
                            
                            if command:
                                # 暂停输入状态，允许日志输出
                                await input_state_manager.set_inputting(False)
                                
                                self.add_to_history(command)
                                self.logger.info(f"执行命令: {command}")
                                
                                # 处理内置命令
                                if command == "help":
                                    self.show_command_suggestions()
                                elif command == "history":
                                    self.show_command_history()
                                elif command == "clear":
                                    os.system('clear' if os.name != 'nt' else 'cls')
                                    self.display_prompt_and_input()
                                elif command == "stats":
                                    self.show_command_statistics()
                                elif command == "test_log":
                                    self.test_log_blocking()
                                elif command == "log_status":
                                    self.show_log_status()
                                else:
                                    await self.keyEvent.publish(command)
                                
                                # 等待命令执行完成后，不要立即启用输入状态
                                await asyncio.sleep(0.1)  # 给命令处理一些时间
                            else:
                                # 空命令，立即恢复输入状态
                                await input_state_manager.set_inputting(False)
                            
                            # 重置输入状态
                            self.current_input = ""
                            self.cursor_position = 0
                            self.history_index = -1
                            
                            # 显示提示符，但不启用输入状态（等待用户真正开始输入）
                            self.display_prompt_and_input()
                            
                        elif ord(char) == 127 or ord(char) == 8:  # 退格键
                            if self.cursor_position > 0:
                                self.current_input = (
                                    self.current_input[:self.cursor_position-1] + 
                                    self.current_input[self.cursor_position:]
                                )
                                self.cursor_position -= 1
                                
                                # 如果输入被清空，释放输入状态以允许日志输出
                                if not self.current_input and input_state_manager.is_inputting:
                                    await input_state_manager.set_inputting(False)
                                
                                self.display_prompt_and_input()
                                
                        elif ord(char) == 3:  # Ctrl+C
                            await input_state_manager.set_inputting(False)
                            sys.stdout.write('\n退出程序...\n')
                            sys.stdout.flush()
                            await self.keyEvent.publish("exit")
                            break
                            
                        elif ord(char) == 4:  # Ctrl+D
                            if not self.current_input:
                                await input_state_manager.set_inputting(False)
                                sys.stdout.write('\n退出程序...\n')
                                sys.stdout.flush()
                                await self.keyEvent.publish("exit")
                                break
                                
                        elif ord(char) == 12:  # Ctrl+L
                            # 清屏，不启用输入状态
                            await input_state_manager.set_inputting(False)
                            os.system('clear' if os.name != 'nt' else 'cls')
                            self.display_prompt_and_input()
                            
                        elif ord(char) == 18:  # Ctrl+R - 显示历史记录
                            # 显示历史记录，不启用输入状态
                            await input_state_manager.set_inputting(False)
                            self.show_command_history()
                            
                        elif ord(char) == 9:  # Tab键 - 命令提示
                            # 显示命令提示，不启用输入状态
                            await input_state_manager.set_inputting(False)
                            self.show_command_suggestions()
                            
                        elif 32 <= ord(char) <= 126:  # 可打印字符
                            # 如果用户开始输入，启用输入状态（阻断日志）
                            if not input_state_manager.is_inputting:
                                await input_state_manager.set_inputting(True)
                            
                            self.current_input = (
                                self.current_input[:self.cursor_position] + 
                                char + 
                                self.current_input[self.cursor_position:]
                            )
                            self.cursor_position += 1
                            self.display_prompt_and_input()
                            
                    await asyncio.sleep(0.01)  # 小延迟避免CPU占用过高
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if not self.main.run:
                        raise
                    self.logger.exception(f"键盘监听异常: {str(e)}")
                    await asyncio.sleep(1)
                    
        finally:
            # 清理输入状态
            await input_state_manager.set_inputting(False)
            self.restore_terminal()
            self.save_command_history()

    def show_command_history(self):
        """显示命令历史记录"""
        sys.stdout.write('\n=== 命令历史记录 ===\n')
        if not self.command_history:
            sys.stdout.write('暂无历史命令\n')
        else:
            for i, cmd in enumerate(self.command_history, 1):
                sys.stdout.write(f'{i:3d}: {cmd}\n')
        sys.stdout.write('==================\n')
        # 显示提示符但不启用输入状态
        self.display_prompt_and_input()

    def show_command_suggestions(self):
        """显示命令建议"""
        suggestions = [
            # 旋转控制命令
            "rotateToAngle:90", "rotateToAngle:-45", "rotateBy:45", "rotateBy:-30",
            "rotateLeft", "rotateLeft:45", "rotateRight", "rotateRight:30",
            
            # 运动控制命令
            "motionTime:forward:2", "motionTime:backward:1", "motionTime:left:1", "motionTime:right:1",
            "motionTime:rotateLeft:2", "motionTime:rotateRight:2",
            
            # 基础控制
            "stopMotion", "getCurrentAngle", "enableSpeedAttenuation", "disableSpeedAttenuation",
            
            # 参数调整
            "setRotationTolerance:1.0", "setRotationKp:0.03", "setMaxRotationSpeed:2.0",
            "showMotionParams",
            
            # 系统命令
            "demonstration", "startMapping", "closeMapping", "exitCabin", "inCabin",
            "calibration", "calibrationByAngle", "searchFire", "testDepthDistance",
            "moveToDistance:1.0", "returnVoyage",
            
            # 特殊命令
            "open", "close", "exit",
            
            # 内置命令
            "help", "history", "clear", "stats", "test_log", "log_status"
        ]
        
        current_prefix = self.current_input.lower()
        matching_suggestions = [cmd for cmd in suggestions if cmd.lower().startswith(current_prefix)]
        
        if matching_suggestions:
            sys.stdout.write('\n=== 命令建议 ===\n')
            for i, suggestion in enumerate(matching_suggestions[:10], 1):  # 最多显示10个建议
                sys.stdout.write(f'{i:2d}: {suggestion}\n')
            if len(matching_suggestions) > 10:
                sys.stdout.write(f'    ... 还有 {len(matching_suggestions) - 10} 个建议\n')
            sys.stdout.write('===============\n')
        else:
            sys.stdout.write('\n=== 可用命令类别 ===\n')
            sys.stdout.write('旋转控制: rotateToAngle:角度, rotateBy:角度, rotateLeft[:角度], rotateRight[:角度]\n')
            sys.stdout.write('运动控制: motionTime:方向:时间 (方向: forward/backward/left/right/rotateLeft/rotateRight)\n')
            sys.stdout.write('基础控制: stopMotion, getCurrentAngle, showMotionParams\n')
            sys.stdout.write('系统操作: demonstration, startMapping, closeMapping, exitCabin, inCabin\n')
            sys.stdout.write('功能测试: searchFire, testDepthDistance, calibration, moveToDistance:距离\n')
            sys.stdout.write('参数设置: setRotationTolerance:值, setRotationKp:值, setMaxRotationSpeed:值\n')
            sys.stdout.write('快捷键: Ctrl+R(历史), Ctrl+L(清屏), Ctrl+C/D(退出), Tab(提示), ↑↓(历史浏览)\n')
            sys.stdout.write('内置命令: help(帮助), history(历史), clear(清屏), stats(统计), test_log(测试日志), log_status(日志状态)\n')
            sys.stdout.write('==================\n')
        
        # 显示提示符但不启用输入状态
        self.display_prompt_and_input()

    def get_command_statistics(self):
        """获取命令使用统计"""
        from collections import Counter
        command_counts = Counter()
        
        for cmd in self.command_history:
            # 提取命令的基础部分（去掉参数）
            base_cmd = cmd.split(':')[0] if ':' in cmd else cmd
            command_counts[base_cmd] += 1
        
        return command_counts

    def show_welcome_message(self):
        """显示欢迎信息"""
        sys.stdout.write('\n' + '='*60 + '\n')
        sys.stdout.write('           🚗 智能小车控制系统 v2.0 🚗\n')
        sys.stdout.write('='*60 + '\n')
        sys.stdout.write('快捷键说明:\n')
        sys.stdout.write('  ↑↓     - 浏览命令历史\n')
        sys.stdout.write('  ←→     - 移动光标\n')
        sys.stdout.write('  Tab    - 显示命令提示\n')
        sys.stdout.write('  Ctrl+R - 显示历史记录\n')
        sys.stdout.write('  Ctrl+L - 清屏\n')
        sys.stdout.write('  Ctrl+C - 退出程序\n')
        sys.stdout.write('='*60 + '\n')
        
        if self.command_history:
            sys.stdout.write(f'已加载 {len(self.command_history)} 条历史命令\n')
            # 显示最常用的命令
            stats = self.get_command_statistics()
            if stats:
                top_commands = stats.most_common(3)
                sys.stdout.write('最常用命令: ')
                sys.stdout.write(', '.join([f'{cmd}({count}次)' for cmd, count in top_commands]))
                sys.stdout.write('\n')
        
        sys.stdout.write('输入 "help" 或按 Tab 键查看可用命令\n')
        sys.stdout.write('='*60 + '\n\n')
        sys.stdout.flush()

    def show_command_statistics(self):
        """显示命令使用统计"""
        stats = self.get_command_statistics()
        
        sys.stdout.write('\n=== 命令使用统计 ===\n')
        if not stats:
            sys.stdout.write('暂无命令使用记录\n')
        else:
            sys.stdout.write(f'总命令数: {len(self.command_history)}\n')
            sys.stdout.write(f'不同命令: {len(stats)}\n\n')
            
            sys.stdout.write('使用频率排行:\n')
            for i, (cmd, count) in enumerate(stats.most_common(10), 1):
                percentage = (count / len(self.command_history)) * 100
                sys.stdout.write(f'{i:2d}. {cmd:<20} {count:3d}次 ({percentage:5.1f}%)\n')
        
        sys.stdout.write('==================\n')
        # 显示提示符但不启用输入状态
        self.display_prompt_and_input()

    def test_log_blocking(self):
        """测试日志阻断功能"""
        sys.stdout.write('\n=== 日志阻断功能测试 ===\n')
        sys.stdout.write('将在5秒内生成测试日志，请观察日志是否被正确缓存\n')
        sys.stdout.write('测试期间可以尝试输入命令\n')
        sys.stdout.flush()
        
        # 启动异步日志生成任务
        asyncio.create_task(self._generate_test_logs())
        
        # 显示提示符但不启用输入状态
        self.display_prompt_and_input()
    
    async def _generate_test_logs(self):
        """生成测试日志"""
        for i in range(10):
            await asyncio.sleep(0.5)
            self.logger.info(f"测试日志 #{i+1} - 这是一条测试日志消息")
            self.logger.warning(f"测试警告 #{i+1} - 模拟警告信息")
            if i % 3 == 0:
                self.logger.error(f"测试错误 #{i+1} - 模拟错误信息")
    
    def show_log_status(self):
        """显示日志处理状态"""
        sys.stdout.write('\n=== 日志处理状态 ===\n')
        
        # 显示输入状态管理器状态
        sys.stdout.write(f'输入状态: {"输入中" if input_state_manager.is_inputting else "空闲"}\n')
        sys.stdout.write(f'缓存日志数量: {len(input_state_manager.pending_logs)}\n')
        
        # 显示日志处理器状态
        if input_state_manager.log_handler:
            with input_state_manager.log_handler.lock:
                sys.stdout.write(f'日志处理器状态: 已连接\n')
                sys.stdout.write(f'日志处理器输入状态: {"输入中" if input_state_manager.log_handler.is_inputting else "空闲"}\n')
                sys.stdout.write(f'日志处理器缓存数量: {len(input_state_manager.log_handler.pending_logs)}\n')
        else:
            sys.stdout.write('日志处理器状态: 未连接\n')
        
        # 显示组件连接状态
        sys.stdout.write(f'输入组件: {"已连接" if input_state_manager.input_component else "未连接"}\n')
        
        sys.stdout.write('==================\n')
        # 显示提示符但不启用输入状态
        self.display_prompt_and_input()

    def __del__(self):
        """析构函数，确保恢复终端设置"""
        self.restore_terminal()
