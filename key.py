import asyncio
from pynput import keyboard

import util
from main import Component, ConfigField


class KeyEventEvent:
    PRESS = 0
    RELEASE = 1
    pass


class KeyEvent:
    keyEventEvent: KeyEventEvent = None
    char: int = 0

    def __init__(self, keyEventEvent, char):
        super().__init__()
        self.keyEventEvent = keyEventEvent
        self.char = char


class KeyComponent(Component):
    keyEvent: util.Broadcaster[KeyEvent] = util.Broadcaster()

    listener: keyboard.Listener = None

    async def init(self):
        await super().init()
        self.startKeyboardListener()
        asyncio.create_task(self.processKeyEventLogLoop())

    def startKeyboardListener(self):
        self.listener = keyboard.Listener(
            on_press=lambda k: self.onPress(k),
            on_release=lambda k: self.onRelease(k)
        )
        self.listener.start()

    # 同步回调函数（在后台线程运行）
    def onPress(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)
        self.keyEvent.publish_nowait(KeyEvent(key_char, KeyEventEvent.PRESS))

    def onRelease(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)
        self.keyEvent.publish_nowait(KeyEvent(key_char, KeyEventEvent.RELEASE))

    async def processKeyEventLogLoop(self):
        queue = await  self.keyEvent.subscribe(asyncio.Queue(maxsize=16))

        while True:
            # 从队列获取事件
            event = await queue.get()

            if event.keyEventEvent == KeyEventEvent.PRESS:
                print(f"异步处理按键按下: {event.char}")
                # 处理按键按下逻辑

            elif event.keyEventEvent == KeyEventEvent.RELEASE:
                print(f"异步处理按键释放: {event.char}")
                # 处理按键释放逻辑
