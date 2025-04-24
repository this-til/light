#!/usr/bin/python3
import asyncio
import serial
import serial_asyncio

import util

print("Hello, World!")

active: bool = True


def exitHandler():
    global active
    active = False
    print("Exiting...")


loop = asyncio.get_event_loop()
loop.set_debug(True)

reader: asyncio.StreamReader = None
writer: asyncio.StreamWriter = None


async def init():

    global reader
    global writer

    reader, writer = await serial_asyncio.open_serial_connection(
        url="/dev/ttyS9",
        baudrate=115200,
        bytesize=serial_asyncio.serial.EIGHTBITS,
        parity=serial_asyncio.serial.PARITY_NONE,
        stopbits=serial_asyncio.serial.STOPBITS_ONE,
    )
    pass


async def asyncReadSerialLoop():
    """
    异步循环读取
    """
    while True:
        try:
            data = await reader.read(1024)
            print(f"Received data: {data.decode('utf-8')}")
            await asyncio.sleep(0.1)  # 休眠100ms
        except Exception as e:
            print(f"Error reading from serial: {e}")
            return


async def main():

    try:

        await init()

        asyncio.create_task(asyncReadSerialLoop())

        while active:
            await asyncio.sleep(1)

    finally:

        await util.gracefulShutdown()

        if writer and not writer.is_closing():
            writer.close()
            await writer.wait_closed()

        exit(0)
        pass


if __name__ == "__main__":
    asyncio.run(main())
