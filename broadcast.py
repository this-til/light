import pygame as pg

from main import Component, ConfigField

class BroadcastComponent(Component):
    
    soundSource : ConfigField[str] = ConfigField()
    
    async def awakeInit(self):
        await super().awakeInit()
        pg.init()
        
        
        
    
    pass