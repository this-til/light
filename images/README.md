# 图像轮播文件夹

此文件夹用于存放图像轮播的图片文件。

## 支持的图像格式
- JPG/JPEG
- PNG
- BMP
- GIF

## 使用方法

1. 将需要轮播的图片文件放置在此文件夹中
2. 在 `config.json` 中配置 Carousel 组件：
   ```json
   "Carousel": {
     "enable": true,
     "images_folder": "images",
     "interval": 3.0,
     "fullscreen": true,
     "width": 1920,
     "height": 1080,
     "fade_duration": 0.5
   }
   ```
3. 启动应用程序

## 配置说明

- `enable`: 是否启用图像轮播功能
- `images_folder`: 图像文件夹路径
- `interval`: 轮播间隔时间（秒）
- `fullscreen`: 是否全屏显示
- `width`: 窗口模式下的宽度
- `height`: 窗口模式下的高度
- `fade_duration`: 淡入淡出时间（暂未实现）

## 控制键

- `ESC`: 退出轮播
- `空格键`: 手动切换到下一张图片
- `左箭头键`: 切换到上一张图片
- `右箭头键`: 切换到下一张图片

## 注意事项

- 图片会自动缩放以适应屏幕尺寸，保持原始宽高比
- 图片会居中显示，空白区域用黑色填充
- 支持的图片数量仅受内存限制 