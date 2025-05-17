#!/usr/bin/python3


import logging
import asyncio
import cv2
import numpy as np
import util
import threading
import time
from util import Box, Color

from rknn.api import RKNN

from main import Component, ConfigField


logger: logging.Logger = logging.getLogger(__name__)

inferenceLock = threading.Lock()


class Item:

    name: str
    color: Color

    def __init__(self, name: str, color: Color):
        self.name = name
        self.color = color
        pass

    pass


class Cell:

    item: Item
    box: Box
    probability: float

    def __init__(self, item: Item, box: Box, probability: float):
        self.item = item
        self.box = box
        self.probability = probability
        pass

    pass


class Model:

    name: str
    itemList: list[Item]
    path: str
    size: tuple[int, int]

    rknn: RKNN = None  # type: ignore

    def __init__(
        self,
        name: str,
        itemList: list[Item],
        path: str,
        size: tuple[int, int] = (640, 640),
    ):
        self.name = name
        self.itemList = itemList
        self.path = path
        self.size = size

        pass

    def load(self):

        if self.rknn is not None:
            return

        self.rknn = RKNN()
        self.rknn.load_rknn(self.path)

        logger.info(f"load rknn model: {self.path}")

        ret = self.rknn.init_runtime(
            target="rk3588", fallback_prior_device="gpu", core_mask=RKNN.NPU_CORE_ALL
        )

        if ret != 0:
            logger.error(f"init runtime failed: {self.path}")
            pass

        pass

    def run(self, inputImage: cv2.typing.MatLike) -> list[Cell]:
        if self.rknn is None:
            raise RuntimeError("Model not loaded")

        h, w = inputImage.shape[:2]
        originalSize: tuple[int, int] = (h, w)

        inputImage = util.changeSize(inputImage, self.size)
        inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)

        return self.directRun(inputImage, originalSize)

    def directRun(
        self, inputImage: cv2.typing.MatLike, originalSize: tuple[int, int]
    ) -> list[Cell]:

        with inferenceLock:
            try:
                res = self.rknn.inference(inputs=[inputImage], data_format="nhwc")

            except Exception as e:
                logger.exception(
                    f"thie {self.path} model inference raise Exception: {str(e)} "
                )
                return []

        boxes, classes, scores = self.post_process(res)

        outList: list[Cell] = []

        if boxes is not None:
            for box, score, cl in zip(
                util.realBox(boxes, originalSize, self.size), scores, classes  # type: ignore
            ):
                outList.append(Cell(self.itemList[cl], box, score))
                pass

        return outList

    def post_process(self, input_data):
        boxes, scores, classes_conf = [], [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])
            scores.append(
                np.ones_like(
                    input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32
                )
            )

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.size[1] // grid_h, self.size[0] // grid_w]).reshape(
            1, 2, 1, 1
        )

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold."""
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(
            class_max_score * box_confidences >= OBJ_THRESH
        )
        scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        import torch

        x = torch.tensor(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y.numpy()


class Result:

    inputImage: cv2.typing.MatLike
    outputImage: cv2.typing.MatLike | None

    cellMap: dict[Model, list[Cell]] = {}

    def __init__(
        self, inputImage: cv2.typing.MatLike, cellMap: dict[Model, list[Cell]]
    ):
        self.inputImage = inputImage
        self.outputImage = None
        self.cellMap = cellMap

    async def drawOutputImageAsunc(self) -> cv2.typing.MatLike:
        if self.outputImage is None:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.drawOutputImage
            )
        return self.outputImage

    def drawOutputImage(self) -> cv2.typing.MatLike:
        """
        在outputImage上绘制所有检测框和标签
        """

        if self.outputImage is None:
            # 深拷贝原始图像，避免修改原图
            self.outputImage = self.inputImage.copy()

            if self.cellMap is None:
                return self.outputImage

            # 遍历所有模型及其检测结果
            for model, cells in self.cellMap.items():
                for cell in cells:
                    # 获取检测框坐标 (转换为整数)
                    x = int(cell.box.x)
                    y = int(cell.box.y)
                    w = int(cell.box.w)
                    h = int(cell.box.h)

                    # 计算矩形坐标 (OpenCV需要左上角和右下角)
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                    # 获取颜色 (BGR格式)
                    color = (
                        cell.item.color.b,  # OpenCV使用BGR通道顺序
                        cell.item.color.g,
                        cell.item.color.r,
                    )

                    # 绘制矩形框
                    cv2.rectangle(
                        img=self.outputImage,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=color,
                        thickness=2,  # 线宽
                    )

                    # 构建标签文本 (类别名 + 置信度)
                    label = f"{cell.item.name} {cell.probability:.2f}"

                    # 计算文本位置 (左上角偏移)
                    text_x = x1 + 5
                    text_y = y1 - 10 if y1 > 20 else y1 + 20  # 避免超出图像顶部

                    # 绘制文本背景
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1  # 字体大小  # 线宽
                    )
                    cv2.rectangle(
                        img=self.outputImage,
                        pt1=(x1, y1 - text_h - 5),
                        pt2=(x1 + text_w + 5, y1),
                        color=color,
                        thickness=cv2.FILLED,  # 填充模式
                    )

                    # 绘制文本
                    cv2.putText(
                        img=self.outputImage,
                        text=label,
                        org=(text_x, text_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,  # 字体缩放系数
                        color=(
                            (0, 0, 0) if sum(color) > 382 else (255, 255, 255)
                        ),  # 自动选择文本颜色
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        return self.outputImage


accident = Item("accident", Color(255, 0, 0))
car_accident_model = Model(
    "car_accident", [accident], "/home/elf/light/model/car_accident.rknn"
)


fall_down = Item("fall down", Color(255, 150, 51))
stand_person = Item("stand person", Color(100, 255, 100))
fall_down_model = Model(
    "fall_down", [fall_down, stand_person], "/home/elf/light/model/fall_down.rknn"
)

modelMap = {
    car_accident_model.name: car_accident_model,
    fall_down_model.name: fall_down_model,
}

models = list(modelMap.values())


OBJ_THRESH: float = 0.5
NMS_THRESH: float = 0.5


class DetectionComponent(Component):

    OBJ_THRESH: ConfigField[float] = ConfigField()
    NMS_THRESH: ConfigField[float] = ConfigField()

    async def init(self):
        for name, model in modelMap.items():
            model.load()

    async def initEnd(self):
        await super().initEnd()
        global OBJ_THRESH, NMS_THRESH
        OBJ_THRESH = self.OBJ_THRESH
        NMS_THRESH = self.NMS_THRESH

    def runDetection(
        self, inputImage: cv2.typing.MatLike, useModel: set[Model] | list[Model]
    ) -> Result:

        h, w = inputImage.shape[:2]
        originalSize: tuple[int, int] = (h, w)

        cellMap: dict[Model, list[Cell]] = {}

        sizeMap: dict[tuple[int, int], list[Model]] = {}

        for m in useModel:

            if m.size not in sizeMap:
                sizeMap[m.size] = []

            sizeMap[m.size].append(m)

        for size, modelList in sizeMap.items():
            _inputImage = util.changeSize(inputImage, size)
            _inputImage = cv2.cvtColor(_inputImage, cv2.COLOR_BGR2RGB)

            for m in modelList:
                cellRes: list[Cell] = m.directRun(_inputImage, originalSize)
                cellMap[m] = cellRes

        return Result(inputImage, cellMap)
