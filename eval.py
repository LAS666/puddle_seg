#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from tools import fp as fp_logic

IMAGE_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tif',
    '.tiff',
    '.webp',
}

def check_map(array, num_classes, label):
    array = np.asarray(array)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f'{label} 必须是二维整数类别图，不能是 RGB 彩色图或概率图')
    if not np.all(((array >= 0) & (array <= num_classes)) | (array == 255)):
        raise ValueError(f'{label} 类别值必须为 0..{num_classes} 或 255')
    return array

def build_valid_mask(gt, roi=None):
    valid = gt != 255
    if roi is not None:
        roi = np.asarray(roi)
        if roi.shape != gt.shape or not np.all(np.isin(roi, [0, 1, 255])):
            raise ValueError('ROI 必须与 GT 同尺寸，且为 0/1 或 0/255 二值图')
        valid &= roi != 0
    return valid

class PixelMetrics:

    """累计像素级指标；Recall/IoU/FPR 仍然全部按像素计算。"""

    def __init__(self, names):

        self.names = list(names)

        if not 1 <= len(self.names) <= 254:

            raise ValueError('前景类别数须为 1..254；255 保留给 ignore/unknown')

        # 每类顺序：TP, FP, FN, TN，全部是像素数。

        self.counts = np.zeros((len(names), 4), dtype=np.int64)

        self.images = 0

        self.valid_pixels = 0

        self.unknown_pixels = 0

        self.images_with_valid_pixels = 0

    def update(self, pred, gt, roi=None):

        pred = check_map(pred, len(self.names), '预测')

        gt = check_map(gt, len(self.names), 'GT')

        if pred.shape != gt.shape:

            raise ValueError(f'预测与 GT 尺寸不一致：{pred.shape} / {gt.shape}')

        valid = build_valid_mask(gt, roi)

        self.images += 1

        self.valid_pixels += int(valid.sum())

        self.images_with_valid_pixels += int(valid.any())

        self.unknown_pixels += int(np.count_nonzero(valid & (pred == 255)))

        for i in range(len(self.names)):

            pred_region = pred == i + 1

            gt_region = gt == i + 1

            self.counts[i] += [

                np.count_nonzero(valid & pred_region & gt_region),

                np.count_nonzero(valid & pred_region & ~gt_region),

                np.count_nonzero(valid & ~pred_region & gt_region),

                np.count_nonzero(valid & ~pred_region & ~gt_region),

            ]

    def rows(self):

        """返回像素级统计；Pixel_FP 保留用于正确计算 IoU/FPR。"""

        def divide(a, b):

            return float(a / b) if b else None

        rows = []

        for i, (tp, fp, fn, tn) in enumerate(self.counts.tolist()):

            rows.append(

                dict(

                    class_id=i + 1,

                    name=self.names[i],

                    Pixel_TP=tp,

                    Pixel_FP=fp,

                    Pixel_FN=fn,

                    Pixel_TN=tn,

                    Recall=divide(tp, tp + fn),

                    IoU=divide(tp, tp + fp + fn),

                    Pixel_FPR=divide(fp, fp + tn),

                )

            )

        return rows

INSTANCE_MIOU_DEFINITION = (
    'For each matched prediction P, U is the union of its assigned matched GTs; '
    'IoU=area(P intersection U)/area(P union U); average equally over matched '
    'predictions; unmatched predictions and GTs excluded; null if no matches'
)

class InstanceMetrics:
    """共享 fp.py 的去重、GT 覆盖匹配、Fragment 和小 FP 过滤规则。

    每个类别独立匹配和统计；单类别规则保持不变。
    Matched_Instances/GT_TP 是 GT 侧计数，Prediction_TP 是预测侧计数。
    """
    def __init__(self, names, iou_threshold, overlap_threshold=fp_logic.MASK_OVERLAP_THRESHOLD,
                 overlap_class_agnostic=fp_logic.MASK_OVERLAP_CLASS_AGNOSTIC,
                 min_area=fp_logic.MIN_INSTANCE_AREA, min_fp_area=fp_logic.MIN_FP_AREA):
        self.names = list(names)
        self.overlap_threshold = overlap_threshold
        self.overlap_class_agnostic = overlap_class_agnostic
        self.min_area = min_area
        self.min_fp_area = min_fp_area
        self.totals = dict(Pred_Instances=0, GT_Instances=0, Matched_Instances=0,
                           GT_TP=0, Prediction_TP=0, FP_Count=0, Instance_FN=0,
                           Fragment_Ignore=0, Duplicate_Ignore=0, Small_FP_Ignore=0)
        self.ious = []
        self.class_totals = [self.totals.copy() for _ in self.names]
        self.class_ious = [[] for _ in self.names]

    def update(self, pred_masks, pred_classes, pred_scores, gt_masks, gt_classes, valid):
        pred = np.asarray(pred_masks, dtype=bool) & valid[None, :, :]
        gt = np.asarray(gt_masks, dtype=bool) & valid[None, :, :]
        gt_classes = np.asarray(gt_classes)
        pred_classes = np.asarray(pred_classes)
        for classes_input, masks_input in ((gt_classes, gt), (pred_classes, pred)):
            if classes_input.shape != (len(masks_input),) or not np.all(
                (classes_input >= 0) & (classes_input < len(self.names)) &
                (classes_input == classes_input.astype(int))
            ):
                raise ValueError('实例类别与 mask 不匹配或超出 data.yaml 范围')
        pred, classes, scores = fp_logic.filter_small_instances(
            pred, np.asarray(pred_classes), np.asarray(pred_scores), self.min_area)
        gt_keep = np.asarray([np.any(m) and
              (not fp_logic.FILTER_SMALL_GT or np.count_nonzero(m) >= self.min_area) for m in gt], dtype=bool)
        gt, gt_classes = gt[gt_keep], gt_classes[gt_keep]
        before_classes = classes.copy()
        pred, classes, scores = fp_logic.remove_overlapping_masks(
            pred, classes, scores, self.overlap_threshold, self.overlap_class_agnostic)
        for cid in range(len(self.names)):
            cp, cg = pred[classes == cid], gt[gt_classes == cid]
            stats = {}
            ids = fp_logic.find_fp(cp, cg, stats, min_fp_area=self.min_fp_area)
            increments = dict(Pred_Instances=len(cp), GT_Instances=len(cg),
                              Matched_Instances=stats['gt_tp'], GT_TP=stats['gt_tp'],
                              Prediction_TP=stats['prediction_tp'], FP_Count=len(ids),
                              Instance_FN=stats['gt_fn'], Fragment_Ignore=stats['fragment_ignore'],
                              Duplicate_Ignore=int(np.count_nonzero(before_classes == cid))-len(cp),
                              Small_FP_Ignore=stats['small_fp_ignore'])
            for key, value in increments.items():
                self.class_totals[cid][key] += value
                self.totals[key] += value
            for match in stats['matches']:
                prediction = cp[match['prediction_id'] - 1]
                gt_union = np.zeros_like(prediction, dtype=bool)
                for gid in match['gt_ids']:
                    gt_union |= cg[gid - 1]
                iou = fp_logic.mask_iou(prediction, gt_union)
                self.class_ious[cid].append(iou)
                self.ious.append(iou)

    def rows(self):
        rows = []
        for i in range(len(self.names)):
            row = self.class_totals[i].copy()
            denom = row['Prediction_TP'] + row['FP_Count']
            row['FP'] = row['FP_Count'] / denom if denom else None
            row['Instance_Precision'] = row['Prediction_TP'] / denom if denom else None
            row['Instance_Recall'] = row['GT_TP'] / row['GT_Instances'] if row['GT_Instances'] else None
            row['Instance_mIoU'] = float(np.mean(self.class_ious[i])) if self.class_ious[i] else None
            rows.append(row)
        return rows

def merge_instances(masks, classes, scores, shape, num_classes):

    """实例 mask 合并为像素类别图，用于像素级 Recall/IoU 计算。"""

    output = np.zeros(shape, dtype=np.uint8)

    if len(classes) == 0:

        if len(masks) or len(scores):

            raise ValueError('实例掩码、类别与分数数量不一致')

        return output

    masks = np.asarray(masks)

    classes = np.asarray(classes)

    scores = np.asarray(scores)

    if masks.shape != (len(classes), *shape) or scores.shape != classes.shape:

        raise ValueError('实例掩码必须对齐原图尺寸，且与类别、分数一一对应')

    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(masks)):

        raise ValueError('实例输出包含 NaN/Inf')

    if not np.all(

        (classes == classes.astype(int))

        & (classes >= 0)

        & (classes < num_classes)

    ):

        raise ValueError('模型实例类别不在 data.yaml 定义的范围内')

    for idx in sorted(

        range(len(classes)),

        key=lambda i: (-float(scores[i]), int(classes[i]), i),

    ):

        region = (masks[idx] > 0.5) & (output == 0)

        output[region] = int(classes[idx]) + 1

    return output

def filter_small_instances(

    masks,

    classes,

    scores=None,

    min_area=0,

):

    """过滤面积过小的实例，面积单位为 px²。"""

    if min_area <= 0 or len(masks) == 0:

        return masks, classes, scores

    areas = np.sum(masks, axis=(1, 2))

    keep = areas >= min_area

    masks = masks[keep]

    classes = classes[keep]

    if scores is not None:

        scores = scores[keep]

    return masks, classes, scores

def remove_overlapping_masks(

    masks,

    classes,

    scores,

    overlap_threshold=0.5,

    class_agnostic=False,

):

    """按交叠像素 / 较小 mask 面积去重，优先保留面积大的预测。

    规则与第二个 FP 可视化脚本一致：

        intersection(mask_a, mask_b) / min(area_a, area_b) > threshold

    先按面积降序处理；等面积按置信度降序，再按输入索引升序。

    返回结果保持原输入顺序。GT 不参与此去重。

    """

    masks = np.asarray(masks)

    classes = np.asarray(classes)

    scores = np.asarray(scores, dtype=np.float64)

    overlap_threshold = float(overlap_threshold)

    if masks.ndim != 3:

        raise ValueError('masks 必须是 [N, H, W] 三维数组')

    if classes.shape != (len(masks),) or scores.shape != (len(masks),):

        raise ValueError('masks、classes、scores 必须一一对应')

    if not np.isfinite(overlap_threshold) or not 0 <= overlap_threshold <= 1:

        raise ValueError('overlap_threshold 必须在 0..1')

    if not np.all(np.isfinite(scores)):

        raise ValueError('scores 包含 NaN/Inf')

    if masks.dtype != np.bool_:

        if not np.all((masks == 0) | (masks == 1) | (masks == 255)):

            raise ValueError('masks 必须先二值化，不能直接传入概率图')

        masks = masks.astype(bool)

    count = len(masks)

    if count <= 1 or overlap_threshold == 1.0:

        return masks, classes, scores

    areas = np.count_nonzero(masks, axis=(1, 2))

    order = sorted(

        (index for index in range(count) if areas[index] > 0),

        key=lambda index: (-int(areas[index]), -float(scores[index]), index),

    )

    bounds = {}

    for index in order:

        ys = np.flatnonzero(np.any(masks[index], axis=1))

        xs = np.flatnonzero(np.any(masks[index], axis=0))

        bounds[index] = (

            int(ys[0]), int(ys[-1]) + 1,

            int(xs[0]), int(xs[-1]) + 1,

        )

    keep = np.ones(count, dtype=bool)

    kept_indices = []

    for small_index in order:

        small_area = int(areas[small_index])

        sy0, sy1, sx0, sx1 = bounds[small_index]

        for large_index in kept_indices:

            if not class_agnostic and classes[small_index] != classes[large_index]:

                continue

            ly0, ly1, lx0, lx1 = bounds[large_index]

            y0, y1 = max(sy0, ly0), min(sy1, ly1)

            x0, x1 = max(sx0, lx0), min(sx1, lx1)

            if y0 >= y1 or x0 >= x1:

                continue

            intersection = int(np.count_nonzero(

                masks[small_index, y0:y1, x0:x1]

                & masks[large_index, y0:y1, x0:x1]

            ))

            overlap = intersection / small_area

            if overlap > overlap_threshold:

                keep[small_index] = False

                break

        if keep[small_index]:

            kept_indices.append(small_index)

    return masks[keep], classes[keep], scores[keep]

def remove_small_regions(class_map, min_area):

    """删除类别图中面积小于阈值的连通区域。"""

    if min_area <= 0:

        return class_map

    import cv2

    output = class_map.copy()

    for cls in np.unique(class_map):

        if cls == 0 or cls == 255:

            continue

        binary = (class_map == cls).astype(np.uint8)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(

            binary,

            connectivity=8,

        )

        for index in range(1, count):

            area = stats[index, cv2.CC_STAT_AREA]

            if area < min_area:

                output[labels == index] = 0

    return output

def empty_instances(shape):

    """创建空实例集合。"""

    return (

        np.zeros((0, *shape), dtype=bool),

        np.zeros((0,), dtype=np.int64),

        np.zeros((0,), dtype=np.float64),

    )

def yolo_txt_to_instances(path, shape, num_classes, overlap='error'):

    """读取 YOLO 分割 TXT，同时返回语义类别图和原始 GT 实例。"""

    import cv2

    path = Path(path)

    if not path.is_file():

        raise FileNotFoundError(path)

    height, width = shape

    semantic = np.zeros(shape, dtype=np.uint8)

    instance_masks = []

    instance_classes = []

    for line_no, line in enumerate(

        path.read_text(encoding='utf-8-sig').splitlines(),

        1,

    ):

        if not line.strip():

            continue

        values = np.asarray([float(x) for x in line.split()])

        if (

            len(values) < 7

            or (len(values) - 1) % 2

            or not np.isfinite(values).all()

            or values[0] != int(values[0])

            or not 0 <= values[0] < num_classes

            or np.any(values[1:] < 0)

            or np.any(values[1:] > 1)

        ):

            raise ValueError(f'{path}:{line_no} 不是有效 YOLO 多边形标注')

        points = values[1:].reshape(-1, 2)

        # 与 YOLO 常用栅格化一致：归一化坐标乘 W/H，然后截断为整数。

        points = (points * [width, height]).astype(np.int32)

        mask = np.zeros(shape, dtype=np.uint8)

        cv2.fillPoly(mask, [points], 1)

        mask = mask.astype(bool)

        class_index = int(values[0])

        class_value = class_index + 1

        conflict = mask & (semantic != 0) & (semantic != class_value)

        if conflict.any() and overlap == 'error':

            raise ValueError(

                f'{path}:{line_no} 不同类别 GT 多边形重叠；请提供语义 PNG，'

                '或明确使用 --gt-overlap first/last（会影响评测定义）'

            )

        semantic_mask = mask.copy()

        if overlap == 'first':

            semantic_mask &= semantic == 0

        semantic[semantic_mask] = class_value

        # 实例级 GT 保留原始多边形，不因语义图的覆盖顺序丢失实例。

        if mask.any():

            instance_masks.append(mask)

            instance_classes.append(class_index)

    if instance_masks:

        masks = np.stack(instance_masks, axis=0)

        classes = np.asarray(instance_classes, dtype=np.int64)

    else:

        masks = np.zeros((0, *shape), dtype=bool)

        classes = np.zeros((0,), dtype=np.int64)

    return semantic, masks, classes

def class_map_to_instances(class_map, num_classes, valid=None):

    """类别 PNG -> 连通域实例。

    注意：类别 PNG 本身没有实例 ID，因此这里只能把同类别的每个 8 邻域连通域

    当作一个实例；两个同类目标如果在 PNG 中粘连，会被视为同一个实例。

    """

    import cv2

    class_map = check_map(class_map, num_classes, '类别图')

    if valid is None:

        valid = class_map != 255

    else:

        valid = np.asarray(valid, dtype=bool)

        if valid.shape != class_map.shape:

            raise ValueError('valid 与类别图尺寸不一致')

    masks = []

    classes = []

    for class_index in range(num_classes):

        binary = ((class_map == class_index + 1) & valid).astype(np.uint8)

        component_count, labels = cv2.connectedComponents(binary, connectivity=8)

        for component_id in range(1, component_count):

            mask = labels == component_id

            if mask.any():

                masks.append(mask)

                classes.append(class_index)

    if masks:

        return np.stack(masks, axis=0), np.asarray(classes, dtype=np.int64)

    return (

        np.zeros((0, *class_map.shape), dtype=bool),

        np.zeros((0,), dtype=np.int64),

    )

def load_map(path):

    path = Path(path)

    if not path.is_file():

        raise FileNotFoundError(path)

    with Image.open(path) as image:

        return np.array(image)

def save_visualizations(image_path, pred, gt, output_dir, relative, num_classes):

    """每个样本导出四联图：原图、像素误差、GT、预测。"""

    from ultralytics.utils.plotting import colors
    import cv2

    with Image.open(image_path) as image:

        rgb = np.array(image.convert('RGB'))

    panels = []

    for class_map in (gt, pred):

        overlay = rgb.copy()

        for class_id in range(1, num_classes + 1):

            region = class_map == class_id

            color = np.asarray(colors(class_id - 1), dtype=np.float32)

            overlay[region] = np.rint(

                rgb[region] * 0.5 + color * 0.5

            ).astype(np.uint8)

        # 只改变可视化：语义前景 1..N 显示为 YOLO 类别编号 0..N-1。
        # 每个连通区域标一次；背景和 ignore 不标注。
        for class_id in range(1, num_classes + 1):
            count, components, stats, centers = cv2.connectedComponentsWithStats(
                (class_map == class_id).astype(np.uint8), connectivity=8
            )
            for component_id in range(1, count):
                x0, y0, width, height, _ = stats[component_id]
                ys, xs = np.where(components[y0:y0 + height, x0:x0 + width] == component_id)
                xs, ys = xs + x0, ys + y0
                cx, cy = centers[component_id]
                closest = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
                text = str(class_id - 1)
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                x = max(0, min(int(xs[closest]) - tw // 2, overlay.shape[1] - tw - 1))
                y = max(th, min(int(ys[closest]) + th // 2, overlay.shape[0] - baseline - 1))
                cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panels.append(overlay)

    # 可视化仍然是像素级错误，仅用于观察分割误差。

    error = rgb.copy()

    valid = gt != 255

    false_positive = valid & (gt == 0) & (pred != 0)

    false_negative = valid & (gt != 0) & (pred == 0)

    wrong_class = valid & (gt != 0) & (pred != 0) & (gt != pred)

    for region, color in (

        (false_positive, (255, 0, 0)),

        (false_negative, (0, 100, 255)),

        (wrong_class, (255, 255, 0)),

    ):

        error[region] = color

    top = np.concatenate((rgb, error), axis=1)

    bottom = np.concatenate((panels[0], panels[1]), axis=1)

    destination = Path(output_dir) / 'comparison' / relative

    destination.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.concatenate((top, bottom), axis=0)).save(destination)

def load_names(data):

    names = data.get('names')

    if isinstance(names, dict):

        names = {int(k): str(v) for k, v in names.items()}

        if sorted(names) != list(range(len(names))):

            raise ValueError('data.yaml 的 names 编号必须从 0 连续递增（仅前景类别）')

        return [names[i] for i in range(len(names))]

    if isinstance(names, list) and names:

        return list(map(str, names))

    raise ValueError('data.yaml 必须包含 names')

def parse_args():

    parser = argparse.ArgumentParser(

        description=(

            '分割评测：Recall/IoU 为像素级；CSV 中 FP 为实例级误检率 FP_Count/(TP+FP_Count)。'

        ),

        formatter_class=argparse.RawDescriptionHelpFormatter,

    )

    parser.add_argument(

        '--data',

        type=Path,

        required=True,

        help='names 为 N 个前景类别；类别图为背景0、前景1..N',

    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument('--weights', type=Path, help='YOLO 分割权重')

    mode.add_argument(

        '--pred-dir',

        type=Path,

        help='已有预测类别 PNG 根目录；255 表示 unknown',

    )

    parser.add_argument(

        '--gt-dir',

        type=Path,

        help='GT 类别 PNG 根目录；预测 PNG 模式必填',

    )

    parser.add_argument(

        '--roi-dir',

        type=Path,

        help='可选独立二值 ROI PNG 根目录，非零像素参与评测',

    )

    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')

    parser.add_argument(

        '--conf',

        type=float,

        default=0.25,

        help='YOLO 实例置信度阈值，默认0.25',

    )

    parser.add_argument('--nms-iou', type=float, default=0.7)

    parser.add_argument(

        '--instance-iou',

        type=float,

        default=0.5,

        help='实例 TP/FP 一对一匹配的 Mask IoU 阈值，默认0.5',

    )

    parser.add_argument('--max-det', type=int, default=300)

    parser.add_argument(

        '--mask-overlap-threshold',

        type=float,

        default=0.5,

        help=(

            '预测 mask 去重：intersection/min(area_a, area_b) 严格大于该阈值'

            '时删除较小 mask；默认0.5；1.0关闭'

        ),

    )

    parser.add_argument(

        '--mask-overlap-class-agnostic',

        action='store_true',

        help='跨类别去重；默认仅同类别内部去重',

    )

    parser.add_argument(

        '--min-instance-area',

        type=int,

        default=fp_logic.MIN_INSTANCE_AREA,

        help='实例匹配时预测与 GT 面积过滤阈值，与 fp.py 一致',

    )

    parser.add_argument('--imgsz', type=int, default=640)

    parser.add_argument(

        '--device',

        default=None,

        help='例如 cpu 或 0；默认由 Ultralytics 选择',

    )

    parser.add_argument(

        '--gt-overlap',

        choices=['error', 'first', 'last'],

        default='error',

        help='TXT GT 异类重叠策略；默认报错，避免隐式改变 GT',

    )

    parser.add_argument(

        '--output',

        type=Path,

        default=Path('seg_metrics_instance_fp.csv'),

    )

    parser.add_argument(

        '--visualize-dir',

        type=Path,

        help='每个样本保存 2×2 四联图：左上原图、右上误差、左下真值、右下预测',

    )

    parser.add_argument('--min-fp-area', type=int, default=fp_logic.MIN_FP_AREA)

    args = parser.parse_args()
    if args.min_instance_area < 0 or args.min_fp_area < 0:
        parser.error('面积阈值不能为负数')

    if args.pred_dir and not args.gt_dir:

        parser.error('--pred-dir 模式必须提供 --gt-dir')

    if args.pred_dir and args.visualize_dir:

        parser.error('--visualize-dir 需要 --weights 模式提供原图')

    if not (

        0 <= args.conf <= 1

        and 0 <= args.nms_iou <= 1

        and 0 <= args.instance_iou <= 1

        and 0 <= args.mask_overlap_threshold <= 1

    ):

        parser.error('conf/nms-iou/instance-iou/mask-overlap-threshold 必须在 0..1')

    if args.max_det < 1 or args.imgsz < 1:

        parser.error('imgsz/max-det 必须为正数')

    if args.output.suffix.lower() != '.csv':

        parser.error('--output 必须以 .csv 结尾')

    return args

def combine_rows(pixel_rows, instance_rows):

    """合并像素级和实例级结果；FP 字段为实例级误检率。"""

    if len(pixel_rows) != len(instance_rows):

        raise ValueError('像素级与实例级类别数不一致')

    rows = []

    for pixel_row, instance_row in zip(pixel_rows, instance_rows):

        row = dict(pixel_row)

        row.update(instance_row)

        rows.append(row)

    return rows

def evaluate(args):

    data = yaml.safe_load(args.data.read_text(encoding='utf-8'))

    names = load_names(data)

    pixel_metric = PixelMetrics(names)

    instance_metric = InstanceMetrics(

        names,

        args.instance_iou,

        overlap_threshold=args.mask_overlap_threshold,

        overlap_class_agnostic=args.mask_overlap_class_agnostic,
        min_area=args.min_instance_area, min_fp_area=args.min_fp_area,

    )

    samples = []

    if args.pred_dir:

        # PNG 类别图没有实例 ID，因此实例级 FP 通过同类 8 邻域连通域近似恢复。

        inputs = sorted(args.gt_dir.rglob('*.png'))

        if not inputs:

            raise ValueError('GT 目录没有 PNG 类别图')

        for index, gt_path in enumerate(inputs):

            relative = gt_path.relative_to(args.gt_dir)

            pred = load_map(args.pred_dir / relative)

            gt = load_map(gt_path)

            pred = check_map(pred, len(names), '预测')

            gt = check_map(gt, len(names), 'GT')

            if pred.shape != gt.shape:

                raise ValueError(

                    f'预测与 GT 尺寸不一致：{pred.shape} / {gt.shape}，文件：{relative}'

                )

            roi = load_map(args.roi_dir / relative) if args.roi_dir else None

            valid = build_valid_mask(gt, roi)

            pred_masks, pred_classes = class_map_to_instances(

                pred,

                len(names),

                valid=valid,

            )

            gt_masks, gt_classes = class_map_to_instances(

                gt,

                len(names),

                valid=valid,

            )

            pred_scores = np.ones(len(pred_classes), dtype=np.float64)

            pixel_metric.update(pred, gt, roi)

            instance_metric.update(

                pred_masks,

                pred_classes,

                pred_scores,

                gt_masks,

                gt_classes,

                valid,

            )

            samples.append(str(relative))

            print(f'评估 {index + 1}/{len(inputs)}: {relative}', flush=True)

    else:

        from ultralytics import YOLO

        if not args.weights.is_file():

            raise FileNotFoundError(args.weights)

        root = Path(data.get('path') or '.')

        if not root.is_absolute():

            root = args.data.resolve().parent / root

        split_value = data.get(args.split)

        if not isinstance(split_value, str):

            raise ValueError('本脚本要求数据 split 指向单个图片目录')

        image_dir = (root / split_value).resolve()

        if not image_dir.is_dir():

            raise ValueError(f'图片目录不存在：{image_dir}')

        inputs = sorted(

            path

            for path in image_dir.rglob('*')

            if path.suffix.lower() in IMAGE_EXTENSIONS

        )

        if not inputs:

            raise ValueError('图片目录为空')

        relative_keys = [

            path.relative_to(image_dir).with_suffix('.png') for path in inputs

        ]

        if len(set(relative_keys)) != len(inputs):

            raise ValueError('存在同路径同主文件名但不同扩展名的图片，无法唯一匹配标注')

        model = YOLO(str(args.weights.resolve()))

        if model.task != 'segment':

            raise ValueError('权重必须是实例分割模型')

        model_names = [str(model.names[i]) for i in range(len(model.names))]

        if model_names != names:

            raise ValueError(f'模型类别与 data.yaml 不一致：{model_names} != {names}')

        for index, image in enumerate(inputs):

            relative = image.relative_to(image_dir).with_suffix('.png')

            with Image.open(image) as im:

                shape = (im.height, im.width)

                if im.getexif().get(274, 1) != 1:

                    raise ValueError(f'{image} 存在 EXIF 旋转；请先统一图片及标注方向')

            if args.gt_dir:

                gt = load_map(args.gt_dir / relative)

                gt = check_map(gt, len(names), 'GT')

                if gt.shape != shape:

                    raise ValueError(

                        f'{image} 的 GT 尺寸不等于原图尺寸；不自动缩放类别图'

                    )

                # PNG 没有实例 ID，只能用连通域恢复 GT 实例。

                roi = load_map(args.roi_dir / relative) if args.roi_dir else None

                valid = build_valid_mask(gt, roi)

                gt_masks, gt_classes = class_map_to_instances(

                    gt,

                    len(names),

                    valid=valid,

                )

                # GT 全部保留，不受 --min-instance-area 影响。

            else:

                parts = list(image.parts)

                if 'images' not in parts:

                    raise ValueError('TXT GT 自动匹配需要 images/... 与 labels/... 对称目录')

                pos = len(parts) - 1 - parts[::-1].index('images')

                parts[pos] = 'labels'

                label_path = Path(*parts).with_suffix('.txt')

                gt, gt_masks, gt_classes = yolo_txt_to_instances(

                    label_path,

                    shape,

                    len(names),

                    args.gt_overlap,

                )

                gt_masks, gt_classes = fp_logic.load_gt_masks(label_path, shape, return_classes=True)
                gt_masks = np.asarray(gt_masks, dtype=bool)
                if len(gt_masks) == 0:
                    gt_masks = np.zeros((0, *shape), dtype=bool)

                roi = load_map(args.roi_dir / relative) if args.roi_dir else None

                valid = build_valid_mask(gt, roi)

            kwargs = dict(

                source=str(image),

                conf=args.conf,

                iou=args.nms_iou,

                max_det=args.max_det,

                imgsz=args.imgsz,

                retina_masks=True,

                verbose=False,

                save=False,

            )

            if args.device is not None:

                kwargs['device'] = args.device

            result = model.predict(**kwargs)[0]

            if result.boxes is None or len(result.boxes) == 0:

                pred = np.zeros(shape, dtype=np.uint8)

                pred_masks, pred_classes, pred_scores = empty_instances(shape)

            else:

                if result.masks is None:

                    raise ValueError('存在检测框但没有分割掩码')

                raw_masks = result.masks.data.cpu().numpy()

                pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)

                pred_scores = result.boxes.conf.cpu().numpy().astype(np.float64)

                if raw_masks.shape[1:] != shape:

                    raise ValueError(

                        f'模型 mask 尺寸不是原图尺寸：{raw_masks.shape[1:]} != {shape}；'

                        '请确认 retina_masks=True 生效'

                    )

                pred_masks = raw_masks > 0.5

                pred_masks, pred_classes, pred_scores = filter_small_instances(

                    pred_masks,

                    pred_classes,

                    pred_scores,

                    min_area=args.min_instance_area,

                )

                pred = merge_instances(

                    pred_masks,

                    pred_classes,

                    pred_scores,

                    shape,

                    len(names),

                )

                pred = remove_small_regions(

                    pred,

                    args.min_instance_area,

                )

            pixel_metric.update(pred, gt, roi)

            instance_metric.update(

                pred_masks,

                pred_classes,

                pred_scores,

                gt_masks,

                gt_classes,

                valid,

            )

            samples.append(str(image))

            if args.visualize_dir:

                save_visualizations(

                    image,

                    pred,

                    gt,

                    args.visualize_dir,

                    relative,

                    len(names),

                )

            print(f'评估 {index + 1}/{len(inputs)}: {image.name}', flush=True)

    print('实例计数:', instance_metric.totals)
    pixel_rows = pixel_metric.rows()

    instance_rows = instance_metric.rows()

    rows = combine_rows(pixel_rows, instance_rows)

    report = dict(

        images=pixel_metric.images,

        images_with_valid_pixels=pixel_metric.images_with_valid_pixels,

        valid_pixels=pixel_metric.valid_pixels,

        unknown_pixels=pixel_metric.unknown_pixels,

        status='ok' if pixel_metric.valid_pixels else 'insufficient_evidence',

        pixel_aggregation='sum pixel counts across images, then divide per class',

        instance_miou_definition=INSTANCE_MIOU_DEFINITION,
        instance_matching=(

            'same class; dedup before matching; GT intersection/GT area > '

            f'{fp_logic.GT_COVERAGE_THRESHOLD}; shared tools.fp.find_fp; small FP ignored'

        ),

        fp_definition=(

            'FP is instance-level false-positive rate: '

            'FP_Count / (Prediction_TP + FP_Count) = 1 - Instance_Precision'

        ),

        class_map_encoding='0 background; 1..N foreground; GT255 ignore; pred255 unknown',

        evaluation_scope=(

            'provided ROI' if args.roi_dir else 'full image (no distance restriction)'

        ),

        gt_source=(

            'class PNG; instances approximated by connected components'

            if args.gt_dir

            else 'YOLO TXT polygons; each line is one GT instance'

        ),

        prediction_source=(

            'class PNG; instances approximated by connected components'

            if args.pred_dir

            else 'Ultralytics native instance masks on original image grid'

        ),

        prediction_overlap='score descending, class ID ascending, input index ascending',

        config={

            key: str(value.resolve()) if isinstance(value, Path) else value

            for key, value in vars(args).items()

        },

        samples=samples,

        per_class=rows,

    )

    # 所有样本成功评估后才写报告，避免缺失标注/预测被静默跳过。

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open('w', newline='', encoding='utf-8') as file:

        writer = csv.DictWriter(file, fieldnames=list(rows[0]))

        writer.writeheader()

        writer.writerows(

            {

                key: 'N/A' if value is None else value

                for key, value in row.items()

            }

            for row in rows

        )

    args.output.with_suffix('.json').write_text(

        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),

        encoding='utf-8',

    )

    print(

        '\nclass                     PixelRecall  PixelIoU  InstanceRecall  Instance_mIoU  FP(rate)'

    )

    print('Instance_mIoU: 每个匹配预测与其匹配 GT 并集的标准 IoU，按预测等权平均')
    for row in rows:

        pixel_recall = (

            'N/A' if row['Recall'] is None

            else f"{row['Recall']:.2%}"

        )

        pixel_iou = (

            'N/A' if row['IoU'] is None

            else f"{row['IoU']:.2%}"

        )

        instance_recall = (

            'N/A' if row['Instance_Recall'] is None

            else f"{row['Instance_Recall']:.2%}"

        )

        instance_miou = (

            'N/A' if row['Instance_mIoU'] is None

            else f"{row['Instance_mIoU']:.2%}"

        )

        fp_rate = (

            'N/A' if row['FP'] is None

            else f"{row['FP']:.2%}"

        )

        print(

            f"{row['name']:<24} "

            f"{pixel_recall:>11} "

            f"{pixel_iou:>9} "

            f"{instance_recall:>14} "

            f"{instance_miou:>13} "

            f"{fp_rate:>9}"

        )

    print('\nclass                         GT_TP  Prediction_TP  FP_Count  FN  Precision')
    for row in rows:
        precision = 'N/A' if row['Instance_Precision'] is None else f"{row['Instance_Precision']:.2%}"
        print(f"{row['name']:<28} {row['GT_TP']:>5} {row['Prediction_TP']:>14} "
              f"{row['FP_Count']:>9} {row['Instance_FN']:>3} {precision:>10}")
    print(

        f'有效像素: {pixel_metric.valid_pixels}; '

        f'unknown: {pixel_metric.unknown_pixels}; '

        f'GT覆盖率阈值(严格大于): {fp_logic.GT_COVERAGE_THRESHOLD:.2f}; '

        f'结果: {args.output.resolve()}'

    )

    if not pixel_metric.valid_pixels:

        print('证据不足：没有有效评测像素，不能判定达标。')

if __name__ == '__main__':

    evaluate(parse_args())
