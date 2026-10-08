#!/usr/bin/env python3

import os
import cv2
import numpy as np
from tqdm import tqdm

DATASET = "./datasets/data_795"



IMAGE_DIR = os.path.join(
    DATASET,
    "images/train"
)

LABEL_DIR = os.path.join(
    DATASET,
    "labels/train"
)


OUTPUT_DIR = os.path.join(
    DATASET,
    "visualize"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# 类别名称
NAMES = [
    "puddle"
]


# ==================================================
# 读取 YOLO-Seg 标签
# ==================================================

def read_yolo_seg(label_path, img_shape):

    h, w = img_shape[:2]

    polygons = []


    if not os.path.exists(label_path):
        return polygons


    with open(
        label_path,
        "r"
    ) as f:

        lines = f.readlines()


    for line_id, line in enumerate(lines):

        data = line.strip().split()


        if len(data) < 7:
            continue


        cls = int(data[0])


        points = np.array(
            data[1:],
            dtype=np.float32
        ).reshape(-1, 2)


        # 归一化坐标转像素坐标

        points[:, 0] *= w
        points[:, 1] *= h


        points = points.astype(
            np.int32
        )


        polygons.append(
            {
                "id": line_id + 1,

                "cls": cls,

                "points": points
            }
        )


    return polygons



# ==================================================
# 绘制mask和编号
# ==================================================

def draw_mask(img, polygons):


    overlay = img.copy()


    for item in polygons:


        puddle_id = item["id"]

        cls = item["cls"]

        pts = item["points"]


        # -------------------------
        # 填充mask
        # -------------------------

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 255, 0)
        )


        # -------------------------
        # 绘制边界
        # -------------------------

        cv2.polylines(
            img,
            [pts],
            True,
            (0, 0, 255),
            2
        )


        # -------------------------
        # 计算中心点
        # -------------------------

        M = cv2.moments(
            pts
        )


        if M["m00"] != 0:

            cx = int(
                M["m10"] / M["m00"]
            )

            cy = int(
                M["m01"] / M["m00"]
            )

        else:

            cx, cy = pts[0]


        # -------------------------
        # 显示编号
        # -------------------------

        text = f"puddle{puddle_id}"


        cv2.putText(
            img,
            text,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 255),
            3,
            cv2.LINE_AA
        )


    # 半透明mask

    img = cv2.addWeighted(
        overlay,
        0.35,
        img,
        0.65,
        0
    )


    return img



# ==================================================
# 主程序
# ==================================================

def main():


    images = [
        x for x in os.listdir(IMAGE_DIR)
        if x.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]


    print(
        "图片数量:",
        len(images)
    )


    for name in tqdm(images):


        img_path = os.path.join(
            IMAGE_DIR,
            name
        )


        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(name)[0] + ".txt"
        )


        img = cv2.imread(
            img_path
        )


        if img is None:
            continue



        polygons = read_yolo_seg(
            label_path,
            img.shape
        )


        # ==================================================
        # 生成标注图
        # ==================================================

        vis = draw_mask(
            img.copy(),
            polygons
        )


        # ==================================================
        # 左侧原图 + 右侧标注图
        # ==================================================

        original = img.copy()


        # 保证高度一致

        if original.shape[0] != vis.shape[0]:

            vis = cv2.resize(
                vis,
                (
                    int(
                        vis.shape[1]
                        *
                        original.shape[0]
                        /
                        vis.shape[0]
                    ),
                    original.shape[0]
                )
            )


        combined = cv2.hconcat(
            [
                original,
                vis
            ]
        )


        save_path = os.path.join(
            OUTPUT_DIR,
            name
        )


        cv2.imwrite(
            save_path,
            combined
        )


    print("\n完成")

    print(
        "结果:",
        OUTPUT_DIR
    )



if __name__ == "__main__":

    main()