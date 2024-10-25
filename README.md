# 适用于RK3588的模板匹配跟踪与深度学习跟踪

## 介绍

文件夹安排如下：
**tracker**
.
├── README.md
├── __init__.py
├── deepvisionTrack
│   ├── DeepSORT
│   │   ├── deep
│   │   │   ├── __init__.py
│   │   │   └── feature_extractor.py
│   │   ├── deep_sort.py
│   │   └── sort
│   │       ├── __init__.py
│   │       ├── detection.py
│   │       ├── iou_matching.py
│   │       ├── kalman_filter-modified.py
│   │       ├── kalman_filter.py
│   │       ├── linear_assignment.py
│   │       ├── nn_matching.py
│   │       ├── preprocessing.py
│   │       ├── track.py
│   │       └── tracker.py
│   ├── YOlOv8
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── func.py
│   ├── rknnModel
│   │   ├── deepsort.rknn
│   │   ├── yolov8m.rknn
│   │   ├── yolov8n.rknn
│   │   └── yolov8s.rknn
│   └── visionar.py
├── templaterTrack
│   └── templar.py
└── uitls
│   ├── draw.py
│   └── log.py

`deepvisionTrack`：深度学习方法。
`templaterTrack`：模板匹配方法。
`utils`：通用工具。

## 使用方法

1. 模板匹配跟踪

    ```python
    from tacker.templaterTrack.templar import TemplateTracker

    method = ["BOOSTING", "TM_CCOEFF_NORMED"] # 可选两种，默认为BOOSTING
    # ==>初始化：frame为首帧，roi为框选结果
    # frame(ndarray): ret, frame = video.read()
    # roi(tuple): (x, y, w, h) format, roi = cv2.selectROI(frame, False)
    tracker_boosting = TemplateTracker(frame, roi, method=method[0])
    # ==>跟踪：frame为当前帧，返回跟踪结果
    # ltbrbbox(ndarray): np.array([left, top, right, bottom])
    ltbrbbox = tracker_boosting.track(frame)
    ```

2. 深度学习跟踪

    ```python
    from tacker.deepvisionTrack.visionar import VisionTracker

    tracking = True # 默认为True，输出跟踪结果。设置为False时只输出YOLO检测结果。

    # ==>初始化：TPEs，目标检测线程池设置参数，默认为2。
    # 考虑到RK3588只有3块3588，只有目标检测推理时可设为3。大于3时速度提升效果不明显。
    # 同时加入deepsort特征推理时，建议设置为2，NPU会自动把空闲NPU绑定到deepsort推理。
    visioner = VisionTracker(TPEs=2)              # 初始化rknn模型、线程池
    visioner.set_tracking_mode(tracking=tracking) # 可选参数，设置跟踪还是只检测
    # visioner.set_target_id(id=1)                  # 可选参数，指定跟踪id，由手动输入，id为deepsort跟踪id
    # ==>跟踪：frame为当前帧，返回跟踪结果
    outputs = visioner.detect_and_track(frame)
    # 当为跟踪时：outputs(dict): {"frame": cur_frame,"boxes": tracked_boxes，"classes": tracked_ids}，classes返回的是多目标跟踪的id
    # 当为检测时：outputs(dict): {"frame": cur_frame, "boxes": ltrb_boxes, "classes": cls_ids}，classes返回的是目标的类别
    # "boxes":N*4维ndarray数组，N表示检测到的目标数目，4表示为[left, top, right, bottom]
    ```