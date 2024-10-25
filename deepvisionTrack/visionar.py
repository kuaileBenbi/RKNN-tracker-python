from DeepSORT.deep_sort import DeepSort
from YOlOv8.detector import detectExecutor
from YOlOv8.func import expand_bbox_xyxy, myFunc
from queue import Empty
import os

current_dir = os.path.dirname(__file__)
det_model = os.path.join(current_dir, "rknnModel/yolov8s.rknn")
extractor_model = os.path.join(current_dir, "rknnModel/deepsort.rknn")


class VisionTracker:
    def __init__(self, TPEs=2):
        self.detector = detectExecutor(det_model=det_model, TPEs=TPEs, func=myFunc)
        self.tracker = DeepSort(
            extractor_model=extractor_model,
            max_dist=0.2,
            min_confidence=0.5,
            nms_max_overlap=0.5,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
        )
        self.tracking = True
        self.target_id = None
        self.frame_count = 0
        self.TPEs = TPEs

    def set_tracking_mode(self, tracking):
        """切换是否开启跟踪模式"""
        self.tracking = tracking

    def set_target_id(self, target_id):
        """设置要跟踪的目标ID"""
        self.target_id = target_id

    def detect_and_track(self, img):
        self.detector.put(img)
        try:
            # 尝试从检测器线程池中获取检测结果
            (cur_frame, results), flag = self.detector.get_nowait()  # 使用非阻塞获取
            if not flag:
                return None  # 检测未完成，继续处理下一帧

            ltrb_boxes = results["ltrb_boxes"]
            cls_ids = results["classes_id"]  # 此id非track的id
            cls_conf = results["scores"]

            # 如果未启用跟踪，仅返回检测结果
            if not self.tracking:
                return {"frame": cur_frame, "boxes": ltrb_boxes, "classes": cls_ids}

            # 如果启用了跟踪，执行跟踪操作
            bbox_cxcywh = expand_bbox_xyxy(ltrb_boxes)
            outputs, _ = self.tracker.update(bbox_cxcywh, cls_conf, cls_ids, cur_frame)

            # 如果设置了目标ID，则过滤跟踪结果
            if self.target_id is not None:
                outputs = [obj for obj in outputs if obj[4] == self.target_id]

            if outputs is not None and len(outputs) > 0:
                tracked_boxes = outputs[:, :4]  # 前4列是坐标
                tracked_ids = outputs[:, 4]  # 第5列是类别ID
                return {
                    "frame": cur_frame,
                    "boxes": tracked_boxes,
                    "classes": tracked_ids,
                }
            else:
                return None

        except Empty:
            # 如果检测器池中还没有结果，返回 None，继续输入帧
            return None


if __name__ == "__main__":
    import cv2
    import os

    output_dir = "vision_tracking_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tracking = True

    if tracking:
        output_file = os.path.join(output_dir, "track_result.mp4")
    else:
        output_file = os.path.join(output_dir, "detect_result.mp4")

    video = cv2.VideoCapture("test.mp4")
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    videowriter = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    visioner = VisionTracker(TPEs=2)
    visioner.set_tracking_mode(tracking=tracking)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        outputs = visioner.detect_and_track(frame)
        if outputs is not None:
            curframe = outputs["frame"]
            boxes = outputs["boxes"]
            classids = outputs["classes"]

            for i, box in enumerate(boxes):
                l, t, b, r = (int(x) for x in box)
                cv2.rectangle(
                    curframe,
                    (l, t),
                    (b, r),
                    (0, 255, 0),
                    2,
                )

            videowriter.write(curframe)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    video.release()
    cv2.destroyAllWindows()

    videowriter.release()

    print("Done!")
