import cv2
import numpy as np


class TemplateTracker:
    def __init__(self, frame, bbox, method="BOOSTING"):
        """
        Initialize the tracker with the provided frame, bounding box, and tracking method.

        Args:
            frame (ndarray): The initial frame in which the object is located.
            bbox (tuple): Bounding box in (x, y, w, h) format.
            method (str): The tracking method, either 'BOOSTING' or 'TM_CCOEFF_NORMED'.
        """
        self.frame = frame
        self.bbox = bbox
        self.method = method

        if self.method == "BOOSTING":
            self.tracker = cv2.legacy.TrackerBoosting_create()
            self.ok = self.tracker.init(self.frame, self.bbox)
            if not self.ok:
                raise RuntimeError(
                    "Failed to initialize BOOSTING tracker after re-selection."
                )
        elif self.method == "TM_CCOEFF_NORMED":
            bbox_w, bbox_h = int(self.bbox[2]), int(self.bbox[3])
            self.template = self.frame[
                self.bbox[1] : self.bbox[1] + bbox_h,
                self.bbox[0] : self.bbox[0] + bbox_w,
            ]
            self.template_shape = (bbox_w, bbox_h)
            self.tracker = None
        else:
            raise ValueError(
                "Unsupported tracking method. Choose 'BOOSTING' or 'TM_CCOEFF_NORMED'."
            )

    def track(self, new_frame):
        """
        Track the object in the new frame and return the updated bounding box.

        Args:
            new_frame (ndarray): The new frame in which the object needs to be tracked.

        Returns:
            ndarray: left, top, right, bottom.
        """
        if self.method == "BOOSTING":
            ok, bbox_result = self.tracker.update(new_frame)
            if ok:
                x, y, w, h = bbox_result
                top, left = int(y), int(x)
                bottom, right = int(y + h), int(x + w)
                return np.array([left, top, right, bottom])
            else:
                return None
        elif self.method == "TM_CCOEFF_NORMED":
            w, h = self.template_shape
            res = cv2.matchTemplate(new_frame, self.template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(
                res
            )  # 模板匹配效果不佳（例如模板与图像中的内容不匹配），max_loc 依然会有一个值，只不过此时的 max_val 可能会非常低，表示匹配不理想。可以增加阈值修改！！
            top, left = max_loc[1], max_loc[0]
            bottom, right = top + h, left + w
            return np.array([left, top, right, bottom])


if __name__ == "__main__":
    import cv2
    import os

    output_dir = "template_tracking_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_BOOSTING = os.path.join(output_dir, "BOOSTING_result.mp4")
    output_file_CCOEFF = os.path.join(output_dir, "TM_CCOEFF_NORMED.mp4")

    video = cv2.VideoCapture("Video_0009839_20230908-170751.mp4")
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    videowriter_BOOSTING = cv2.VideoWriter(
        output_file_BOOSTING,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    videowriter_CCOEFF = cv2.VideoWriter(
        output_file_CCOEFF,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    ret, frame = video.read()
    if not ret:
        print("无法读取视频帧")
        exit()

    # 选择初始目标区域
    roi = cv2.selectROI(frame, False)
    print(f"bbox_xywh: {roi}")

    tracker_boosting = TemplateTracker(frame, roi, method="BOOSTING")
    tracker_ccoeff = TemplateTracker(frame, roi, method="TM_CCOEFF_NORMED")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_b = frame.copy()
        frame_c = frame.copy()

        tracker_boosting_ltbrbbox = tracker_boosting.track(frame)
        print(f"tracker_boosting_ltbrbbox: {tracker_boosting_ltbrbbox}")
        tracker_ccoeff_ltbrbbox = tracker_ccoeff.track(frame)
        print(f"tracker_ccoeff_ltbrbbox: {tracker_ccoeff_ltbrbbox}")

        cv2.rectangle(
            frame_b,
            tracker_boosting_ltbrbbox[:2],
            tracker_boosting_ltbrbbox[2:],
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            frame_c,
            tracker_ccoeff_ltbrbbox[:2],
            tracker_ccoeff_ltbrbbox[2:],
            (0, 255, 0),
            2,
        )

        videowriter_BOOSTING.write(frame_b)
        videowriter_CCOEFF.write(frame_c)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()

    videowriter_BOOSTING.release()
    videowriter_CCOEFF.release()

    print("Done!")
