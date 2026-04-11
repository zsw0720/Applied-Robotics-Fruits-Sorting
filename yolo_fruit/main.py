import cv2
from camera_module import open_camera, read_frame, release_camera
from yolo_detector import YoloFruitDetector


def main():
    cap = open_camera(camera_index=2, width=1280, height=720)
    detector = YoloFruitDetector("best.pt")

    print("按 q 退出")

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            print("读取摄像头失败")
            break

        vis, detections = detector.detect(frame, conf=0.35)

        # 只显示水果相关类别计数
        banana_count = sum(1 for d in detections if d["label"] == "banana")
        apple_count = sum(1 for d in detections if d["label"] == "strawberry")
        orange_count = sum(1 for d in detections if d["label"] == "tomato")

        info = f"banana:{banana_count} strawberry:{apple_count} tomato:{orange_count}"
        cv2.putText(
            vis,
            info,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLO Fruit Detection", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera(cap)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
