import cv2
import platform


def _get_backend_candidates():
    system_name = platform.system()

    if system_name == "Darwin":      # macOS
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif system_name == "Windows":   # Windows
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        return [cv2.CAP_ANY]


def open_camera(camera_index=0, width=1280, height=720):
    backends = _get_backend_candidates()
    last_error = None

    for backend in backends:
        try:
            if backend == cv2.CAP_ANY:
                cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index, backend)

            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap

            if cap is not None:
                cap.release()

        except Exception as e:
            last_error = e

    raise RuntimeError(f"无法打开摄像头。最后错误: {last_error}")


def read_frame(cap):
    if cap is None or not cap.isOpened():
        return False, None
    return cap.read()


def release_camera(cap):
    if cap is not None:
        cap.release()
