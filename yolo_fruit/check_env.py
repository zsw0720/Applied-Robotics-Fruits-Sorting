import torch
import cv2

print("torch version:", torch.__version__)
print("opencv version:", cv2.__version__)
print("cuda available:", torch.cuda.is_available())

has_mps = hasattr(torch.backends, "mps")
print("mps backend exists:", has_mps)
if has_mps:
    print("mps available:", torch.backends.mps.is_available())
