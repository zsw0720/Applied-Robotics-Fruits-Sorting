import cv2
import numpy as np
import socket
import struct
import time
from ultralytics import YOLO

# =====================================================================
# 🛠️ 1. Core Configuration & Network Initialization
# =====================================================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"📡 UDP Transmitter started, target port: {UDP_PORT}")

# =====================================================================
# 🧠 2. Load YOLO Deep Learning Model
# =====================================================================
model_path = 'best.pt'  # Ensure your YOLO weights file is in the same directory
print(f"⏳ Loading YOLO model ({model_path})...")
try:
    model = YOLO(model_path)
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model, check path. Error: {e}")
    exit()

# =====================================================================
# 👁️ 3. Initialize Camera (Physical Environment)
# =====================================================================
cam_index = 2  # External cameras are usually 1 or 2, change to 0 if it fails to open
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print(f"❌ WARNING: Cannot open camera with device index {cam_index}! Check USB connection.")
    exit()

print("👁️ Camera connected. Vision and control loop started. Press 'q' to exit.")

# =====================================================================
# 🚀 4. Main Control Loop (Vision-to-Control Loop)
# =====================================================================
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            # [Heartbeat Mechanism] If no frame is read, send safe standby coordinates to prevent Simulink hardware lock
            sock.sendto(struct.pack('<dddd', 0.45, 0.0, 0.49, 0.0), (UDP_IP, UDP_PORT))
            time.sleep(0.033)  # Maintain approx. 30Hz transmission rate
            continue

        # Initialize standby coordinates (Home Position)
        target_x, target_y, target_z = 0.45, 0.0, 0.49
        fruit_id = 0.0
        name = "None"

        # ---------------------------------------------------------
        # 🔍 A. YOLO Real-time Inference (Set conf threshold to filter noise)
        # ---------------------------------------------------------
        results = model.predict(source=frame, conf=0.6, verbose=False)

        # ---------------------------------------------------------
        # 📊 B. Parse Recognition Results and Extract Parameters
        # ---------------------------------------------------------
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                # Only take the first target with the highest confidence to prevent logic conflicts
                box = boxes[0]

                # Extract bounding box coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Extract Class ID and Confidence
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                # Calculate target center pixel coordinates (cx, cy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # ---------------------------------------------------------
                # 🎯 C. Core Mapping Logic: YOLO ID -> Stateflow ID
                # (Your trained YOLO model: 0=Banana, 1=Strawberry, 2=Tomato)
                # ---------------------------------------------------------
                if cls_id == 0:
                    fruit_id = 2.0  # Map to Simulink Banana
                    name = "Banana"
                elif cls_id == 1:
                    fruit_id = 1.0  # Map to Simulink Tomato
                    name = "Tomato"
                elif cls_id == 2:
                    fruit_id = 3.0  # Map to Simulink Strawberry
                    name = "Strawberry"

                # ---------------------------------------------------------
                # 📐 D. Pixel to Physical Coordinates (Pinhole Camera Mapping)
                # ---------------------------------------------------------
                target_x = 0.50 - (cy - 240) * 0.001
                target_y = 0.00 - (cx - 320) * 0.001

                # Hardware descent height (Start at 0.10m to test, lower to 0.08 etc. if needed)
                target_z = 0.10

                # ---------------------------------------------------------
                # 🛡️ E. Physical Limit Safety Lock (Ultimate defense for real robot)
                # ---------------------------------------------------------
                target_x = max(0.25, min(0.65, target_x))  # Prevent hitting base or reaching out of bounds
                target_y = max(-0.40, min(0.40, target_y))  # Limit left/right swing range
                target_z = max(0.05, min(0.60, target_z))  # Prevent smashing desk causing joint limit errors

                # ---------------------------------------------------------
                # 🎨 F. Draw UI with confidence rate on video stream
                # ---------------------------------------------------------
                # Format display text: Class Name (Confidence %): X=coord, Y=coord
                display_text = f"{name} ({conf * 100:.1f}%): X={target_x:.2f}, Y={target_y:.2f}"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, display_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Process only one highest confidence target per loop, break after parsing
                break

                # ---------------------------------------------------------
        # 🚀 5. Send final coordinates via UDP to Simulink hardware
        # ---------------------------------------------------------
        msg = struct.pack('<dddd', target_x, target_y, target_z, fruit_id)
        sock.sendto(msg, (UDP_IP, UDP_PORT))

        # ---------------------------------------------------------
        # 📺 6. Display real-time rendered frame
        # ---------------------------------------------------------
        cv2.imshow("QArm Deep Learning Vision Node", frame)

        # Press 'q' to safely exit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Program forcibly interrupted by user.")

finally:
    # 🧹 Clean up system resources
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    print("🔌 Camera and UDP resources released, program exited safely.")