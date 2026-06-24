# 🍓 Vision-Based Soft-Robotic Fruit Sorting System

![Robotics](https://img.shields.io/badge/Robotics-Quanser_QArm-red.svg)
![Computer Vision](https://img.shields.io/badge/Vision-YOLOv8-blue.svg)
![Control](https://img.shields.io/badge/Control-Simulink_Stateflow-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

> **A fully autonomous, dual-mode cyber-physical system for non-destructive agricultural sorting, leveraging deep learning and compliant soft-robotic manipulation.** > 
> *Developed as part of the Postgraduate Taught (PGT) Applied Robotics curriculum at the University of Birmingham (Group 6).*

---

## 📖 About The Project

Traditional rigid robotic end-effectors suffer from an unacceptably high bruising rate when handling delicate agricultural products. This project successfully architectures, simulates, and physically deploys a decoupled, vision-driven sorting system using the **Quanser QArm**. 

By transitioning from a rigid carbon-fiber gripper to a custom-designed **hyper-elastic silicone soft gripper**, and integrating it with a real-time **YOLOv8** perception node, the system achieves damage-free manipulation. A rigorous multi-rate control architecture engineered in **MATLAB/Simulink** bridges the asynchronous AI vision stream with the deterministic hardware controller.

### 🎯 Key Achievements
* **Zero-Damage Handling:** Validated through Simscape mechanical analysis and physical trials, drastically reducing peak contact pressure on fragile targets like strawberries.
* **High Reliability:** Achieved an overall autonomous sorting success rate of **93.3%** over 90 empirical trials.
* **Dual-Mode Teleoperation:** Features a prioritized keyboard-based remote protocol for manual override, ensuring industrial safety.

---

## 🧠 System Architecture

The system operates on a decoupled Cyber-Physical architecture, split into two primary nodes communicating via asynchronous UDP:

1. **Python Vision Node (AI Perception):** * Captures Eye-in-Hand camera feed.
   * Runs custom-trained YOLOv8 inference (mAP@0.5 = 0.945) to classify targets (Strawberry, Tomato, Banana).
   * Calculates 2D bounding box centroids and performs Pinhole Camera mapping to estimate 3D Cartesian coordinates.
2. **Simulink Hardware Node (Kinematics & Control):**
   * Acts as the deterministic Finite State Machine (FSM) utilizing Stateflow.
   * Employs a **Zero-Order Hold (ZOH)** rate transition block to synchronize the 33 Hz UDP vision data with the 200 Hz hardware DAQ loop.
   * Computes Inverse Kinematics (IK) and executes cascaded PID joint control.

---

## 🛠️ Tech Stack & Hardware BOM

### Hardware
* **Manipulator:** Quanser QArm (4-DOF)
* **Vision Sensor:** Intel RealSense RGB-D Camera (Eye-in-Hand configuration)
* **End-Effector:** Custom Platinum-cure Silicone Soft Fingers
* **Data Acquisition:** Quanser QFLEX 2 USB Interface

### Software
* **Computer Vision:** Python 3.10, OpenCV, Ultralytics YOLOv8
* **Control Systems:** MATLAB, Simulink, Stateflow, QUARC Hardware-in-the-Loop (HIL)
* **Mechanical Simulation:** Simscape Multibody, SolidWorks (FEA)

---

## 🕹️ Dual-Mode Control Protocol

To satisfy industrial failsafe requirements, the system features a **Supervisory Control Multiplexer**:

* **Autonomous Mode (Priority 3):** Fully driven by YOLOv8 vision triggers and FSM routing logic.
* **Manual Override Mode (Priority 2):** A standardized keyboard mapping protocol allows an operator to interrupt the AI pipeline. Pressing `G` (Grasp) or `R` (Release) immediately suspends the autonomous sequence, mapping manual PWM stroke limits directly to the soft gripper.
* **Watchdog Timer (Priority 1):** Automatically halts the manipulator (`Safe_Halt` state) if UDP packets are dropped for >0.5 seconds.

---

## 🚀 Installation & Usage

### 1. Vision Node Setup (Python)
```bash
git clone [https://github.com/zsw0720/Applied-Robotics-Fruits-Sorting.git](https://github.com/zsw0720/Applied-Robotics-Fruits-Sorting.git)
cd Applied-Robotics-Fruits-Sorting/Vision_Node
pip install -r requirements.txt
# Ensure the custom weights 'best.pt' is in the directory
python yolo_udp_transmitter.py
