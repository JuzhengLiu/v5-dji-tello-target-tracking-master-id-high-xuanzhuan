"""Drone controller for DJI Tello with autonomous tracking.

Handles connection, movement, and PID-based target following.
"""

import cv2
import numpy as np
import time
from typing import Optional
from enum import Enum

try:
    from djitellopy import Tello

    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False
    print("Warning: djitellopy not installed. Install with: pip install djitellopy")

from src.config import Config, DroneConfig
from src.tracker import TrackedObject
from src.utils import PIDController


class DroneState(Enum):
    """Drone state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    HOVERING = "hovering"
    TRACKING = "tracking"
    LANDING = "landing"
    EMERGENCY = "emergency"


class DroneController:
    """High-level controller for DJI Tello drone.

    Provides tracking automation with PID control.
    """

    def __init__(self, config: Config, drone_config: DroneConfig):
        if not TELLO_AVAILABLE:
            raise ImportError("djitellopy not installed. " "Install with: pip install djitellopy")

        self.config = config
        self.drone_config = drone_config
        self.drone: Optional[Tello] = None
        self.state = DroneState.DISCONNECTED

        # PID controllers for each axis
        self.pid_x = PIDController(*config.pid_x, output_limits=(-100, 100))
        self.pid_y = PIDController(*config.pid_y, output_limits=(-100, 100))
        self.pid_z = PIDController(*config.pid_z, output_limits=(-100, 100))
        self.pid_yaw = PIDController(*config.pid_yaw, output_limits=(-100, 100))

        # Tracking state
        self.tracking_enabled = False
        self.frame_center = (config.frame_width // 2, config.frame_height // 2)

        # Safety
        self.last_command_time = time.time()
        # [修改] 将控制间隔从 0.1s 减少到 0.03s，提高响应频率 (33Hz)
        self.command_interval = 0.03  
        # [新增] 惯性追踪记忆变量
        self.last_valid_target_time = 0
        self.last_known_yaw_cmd = 0  # 记住最后一次的旋转指令
        self.blind_chase_duration = 0.8 # 盲追时间：0.8秒

    def connect(self) -> bool:
        """Connect to Tello drone.

        Returns:
            True if connection successful
        """
        try:
            print("Connecting to Tello...")
            self.drone = Tello()
            self.drone.connect()

            # Get drone info
            battery = self.drone.get_battery()
            print(f"Connected! Battery: {battery}%")

            if battery < self.drone_config.min_battery:
                print(
                    f"WARNING: Battery too low ({battery}%). Minimum: {self.drone_config.min_battery}%"
                )
                return False

            self.state = DroneState.CONNECTED

            # Start video stream
            self.drone.streamon()
            time.sleep(2)  # Wait for stream to stabilize

            return True

        except Exception as e:
            print(f"Failed to connect: {e}")
            self.state = DroneState.DISCONNECTED
            return False

    def disconnect(self) -> None:
        """Disconnect from drone."""
        if self.drone is not None:
            try:
                if self.state not in [DroneState.DISCONNECTED, DroneState.CONNECTED]:
                    self.land()

                self.drone.streamoff()
                self.drone.end()
                print("Disconnected from drone")
            except:
                pass

            self.state = DroneState.DISCONNECTED
            self.drone = None

    def takeoff(self) -> bool:
        """Takeoff sequence.

        Returns:
            True if successful
        """
        if self.state != DroneState.CONNECTED:
            print("Drone not connected")
            return False

        try:
            print("Taking off...")
            self.drone.takeoff()
            time.sleep(3)  # Wait for stabilization
            self.state = DroneState.HOVERING
            print("Takeoff successful")
            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        """Landing sequence.

        Returns:
            True if successful
        """
        if self.state == DroneState.DISCONNECTED:
            return False

        try:
            print("Landing...")
            self.state = DroneState.LANDING
            self.tracking_enabled = False

            self.drone.send_rc_control(0, 0, 0, 0)  # Stop all movement
            time.sleep(0.5)

            self.drone.land()
            time.sleep(3)

            self.state = DroneState.CONNECTED
            print("Landed successfully")
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            return False

    def emergency_stop(self) -> None:
        """Emergency stop - cuts motors immediately."""
        if self.drone is not None:
            try:
                print("EMERGENCY STOP!")
                self.drone.emergency()
                self.state = DroneState.EMERGENCY
            except:
                pass

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from drone camera.

        Returns:
            Frame as numpy array (BGR) or None
        """
        if self.drone is None or self.state == DroneState.DISCONNECTED:
            return None

        try:
            frame = self.drone.get_frame_read().frame
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except:
            return None

    def enable_tracking(self) -> None:
        """Enable autonomous tracking mode."""
        if self.state == DroneState.HOVERING:
            self.tracking_enabled = True
            self.state = DroneState.TRACKING

            # Reset PIDs
            self.pid_x.reset()
            self.pid_y.reset()
            self.pid_z.reset()
            self.pid_yaw.reset()

            print("Tracking enabled")

    def disable_tracking(self) -> None:
        """Disable autonomous tracking."""
        self.tracking_enabled = False
        if self.state == DroneState.TRACKING:
            self.state = DroneState.HOVERING
            self.send_rc_control(0, 0, 0, 0)  # Stop movement
            print("Tracking disabled")


    def track_target(self, target: Optional[TrackedObject]) -> None:
        """Track with aggressive forward speed and inertial chasing."""
        if not self.tracking_enabled or self.state != DroneState.TRACKING:
            return

        current_time = time.time()

        # --- 逻辑分支 1: 目标丢失 (Blind Chase / Coasting) ---
        if target is None or target.disappeared > 0:
            time_since_loss = current_time - self.last_valid_target_time
            
            # 如果刚丢不到 0.8秒，且之前有明显的旋转动作
            if time_since_loss < self.blind_chase_duration and abs(self.last_known_yaw_cmd) > 30:
                # 执行“盲追”：继续按原来的方向旋转，试图把人找回来
                # 但是为了安全，不前进(fb=0)，只旋转
                print(f"Blind Chasing... Yaw: {self.last_known_yaw_cmd}")
                self.send_rc_control(0, 0, 0, int(self.last_known_yaw_cmd))
            else:
                # 丢太久了，或者之前没动，就悬停
                self.send_rc_control(0, 0, 0, 0)
            return

        # --- 逻辑分支 2: 目标存在 (Active Tracking) ---
        
        # 更新最后见到目标的时间
        self.last_valid_target_time = current_time

        # 1. 计算基础误差
        error_x = target.center[0] - self.frame_center[0]
        
        target_y_setpoint = self.frame_center[1] + self.config.height_offset
        error_y = target.center[1] - target_y_setpoint

        # 2. 激进的距离计算 (Z轴)
        if target.class_name == "person":
            bbox_h = target.bbox[3] - target.bbox[1]
            frame_h = self.config.frame_height
            current_ratio = bbox_h / frame_h
            target_ratio = self.config.target_height_ratio
            
            # [修改] 放大倍数从 200 提至 400
            # 举例：若目标只占 0.6，要求 0.75，差值 -0.15
            # -0.15 * 400 = -60 (速度指令)，这就有明显的前进动力了
            error_z = (current_ratio - target_ratio) * 100 * 4
        else:
            # 其他物体逻辑
            bbox_w = target.bbox[2] - target.bbox[0]
            bbox_h = target.bbox[3] - target.bbox[1]
            bbox_area = bbox_w * bbox_h
            frame_area = self.config.frame_width * self.config.frame_height
            current_ratio = bbox_area / frame_area
            target_ratio = self.config.target_area_ratio
            error_z = (current_ratio - target_ratio) * 100 * 2

        # 3. PID 计算
        cmd_lr = self.pid_x.update(error_x)
        cmd_ud = -self.pid_y.update(error_y)
        cmd_fb = -self.pid_z.update(error_z)
        cmd_yaw = self.pid_yaw.update(error_x)

        # 4. [修改] 松绑安全抑制
        # 只有当高度偏差极其巨大 (>150) 时才稍微减速，平时不限制
        if abs(error_y) > 150:
            cmd_fb = cmd_fb * 0.6  # 以前是 0.3，现在允许 60% 速度
        
        # 5. 记录当前的旋转指令 (给盲追用)
        self.last_known_yaw_cmd = cmd_yaw

        # 6. 死区控制 (Deadzone)
        # 对 X/Yaw 死区设小，保证灵敏
        if abs(error_x) < 20: 
            cmd_lr = 0
            cmd_yaw = 0
        if abs(error_y) < 30:
            cmd_ud = 0
        # 对 Z轴(距离)死区设极小，只要不到位就一直微调
        if abs(error_z) < 2: 
            cmd_fb = 0

        # 7. 发送指令
        self.send_rc_control(int(cmd_lr), int(cmd_fb), int(cmd_ud), int(cmd_yaw))

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Send RC control command to drone.

        Args:
            lr: Left/Right velocity (-100 to 100)
            fb: Forward/Backward velocity (-100 to 100)
            ud: Up/Down velocity (-100 to 100)
            yaw: Yaw velocity (-100 to 100)
        """
        if self.drone is None:
            return

        # Rate limiting
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return

        # Clamp values AND convert to int
        lr = int(np.clip(lr, -100, 100))
        fb = int(np.clip(fb, -100, 100))
        ud = int(np.clip(ud, -100, 100))
        yaw = int(np.clip(yaw, -100, 100))

        try:
            self.drone.send_rc_control(lr, fb, ud, yaw)
            self.last_command_time = current_time
        except Exception as e:
            print(f"Failed to send command: {e}")

    # ... (后续代码不变，包括 manual_control, get_telemetry 等)
    
    def manual_control(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0) -> None:
        """Manual control in hovering mode."""
        if self.state == DroneState.HOVERING:
            self.send_rc_control(lr, fb, ud, yaw)

    def get_telemetry(self) -> dict:
        """Get drone telemetry data."""
        if self.drone is None or self.state == DroneState.DISCONNECTED:
            return {}

        try:
            return {
                "battery": self.drone.get_battery(),
                "temperature": self.drone.get_temperature(),
                "height": self.drone.get_height(),
                "barometer": self.drone.get_barometer(),
                "flight_time": self.drone.get_flight_time(),
                "speed_x": self.drone.get_speed_x(),
                "speed_y": self.drone.get_speed_y(),
                "speed_z": self.drone.get_speed_z(),
            }
        except:
            return {}

    def is_connected(self) -> bool:
        return self.state != DroneState.DISCONNECTED

    def is_flying(self) -> bool:
        return self.state in [DroneState.HOVERING, DroneState.TRACKING]


class MockDroneController(DroneController):
    # ... (Mock 类不需要修改，保持原样即可)
    def __init__(self, config: Config, drone_config: DroneConfig):
        super().__init__(config, drone_config)
        self.mock_battery = 100
        self.mock_height = 0
        self.mock_flying = False

    def connect(self) -> bool:
        print("Mock: Connected to simulated drone")
        self.state = DroneState.CONNECTED
        return True

    def disconnect(self) -> None:
        print("Mock: Disconnected")
        self.state = DroneState.DISCONNECTED

    def takeoff(self) -> bool:
        print("Mock: Taking off")
        self.state = DroneState.HOVERING
        self.mock_flying = True
        self.mock_height = 100
        return True

    def land(self) -> bool:
        print("Mock: Landing")
        self.state = DroneState.CONNECTED
        self.mock_flying = False
        self.mock_height = 0
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        return None

    def get_telemetry(self) -> dict:
        return {
            "battery": self.mock_battery,
            "height": self.mock_height,
            "temperature": 25,
            "flight_time": 0,
        }

    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        pass