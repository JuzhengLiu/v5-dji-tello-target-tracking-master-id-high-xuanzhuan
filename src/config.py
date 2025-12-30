"""Configuration management for the tracking system.

Centralizes all parameters and settings.
"""

from dataclasses import dataclass, field
from typing import Tuple
import torch


@dataclass
class Config:
    """Main configuration class for tracking system."""

    # Detection settings
    model_name: str = "yolov8n"  # YOLOv8 nano for speed
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    target_classes: list = field(default_factory=lambda: ["person", "ball", "sports ball"])

    # Device settings
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Tracking settings
    max_disappeared: int = 50  # Max frames before object considered lost
    max_distance: int = 50  # Max distance for centroid association

    # HSV color ranges (backup for color-based tracking)
    hsv_ranges: dict = field(
        default_factory=lambda: {
            "green": ((50, 50, 50), (70, 255, 255)),
            "red": ((0, 50, 50), (20, 255, 255)),
            "blue": ((110, 50, 50), (130, 255, 255)),
        }
    )

    # Drone control settings
    drone_speed: int = 60         # [修改] 提高基础限速上限 (原50)
    movement_threshold: int = 30  # [修改] 减小死区，微小移动也反应
    frame_width: int = 960
    frame_height: int = 720
    
    # 高度偏移 (30cm 约 100像素)
    height_offset: int = 100 

    # [修改] 距离控制：0.75 表示人要占画面高度的 75% (非常近，约1.2米)
    target_height_ratio: float = 0.75
    target_area_ratio: float = 0.7

    # PID controller gains (激进参数)
    pid_x: Tuple[float, float, float] = (0.8, 0.0, 0.1)
    pid_y: Tuple[float, float, float] = (0.6, 0.0, 0.1)
    
    # [修改] 大幅提高前后P值，让它敢于加速
    pid_z: Tuple[float, float, float] = (1.2, 0.0, 0.2) 
    
    # [修改] 极速旋转，P=2.5 意味着稍微偏离一点就全速转
    pid_yaw: Tuple[float, float, float] = (2.5, 0.0, 0.3)   

    # Video settings
    fps: int = 30
    record_path: str = "recordings"

    # Demo settings (webcam)
    webcam_id: int = 0
    display_fps: bool = True
    display_tracking_info: bool = True


@dataclass
class DroneConfig:
    """Tello drone specific configuration."""

    # Connection settings
    tello_ip: str = "192.168.10.1"
    tello_port: int = 8889
    local_ip: str = ""
    local_port: int = 9000

    # Safety limits
    max_speed: int = 100
    min_battery: int = 10  # Land if battery below this
    max_tilt: int = 30  # Max tilt angle in degrees

    # Command timeouts
    command_timeout: float = 10.0
    video_timeout: float = 5.0

    # Flight parameters
    takeoff_height: int = 80  # cm
    default_height: int = 100  # cm
    max_height: int = 500  # cm


class ConfigBuilder:
    """Builder pattern for creating custom configurations."""

    def __init__(self):
        self.config = Config()

    def with_model(self, model_name: str) -> "ConfigBuilder":
        """Set detection model."""
        self.config.model_name = model_name
        return self

    def with_confidence(self, threshold: float) -> "ConfigBuilder":
        """Set confidence threshold."""
        self.config.confidence_threshold = threshold
        return self

    def with_device(self, device: str) -> "ConfigBuilder":
        """Set computing device."""
        self.config.device = device
        return self

    def with_target_classes(self, classes: list) -> "ConfigBuilder":
        """Set target detection classes."""
        self.config.target_classes = classes
        return self

    def with_drone_speed(self, speed: int) -> "ConfigBuilder":
        """Set drone movement speed."""
        self.config.drone_speed = speed
        return self

    def build(self) -> Config:
        """Return the built configuration."""
        return self.config


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_webcam_config() -> Config:
    """Get configuration optimized for webcam demo."""
    config = Config()
    config.model_name = "yolov8n"  # Fastest model
    config.confidence_threshold = 0.4  # Lower threshold for demo
    config.display_fps = True
    config.display_tracking_info = True
    return config


def get_drone_config() -> Tuple[Config, DroneConfig]:
    """Get configuration optimized for drone tracking."""
    config = Config()
    config.model_name = "yolov8s"  # Better accuracy for outdoor
    config.confidence_threshold = 0.6
    config.drone_speed = 50

    drone_config = DroneConfig()

    return config, drone_config