# ============================================================
# RGB-IR 棋盘标定配准 — 全局配置文件
# ============================================================
# 可通过修改 CHECKERBOARD 适配不同尺寸的棋盘标定板
# 也可设为 None 启用自动检测模式
# ============================================================

import os

import cv2


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


def _configured_path(env_name, *default_parts):
    """Return an absolute path from the environment or a project-relative default."""
    configured = os.getenv(env_name)
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join(PROJECT_PATH, *default_parts)

# ----------------------------------------------------------
# 棋盘格参数
# ----------------------------------------------------------
# 设为 (cols, rows) 使用固定尺寸；设为 None 启用自动检测
# 自动检测会依次尝试 (9,6)→(8,6)→(7,5)→(6,5)→(5,4)，取最大有效尺寸
CHECKERBOARD = (11, 6)              # None=自动检测, 或手动设为 e.g. (5, 4)

SQUARE_SIZE = 25                 # 每格实际尺寸 (mm)

# ----------------------------------------------------------
# 标定图像路径
# ----------------------------------------------------------
RGB_PATH = _configured_path("RGB_IR_CALIBRATION_RGB_PATH", "calibration", "rgb")
IR_PATH = _configured_path("RGB_IR_CALIBRATION_IR_PATH", "calibration", "ir")

# 尺寸均按 (width, height) 记录。IR 原图 1920x1080 顺时针旋转后为 1080x1920。
# 当前 processing_workspace 已把两路图像规范化到相同的横向分辨率。
CALIBRATION_RGB_REFERENCE_SIZE = (1920, 1080)
CALIBRATION_IR_REFERENCE_SIZE = (1920, 1080)
CALIBRATION_IR_ROTATION = "none"

# 运行图与标定参考图的宽高比相对差异不得超过 0.5%。超过该值通常意味着
# 裁剪或 FOV 已改变，不能再把差异当成单纯分辨率缩放。
ASPECT_RATIO_TOLERANCE = 0.005

# 测试图像路径
RGB_TEST = _configured_path("RGB_IR_TEST_RGB_PATH", "test", "rgb_test", "001.png")
IR_TEST = _configured_path("RGB_IR_TEST_IR_PATH", "test", "ir_test", "001.png")

# 最终输出目录（按当前项目要求保存到项目内部）
SAVE_PATH = _configured_path("RGB_IR_OUTPUT_PATH", "output")

# ----------------------------------------------------------
# 已有模型的实际应用路径
# ----------------------------------------------------------
# 将待处理图像分别放入 input/rgb 和 input/ir；同名文件视为一对，
# 文件扩展名可以不同，例如 rgb/001.jpg 与 ir/001.png。
APPLY_INPUT_PATH = os.path.join(PROJECT_PATH, "input")
APPLY_RGB_PATH = os.path.join(APPLY_INPUT_PATH, "rgb")
APPLY_IR_PATH = os.path.join(APPLY_INPUT_PATH, "ir")

CALIBRATION_MODEL_PATH = os.path.join(SAVE_PATH, "calibration.yaml")

APPLY_OUTPUT_PATH = os.path.join(SAVE_PATH, "registered")
APPLY_RGB_OUTPUT_PATH = os.path.join(APPLY_OUTPUT_PATH, "rgb")
APPLY_IR_OUTPUT_PATH = os.path.join(APPLY_OUTPUT_PATH, "ir")
APPLY_PREVIEW_OUTPUT_PATH = os.path.join(APPLY_OUTPUT_PATH, "preview")
APPLY_REPORT_PATH = os.path.join(APPLY_OUTPUT_PATH, "application_report.yaml")

# 应用结果统一保存为无损 PNG；预览图中 RGB 与 IR 各占 50%。
APPLY_OUTPUT_EXTENSION = ".png"
APPLY_PREVIEW_RGB_WEIGHT = 0.5

# ----------------------------------------------------------
# 支持的图像格式
# ----------------------------------------------------------
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# ----------------------------------------------------------
# 自适应预处理参数
# ----------------------------------------------------------
CLAHE_CLIP_LIMIT = 5.0           # 默认 CLAHE 对比度限制
CLAHE_CLIP_STRONG = 8.0          # 暗图用强 CLAHE
CLAHE_CLIP_LIGHT = 2.0           # 亮图用弱 CLAHE
CLAHE_TILE_SIZE = (8, 8)         # CLAHE 网格大小

# 自适应阈值：灰度均值低于此值 → 强增强
DARK_THRESHOLD = 60
# 灰度均值高于此值 → 弱增强
BRIGHT_THRESHOLD = 180

# ----------------------------------------------------------
# 角点检测参数
# ----------------------------------------------------------
SUBPIX_WINDOW = (11, 11)         # 亚像素精调窗口
SUBPIX_ZONE = (-1, -1)           # 死区
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 动态尺度：根据图像最小边长自动计算基准分辨率
# 基准分辨率 = min(w, h) / SCALE_BASE，然后 × [0.5, 1.0, 2.0] 三级
SCALE_BASE = 500                 # 基准分辨率对应 ~500px

# ----------------------------------------------------------
# Homography 计算参数
# ----------------------------------------------------------
RANSAC_THRESHOLD = 3.0           # RANSAC 阈值 (pixels)
RANSAC_MAX_ITERS = 2000          # 最大迭代次数
RANSAC_CONFIDENCE = 0.995        # 置信度

# 单对角点质量门槛；任何角点异常都会拒绝整对，不删除单个角点。
# 旧门槛以 RGB 短边 720 px 为基准；当前短边为 1080 px。像素误差
# 随线性分辨率等效缩放，保持相同的物理/归一化精度要求。
CALIBRATION_THRESHOLD_BASE_SHORT_EDGE = 720
CALIBRATION_THRESHOLD_SCALE = (
    min(CALIBRATION_RGB_REFERENCE_SIZE) / CALIBRATION_THRESHOLD_BASE_SHORT_EDGE
)
PAIR_MAX_RMSE_PX = 0.6 * CALIBRATION_THRESHOLD_SCALE
PAIR_MAX_P95_PX = 1.0 * CALIBRATION_THRESHOLD_SCALE
PAIR_MAX_ERROR_PX = 1.5 * CALIBRATION_THRESHOLD_SCALE

# 全局模型与留一组交叉验证验收门槛（单位为标定 RGB 像素）。
GLOBAL_MAX_RMSE_PX = 0.5 * CALIBRATION_THRESHOLD_SCALE
GLOBAL_MAX_P95_PX = 0.8 * CALIBRATION_THRESHOLD_SCALE
GLOBAL_MAX_ERROR_PX = 1.5 * CALIBRATION_THRESHOLD_SCALE
LOOCV_MAX_RMSE_PX = 0.8 * CALIBRATION_THRESHOLD_SCALE

# ----------------------------------------------------------
# 测试图像精配准参数
# ----------------------------------------------------------
# checkerboard: 仅用于待配准图像本身仍包含完整棋盘的逐点验证
# features:     对齐整幅场景的稳定特征
# calibration:  标定完成后，对不含棋盘的普通场景应用保存的标定矩阵（默认）
REGISTRATION_MODE = "calibration"

# 自动选择角点顺序时，优先选择非镜像且旋转幅度不超过该值的解。
# 两相机若确实倒置安装，可改为 180。
CHECKERBOARD_MAX_ROTATION_DEG = 90.0

# 特征模式参数
ENABLE_PAIR_REFINEMENT = True
FEATURE_MAX_COUNT = 12000
FEATURE_RATIO_TEST = 0.70
REFINE_RANSAC_THRESHOLD = 2.0
REFINE_MIN_INLIERS = 40
REFINE_MIN_INLIER_RATIO = 0.25
REFINE_MIN_COVERAGE = 0.15
REFINE_MAX_MEDIAN_ERROR = 2.0

# ----------------------------------------------------------
# 质量控制
# ----------------------------------------------------------
MIN_VALID_PAIRS = 18             # 高精度全局模型的最少合格标定对数
MIN_PROVISIONAL_VALID_PAIRS = 4  # 当前 12 对试运行允许应用模型的最低有效对数

# 棋盘姿态多样性只作为警告，不阻止模型应用。
CALIBRATION_DIVERSITY_MIN_CENTER_SPAN_RATIO = 0.05
CALIBRATION_DIVERSITY_MIN_AREA_RATIO = 1.05

# 配准有效性：warp 后有效像素比例低于此值 → 警告
MIN_VALID_WARP_RATIO = 0.95

# 批量同步诊断只报告整数帧偏移，不自动移动或重命名帧。
SYNC_MAX_LAG_FRAMES = 5
SYNC_DIAGNOSTIC_MAX_DIM = 240

# ----------------------------------------------------------
# RGB/IR 同步视频抽帧
# ----------------------------------------------------------
# 输入目录结构：video_exporter/input/<group_name>/ir.<ext> + rgb.<ext>
# 输出目录结构：video_exporter/output/<group_name>/ir + rgb
VIDEO_EXPORT_ROOT = os.path.join(PROJECT_PATH, "video_exporter")
VIDEO_EXPORT_INPUT_PATH = os.path.join(VIDEO_EXPORT_ROOT, "input")
VIDEO_EXPORT_OUTPUT_PATH = os.path.join(VIDEO_EXPORT_ROOT, "output")

VIDEO_EXPORT_DEFAULT_COUNT = 750
VIDEO_EXPORT_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
VIDEO_EXPORT_FPS_TOLERANCE = 1e-3
VIDEO_EXPORT_PNG_COMPRESSION = 3
VIDEO_EXPORT_SHARPNESS_MAX_DIM = 480
VIDEO_EXPORT_PREVIEW_COUNT = 12

# ----------------------------------------------------------
# 可重复数据处理工作区
# ----------------------------------------------------------
# 用户把新视频投递到 processing_workspace/input；程序只读这些输入，
# 所有规范化视频、抽帧图片、报告和状态都写入独立目录。
DATASET_ROOT = _configured_path("RGB_IR_DATASET_ROOT", "dataset")
PROCESSING_WORKSPACE_PATH = os.path.join(DATASET_ROOT, "processing_workspace")
PROCESSING_INPUT_PATH = os.path.join(PROCESSING_WORKSPACE_PATH, "input")
PROCESSING_OUTPUT_PATH = os.path.join(PROCESSING_WORKSPACE_PATH, "output")
PROCESSING_SYSTEM_PATH = os.path.join(PROCESSING_WORKSPACE_PATH, "_system")

PROCESSING_TARGET_SIZE = (1920, 1080)  # (width, height)
PROCESSING_CALIBRATION_RGB_SOURCE_SIZE = (720, 1280)
PROCESSING_CALIBRATION_RGB_ROTATION = "counterclockwise_90"
PROCESSING_CALIBRATION_DEFAULT_COUNT = 4
PROCESSING_EXPERIMENT_DEFAULT_COUNT = VIDEO_EXPORT_DEFAULT_COUNT
PROCESSING_EXPERIMENT_CATEGORIES = ("health", "health+sick", "sick")
PROCESSING_FPS_TOLERANCE = VIDEO_EXPORT_FPS_TOLERANCE
PROCESSING_PNG_COMPRESSION = VIDEO_EXPORT_PNG_COMPRESSION
PROCESSING_SHARPNESS_MAX_DIM = VIDEO_EXPORT_SHARPNESS_MAX_DIM
PROCESSING_PREVIEW_COUNT = VIDEO_EXPORT_PREVIEW_COUNT
PROCESSING_H264_CRF = 18
PROCESSING_H264_PRESET = "medium"
PROCESSING_SCHEMA_VERSION = 1

# ----------------------------------------------------------
# 当前分组数据集的标定/配准输出
# ----------------------------------------------------------
DATASET_CALIBRATION_INPUT_PATH = os.path.join(PROCESSING_OUTPUT_PATH, "calibration")
DATASET_EXPERIMENT_INPUT_PATH = os.path.join(PROCESSING_OUTPUT_PATH, "experiment")
DATASET_CALIBRATED_OUTPUT_PATH = os.path.join(DATASET_ROOT, "calibrated_output")
DATASET_MODEL_OUTPUT_PATH = os.path.join(DATASET_CALIBRATED_OUTPUT_PATH, "_models")
DATASET_REGISTERED_OUTPUT_PATH = os.path.join(
    DATASET_CALIBRATED_OUTPUT_PATH, "experiment"
)
DATASET_BATCH_REPORT_PATH = os.path.join(
    DATASET_CALIBRATED_OUTPUT_PATH, "batch_report.yaml"
)
