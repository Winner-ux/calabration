# 待配准图像

- 将可见光图像放入 `rgb/`。
- 将红外图像放入 `ir/`。
- RGB 与 IR 使用相同的文件名主干即可配对，扩展名可以不同。例如：
  `rgb/001.jpg` 与 `ir/001.png`。
- 支持的格式由 `config.py` 中的 `IMAGE_EXTENSIONS` 控制。
- 执行 `python apply_registration.py` 后，结果保存在 `output/registered/`。
