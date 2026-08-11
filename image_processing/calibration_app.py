"""PyCharm entry point for calibration-video processing only."""

import tkinter as tk
from tkinter import messagebox


def main() -> None:
    try:
        from processing_core.desktop import launch

        launch("calibration")
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "标定程序启动失败",
            f"请在 PyCharm 中选择项目的 Anaconda Python 解释器，并安装 requirements.txt。\n\n{exc}",
        )
        root.destroy()


if __name__ == "__main__":
    main()
