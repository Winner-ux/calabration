from __future__ import annotations

import os
import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable

from PIL import Image, ImageTk

from .discovery import (
    discover_numeric_pairs,
    normalize_clip,
    normalize_day,
    normalize_experiment,
    validate_experiment_pair,
)
from .importer import import_confirmed_pairs
from .manifest import find_ready_tasks, validate_task
from .models import CATEGORIES, PairSpec, TaskType
from .paths import ProjectPaths
from .processors import run_calibration_batch, run_experiment_batch
from .video import VideoError, normalize_frame, probe_video, read_frame


def _metadata_text(pair: PairSpec) -> str:
    if not pair.ir_metadata or not pair.rgb_metadata:
        return "-"
    ir, rgb = pair.ir_metadata, pair.rgb_metadata
    return (
        f"IR {ir.width}×{ir.height}@{ir.fps:.3f} {ir.frame_count}f/{ir.duration_seconds:.1f}s | "
        f"RGB {rgb.width}×{rgb.height}@{rgb.fps:.3f} {rgb.frame_count}f/{rgb.duration_seconds:.1f}s"
    )


class ProcessingApp:
    def __init__(self, root: tk.Tk, job_type: TaskType):
        self.root = root
        self.job_type = job_type
        self.paths = ProjectPaths.discover()
        self.paths.ensure_system_dirs()
        self.candidates: list[PairSpec] = []
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False

        is_calibration = job_type == "calibration"
        self.root.title(
            "RGB/IR 标定图像处理" if is_calibration else "RGB/IR 实验图像处理"
        )
        self.root.geometry("1320x760")
        self.root.minsize(1060, 650)
        self.source_var = tk.StringVar(
            value=str(
                self.paths.calibration_source
                if is_calibration
                else self.paths.experiment_source
            )
        )
        self.count_var = tk.IntVar(value=4 if is_calibration else 750)
        self.force_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_messages)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        source = ttk.LabelFrame(outer, text="1. 视频源目录", padding=8)
        source.pack(fill="x")
        ttk.Entry(source, textvariable=self.source_var).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ttk.Button(source, text="选择目录", command=self._choose_source).pack(
            side="left", padx=4
        )
        ttk.Button(source, text="扫描候选配对", command=self.scan).pack(
            side="left", padx=4
        )

        actions = ttk.Frame(outer, padding=(0, 8))
        actions.pack(fill="x")
        ttk.Button(actions, text="预览所选 RGB/IR", command=self.preview_selected).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="确认所选配对", command=self.confirm_selected).pack(
            side="left", padx=6
        )
        ttk.Button(actions, text="手工添加配对", command=self.manual_add).pack(
            side="left", padx=6
        )
        ttk.Button(
            actions, text="导入全部已确认项", command=self.import_confirmed
        ).pack(side="left", padx=6)

        table_frame = ttk.LabelFrame(
            outer, text="2. 配对预览（自动发现仅是候选，必须确认后才能导入）", padding=6
        )
        table_frame.pack(fill="both", expand=True)
        columns = (
            "day",
            "experiment",
            "category",
            "clip",
            "ir",
            "rgb",
            "metadata",
            "status",
        )
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="extended"
        )
        headers = {
            "day": "日期",
            "experiment": "实验",
            "category": "类别",
            "clip": "Clip",
            "ir": "IR 文件",
            "rgb": "RGB 文件",
            "metadata": "尺寸 / FPS",
            "status": "状态",
        }
        widths = {
            "day": 70,
            "experiment": 100,
            "category": 95,
            "clip": 60,
            "ir": 170,
            "rgb": 170,
            "metadata": 310,
            "status": 100,
        }
        for column in columns:
            self.tree.heading(column, text=headers[column])
            self.tree.column(column, width=widths[column], minwidth=50)
        y_scroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.tree.yview
        )
        x_scroll = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.tree.xview
        )
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        run = ttk.LabelFrame(outer, text="3. 工作区校验与批量处理", padding=8)
        run.pack(fill="x", pady=(8, 0))
        ttk.Label(run, text="每个 clip 抽取：").pack(side="left")
        ttk.Spinbox(run, from_=1, to=100000, textvariable=self.count_var, width=8).pack(
            side="left", padx=(0, 12)
        )
        ttk.Checkbutton(
            run, text="重新处理（会先备份旧结果）", variable=self.force_var
        ).pack(side="left", padx=6)
        ttk.Button(run, text="校验 input", command=self.validate_workspace).pack(
            side="left", padx=6
        )
        ttk.Button(run, text="开始批量处理", command=self.process).pack(
            side="left", padx=6
        )
        ttk.Button(run, text="取消", command=self.cancel).pack(side="left", padx=6)
        ttk.Button(run, text="打开 output", command=self.open_output).pack(
            side="right", padx=6
        )

        log_frame = ttk.LabelFrame(outer, text="运行信息", padding=6)
        log_frame.pack(fill="both", pady=(8, 0))
        self.log = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        ttk.Label(outer, textvariable=self.status_var, anchor="w").pack(
            fill="x", pady=(5, 0)
        )

    def _choose_source(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=self.source_var.get(), title="选择视频源目录"
        )
        if selected:
            self.source_var.set(selected)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, value: bool, status: str = "") -> None:
        self.busy = value
        self.status_var.set(status or ("处理中……" if value else "就绪"))

    def _background(self, operation: Callable[[], object], done_event: str) -> None:
        if self.busy:
            messagebox.showinfo("任务进行中", "请等待当前操作结束。")
            return
        self._set_busy(True)

        def worker() -> None:
            try:
                result = operation()
                self.messages.put((done_event, result))
            except Exception as exc:
                self.messages.put(("error", (exc, traceback.format_exc())))
            finally:
                self.messages.put(("idle", None))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_messages(self) -> None:
        try:
            while True:
                event, payload = self.messages.get_nowait()
                if event == "log":
                    self._append_log(str(payload))
                    self.status_var.set(str(payload))
                elif event == "scan_done":
                    pairs, issues = payload  # type: ignore[misc]
                    self._show_candidates(pairs)
                    self._append_log(f"扫描完成：发现 {len(pairs)} 个同编号候选配对。")
                    for issue in issues:
                        self._append_log("需人工处理：" + issue)
                elif event == "import_done":
                    for message in payload:  # type: ignore[union-attr]
                        self._append_log(message)
                    messagebox.showinfo("导入完成", "全部已确认项已复制并校验。")
                elif event == "process_done":
                    results, report = payload  # type: ignore[misc]
                    for result in results:
                        self._append_log(
                            f"[{result.status}] {result.task}：{result.message}"
                        )
                    self._append_log(f"运行报告：{report}")
                    messagebox.showinfo("处理结束", "批量处理已结束，请查看运行信息。")
                elif event == "validate_done":
                    for task, errors in payload:  # type: ignore[union-attr]
                        if errors:
                            self._append_log(f"[失败] {task}：" + "；".join(errors))
                        else:
                            self._append_log(f"[通过] {task}")
                elif event == "error":
                    exc, detail = payload  # type: ignore[misc]
                    self._append_log(detail)
                    messagebox.showerror("操作失败", str(exc))
                elif event == "idle":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_messages)

    def scan(self) -> None:
        source = Path(self.source_var.get()).expanduser()
        self._append_log(f"扫描：{source}")
        self._background(
            lambda: discover_numeric_pairs(source, self.job_type), "scan_done"
        )

    def _show_candidates(self, pairs: list[PairSpec]) -> None:
        self.candidates = pairs
        self.tree.delete(*self.tree.get_children())
        for index, pair in enumerate(pairs):
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    pair.day,
                    pair.experiment,
                    pair.category or "-",
                    pair.clip_id,
                    pair.ir_path.name,
                    pair.rgb_path.name,
                    _metadata_text(pair),
                    pair.status,
                ),
            )

    def _selected_pairs(self) -> list[PairSpec]:
        return [self.candidates[int(item)] for item in self.tree.selection()]

    def confirm_selected(self) -> None:
        selected = self._selected_pairs()
        if not selected:
            messagebox.showinfo("未选择", "请先在表格中选择一项或多项。")
            return
        descriptions = "\n".join(
            f"{p.day}/{p.experiment}/{p.category or '-'} clip{p.clip_id}: {p.ir_path.name} ↔ {p.rgb_path.name}"
            for p in selected
        )
        if not messagebox.askyesno(
            "确认对应关系", "请确认以下类别及 RGB/IR 对应关系正确：\n\n" + descriptions
        ):
            return
        for pair in selected:
            if self.job_type == "experiment":
                errors = validate_experiment_pair(pair)
                if errors:
                    messagebox.showerror(
                        "成片不合格", f"{pair.unit_name}\n" + "\n".join(errors)
                    )
                    continue
            pair.confirmed = True
            pair.status = "已人工确认"
            self.tree.set(str(self.candidates.index(pair)), "status", pair.status)

    def manual_add(self) -> None:
        ir = filedialog.askopenfilename(
            title="选择 IR 视频",
            filetypes=[("视频文件", "*.mp4 *.mov *.avi *.mkv *.m4v")],
        )
        if not ir:
            return
        rgb = filedialog.askopenfilename(
            title="选择 RGB 视频",
            filetypes=[("视频文件", "*.mp4 *.mov *.avi *.mkv *.m4v")],
        )
        if not rgb:
            return
        day = simpledialog.askstring("日期", "输入日期，例如 day5：", parent=self.root)
        experiment = simpledialog.askstring(
            "实验号", "输入实验号，例如 1 或 experiment1：", parent=self.root
        )
        clip = simpledialog.askstring(
            "Clip", "输入对应编号，例如 01：", parent=self.root
        )
        if not day or not experiment or not clip:
            return
        category: str | None = None
        if self.job_type == "experiment":
            category = simpledialog.askstring(
                "实验类别", "输入 health、health+sick 或 sick：", parent=self.root
            )
            if category not in CATEGORIES:
                messagebox.showerror(
                    "类别无效", "类别只能是 health、health+sick 或 sick。"
                )
                return
        try:
            pair = PairSpec(
                job_type=self.job_type,
                day=normalize_day(day),
                experiment=normalize_experiment(experiment),
                clip_id=normalize_clip(clip),
                category=category,
                ir_path=Path(ir),
                rgb_path=Path(rgb),
                status="待人工确认",
            )
            pair.ir_metadata, pair.rgb_metadata = (
                probe_video(pair.ir_path),
                probe_video(pair.rgb_path),
            )
        except (ValueError, VideoError) as exc:
            messagebox.showerror("视频无效", str(exc))
            return
        self.candidates.append(pair)
        index = len(self.candidates) - 1
        self.tree.insert(
            "",
            "end",
            iid=str(index),
            values=(
                pair.day,
                pair.experiment,
                pair.category or "-",
                pair.clip_id,
                pair.ir_path.name,
                pair.rgb_path.name,
                _metadata_text(pair),
                pair.status,
            ),
        )
        self.tree.selection_set(str(index))
        self.preview_selected()

    def preview_selected(self) -> None:
        selected = self._selected_pairs()
        if len(selected) != 1:
            messagebox.showinfo("请选择一项", "预览时请只选择一个 RGB/IR 候选。")
            return
        pair = selected[0]
        try:
            ir_meta = pair.ir_metadata or probe_video(pair.ir_path)
            rgb_meta = pair.rgb_metadata or probe_video(pair.rgb_path)
            frames = [
                ("IR 首帧", read_frame(pair.ir_path, 0)),
                ("RGB 首帧", read_frame(pair.rgb_path, 0)),
                ("IR 中间帧", read_frame(pair.ir_path, ir_meta.frame_count // 2)),
                ("RGB 中间帧", read_frame(pair.rgb_path, rgb_meta.frame_count // 2)),
            ]
        except VideoError as exc:
            messagebox.showerror("预览失败", str(exc))
            return
        window = tk.Toplevel(self.root)
        window.title(
            f"配对预览 - {pair.day}/{pair.experiment}/{pair.category or '-'} clip{pair.clip_id}"
        )
        references: list[ImageTk.PhotoImage] = []
        for index, (label, frame) in enumerate(frames):
            frame = normalize_frame(frame, (480, 270), rotate_portrait=True)
            image = Image.fromarray(frame[:, :, ::-1])
            photo = ImageTk.PhotoImage(image)
            references.append(photo)
            ttk.Label(window, text=label).grid(
                row=(index // 2) * 2, column=index % 2, pady=(8, 2)
            )
            ttk.Label(window, image=photo).grid(
                row=(index // 2) * 2 + 1, column=index % 2, padx=8, pady=(0, 8)
            )
        window._image_references = references  # type: ignore[attr-defined]

    def import_confirmed(self) -> None:
        confirmed = [pair for pair in self.candidates if pair.confirmed]
        if not confirmed:
            messagebox.showinfo("没有已确认项", "请先预览并确认至少一个配对。")
            return
        if not messagebox.askyesno(
            "确认复制",
            f"将复制 {len(confirmed)} 对视频到标准 input，原视频保持不变。继续吗？",
        ):
            return
        self._background(
            lambda: import_confirmed_pairs(self.paths, confirmed), "import_done"
        )

    def validate_workspace(self) -> None:
        tasks = find_ready_tasks(self.paths.input_root, self.job_type)
        if not tasks:
            self._append_log("没有找到 READY 任务。")
            return
        self._background(
            lambda: [(task, validate_task(task, self.job_type)) for task in tasks],
            "validate_done",
        )

    def process(self) -> None:
        try:
            count = int(self.count_var.get())
            if count <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("参数错误", "抽帧数量必须是大于 0 的整数。")
            return
        force = bool(self.force_var.get())
        if force and not messagebox.askyesno(
            "确认重新处理",
            "重新处理会在新结果通过校验后，将旧结果移入 _system/backups。确认继续吗？",
        ):
            return
        self.cancel_event.clear()
        runner = (
            run_calibration_batch
            if self.job_type == "calibration"
            else run_experiment_batch
        )
        self._background(
            lambda: runner(
                self.paths,
                count=count,
                force=force,
                progress=lambda text: self.messages.put(("log", text)),
                cancel_event=self.cancel_event,
            ),
            "process_done",
        )

    def cancel(self) -> None:
        if self.busy:
            self.cancel_event.set()
            self._append_log(
                "已请求取消；程序将在安全检查点停止，当前正式输出不会留下半成品。"
            )

    def open_output(self) -> None:
        output = self.paths.output_root / self.job_type
        output.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(output)  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("输出目录", str(output))

    def _on_close(self) -> None:
        if self.busy:
            if not messagebox.askyesno(
                "任务进行中", "关闭窗口可能中断当前操作。确认关闭吗？"
            ):
                return
            self.cancel_event.set()
        self.root.destroy()


def launch(job_type: TaskType) -> None:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        ProcessingApp(root, job_type)
        root.mainloop()
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("程序启动失败", str(exc))
        root.destroy()
