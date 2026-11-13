"""Native, local-only Kulu desktop assistant."""

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from local_services import analyze_image, run_local_command


APP_DIR = Path(__file__).resolve().parent


class CommandConfirmation(tk.Toplevel):
    def __init__(self, parent, command, cwd):
        super().__init__(parent)
        self.result = False
        self.title("确认执行本地命令")
        self.transient(parent)
        self.resizable(True, True)
        self.geometry("620x360")

        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text="请核对完整命令及工作目录。命令可能修改文件或访问网络；"
            "它将以当前 Windows 用户权限执行，不会提权：",
            wraplength=570,
        ).pack(anchor="w", pady=(0, 8))
        command_frame = ttk.Frame(body)
        command_frame.pack(fill="both", expand=True)
        command_box = tk.Text(command_frame, height=8, wrap="word")
        command_box.pack(side="left", fill="both", expand=True)
        command_scroll = ttk.Scrollbar(
            command_frame, orient="vertical", command=command_box.yview
        )
        command_scroll.pack(side="right", fill="y")
        command_box.configure(yscrollcommand=command_scroll.set)
        command_box.insert("1.0", command)
        command_box.configure(state="disabled")
        ttk.Label(body, text=f"工作目录：{cwd}", wraplength=570).pack(
            anchor="w", pady=8
        )

        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="确认执行", command=self._confirm).pack(
            side="right", padx=(0, 8)
        )
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()
        self.focus_set()

    def _confirm(self):
        self.result = True
        self.destroy()


class LocalAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kulu 本地助手")
        self.root.geometry("940x700")
        self.root.minsize(720, 540)
        self.events = queue.Queue()
        self.command_running = False
        self.image_running = False
        self.image_path = None
        self.preview_image = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self.root.after(100, self._drain_events)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Kulu 本地助手", font=("Segoe UI", 20, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            outer,
            text="命令和图片仅在本机处理；每次执行命令前都会显示完整命令并要求确认。",
        ).pack(anchor="w", pady=(4, 16))

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.command_tab = ttk.Frame(self.tabs, padding=14)
        self.vision_tab = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(self.command_tab, text="本地命令")
        self.tabs.add(self.vision_tab, text="图像识别")
        self._build_command_tab()
        self._build_vision_tab()

    def _build_command_tab(self):
        tab = self.command_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)

        ttk.Label(tab, text="工作目录").grid(row=0, column=0, sticky="w")
        location = ttk.Frame(tab)
        location.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        location.columnconfigure(0, weight=1)
        self.cwd_var = tk.StringVar(value=str(APP_DIR))
        ttk.Entry(location, textvariable=self.cwd_var).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(location, text="选择目录…", command=self._choose_cwd).grid(
            row=0, column=1, padx=(8, 0)
        )

        ttk.Label(tab, text="命令").grid(row=2, column=0, sticky="w")
        self.command_text = tk.Text(tab, height=5, wrap="none", undo=True)
        self.command_text.grid(row=3, column=0, sticky="nsew", pady=(4, 10))
        self.command_text.insert(
            "1.0", "python --version"
        )

        controls = ttk.Frame(tab)
        controls.grid(row=4, column=0, sticky="ew")
        self.run_button = ttk.Button(
            controls, text="执行…", command=self._confirm_and_run
        )
        self.run_button.pack(side="left")
        self.command_status = tk.StringVar(value="命令不会提权，也不会通过网络接口执行。")
        ttk.Label(controls, textvariable=self.command_status).pack(
            side="left", padx=12
        )

        ttk.Label(tab, text="输出").grid(row=5, column=0, sticky="w", pady=(14, 4))
        self.command_output = tk.Text(tab, height=10, wrap="word", state="disabled")
        self.command_output.grid(row=6, column=0, sticky="nsew")
        tab.rowconfigure(6, weight=1)

    def _build_vision_tab(self):
        tab = self.vision_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)

        picker = ttk.Frame(tab)
        picker.grid(row=0, column=0, sticky="ew")
        picker.columnconfigure(0, weight=1)
        self.image_var = tk.StringVar(value="尚未选择图片")
        ttk.Label(picker, textvariable=self.image_var).grid(
            row=0, column=0, sticky="ew"
        )
        self.image_button = ttk.Button(
            picker, text="选择图片…", command=self._choose_image
        )
        self.image_button.grid(row=0, column=1, padx=(8, 0))

        options = ttk.Frame(tab)
        options.grid(row=1, column=0, sticky="w", pady=10)
        self.ocr_var = tk.BooleanVar(value=True)
        self.objects_var = tk.BooleanVar(value=True)
        self.ocr_check = ttk.Checkbutton(
            options, text="文字识别（中/英文）", variable=self.ocr_var
        )
        self.ocr_check.pack(side="left")
        self.objects_check = ttk.Checkbutton(
            options, text="物体识别（80 类）", variable=self.objects_var
        )
        self.objects_check.pack(side="left", padx=(14, 0))
        self.analyze_button = ttk.Button(
            options, text="开始本地识别", command=self._analyze_selected_image
        )
        self.analyze_button.pack(side="left", padx=(18, 0))
        self.image_status = tk.StringVar(value="图片不会离开本机。首次使用会下载本地模型。")
        ttk.Label(tab, textvariable=self.image_status, wraplength=760).grid(
            row=2, column=0, sticky="w", pady=(0, 8)
        )

        self.preview = ttk.Label(tab, text="识别后的标注预览会显示在这里", anchor="center")
        self.preview.grid(row=3, column=0, sticky="nsew", pady=(4, 10))
        self.result_text = tk.Text(tab, height=9, wrap="word", state="disabled")
        self.result_text.grid(row=4, column=0, sticky="nsew")

    def _choose_cwd(self):
        selected = filedialog.askdirectory(
            initialdir=self.cwd_var.get() or str(APP_DIR),
            title="选择命令工作目录",
        )
        if selected:
            self.cwd_var.set(selected)

    def _confirm_and_run(self):
        command = self.command_text.get("1.0", "end-1c")
        cwd = self.cwd_var.get().strip()
        if not command.strip():
            messagebox.showwarning("缺少命令", "请先输入要执行的命令。", parent=self.root)
            return
        if not cwd or not Path(cwd).is_dir():
            messagebox.showerror("工作目录无效", "请选择一个存在的本地目录。", parent=self.root)
            return

        dialog = CommandConfirmation(self.root, command, cwd)
        self.root.wait_window(dialog)
        if not dialog.result:
            return

        self.command_running = True
        self.run_button.configure(state="disabled")
        self.command_status.set("命令正在运行…")
        self._set_text(self.command_output, "")

        def worker():
            try:
                return_code = run_local_command(
                    command,
                    cwd,
                    on_output=lambda line: self.events.put(("command-output", line)),
                )
                self.events.put(("command-done", return_code))
            except Exception as exc:
                self.events.put(("command-error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _choose_image(self):
        selected = filedialog.askopenfilename(
            title="选择本地图片",
            filetypes=[
                ("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not selected:
            return
        self.image_path = selected
        self.image_var.set(selected)
        self.image_status.set("已选择本地图片。选择识别项目后开始分析。")
        self.preview.configure(image="", text="就绪后会显示标注预览")
        self.preview_image = None
        self._set_text(self.result_text, "")

    def _analyze_selected_image(self):
        if not self.image_path:
            messagebox.showinfo("选择图片", "请先选择一张本地图片。", parent=self.root)
            return
        if not self.ocr_var.get() and not self.objects_var.get():
            messagebox.showinfo("选择识别项目", "至少选择一种图像识别方式。", parent=self.root)
            return

        image_path = self.image_path
        use_ocr = self.ocr_var.get()
        use_objects = self.objects_var.get()
        self.image_running = True
        self.analyze_button.configure(state="disabled")
        self.image_button.configure(state="disabled")
        self.ocr_check.configure(state="disabled")
        self.objects_check.configure(state="disabled")
        self.image_status.set("正在本机分析图片；首次识别可能需要下载模型…")
        self._set_text(self.result_text, "")

        def worker():
            try:
                result = analyze_image(image_path, use_ocr, use_objects)
                self.events.put(("image-done", result))
            except Exception as exc:
                self.events.put(("image-error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_events(self):
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "command-output":
                    self._append_text(self.command_output, payload)
                elif event == "command-done":
                    self.command_running = False
                    self.run_button.configure(state="normal")
                    self.command_status.set(f"命令结束，退出代码：{payload}")
                elif event == "command-error":
                    self.command_running = False
                    self.run_button.configure(state="normal")
                    self.command_status.set("命令启动失败")
                    self._append_text(self.command_output, f"\n错误：{payload}\n")
                elif event == "image-done":
                    self.image_running = False
                    self.analyze_button.configure(state="normal")
                    self._set_image_controls_enabled()
                    self._show_analysis(payload)
                elif event == "image-error":
                    self.image_running = False
                    self.analyze_button.configure(state="normal")
                    self._set_image_controls_enabled()
                    self.image_status.set("识别失败")
                    self._set_text(self.result_text, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _show_analysis(self, result):
        from PIL import Image, ImageTk

        preview = Image.fromarray(result["preview"])
        preview.thumbnail((760, 280))
        self.preview_image = ImageTk.PhotoImage(preview)
        self.preview.configure(image=self.preview_image, text="")

        lines = []
        if result["ocr"]:
            lines.append("识别到的文字：")
            lines.extend(
                f"  {item['text']}  ({item['score']:.0%})" for item in result["ocr"]
            )
        elif result["ocr_requested"]:
            lines.append("未识别到文字。")
        if result["objects"]:
            lines.append("\n识别到的物体：")
            lines.extend(
                f"  {item['label']}  ({item['score']:.0%})"
                for item in result["objects"]
            )
        elif result["objects_requested"]:
            lines.append("\n未识别到物体。")
        if result["errors"]:
            lines.append("\n识别提示：")
            lines.extend(f"  {error}" for error in result["errors"])
        self._set_text(self.result_text, "\n".join(lines))
        if result["errors"]:
            self.image_status.set("图片已在本机处理；部分识别步骤未完成。")
        else:
            self.image_status.set("图片已在本机完成识别；结果没有上传。")

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    @staticmethod
    def _append_text(widget, text):
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _set_image_controls_enabled(self):
        for widget in (self.image_button, self.ocr_check, self.objects_check):
            widget.configure(state="normal")

    def _on_close(self):
        if self.command_running or self.image_running:
            messagebox.showwarning(
                "本地任务仍在运行",
                "等待当前任务完成后再关闭，避免遗留后台进程或中断图片处理。",
                parent=self.root,
            )
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    LocalAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
