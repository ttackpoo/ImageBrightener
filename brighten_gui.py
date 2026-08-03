from pathlib import Path
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# 지원 확장자
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in EXTS


def brighten_image(src_path: Path, dst_path: Path, brightness_percent: float):
    """
    Excel/PowerPoint 스타일 밝기:
    새값 = 원래값 + (255 - 원래값) * (brightness_percent / 100)
    예: 95면 거의 흰색에 가까워짐
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        has_alpha = img.mode in ("RGBA", "LA")

        if has_alpha:
            alpha = img.getchannel("A")
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        arr = np.array(img).astype(np.float32)

        # Office(Excel/PowerPoint) 방식
        arr = arr + (255.0 - arr) * (brightness_percent / 100.0)

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr, mode="RGB")

        if has_alpha:
            result = result.convert("RGBA")
            result.putalpha(alpha)

        result.save(dst_path)


class BrightenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("이미지 밝기 조절")
        self.root.geometry("640x420")
        self.root.resizable(False, False)

        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.brightness = tk.DoubleVar(value=95)   # 기본값 95%
        self.recursive = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="준비됨")

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=14, pady=14)

        ttk.Label(frm, text="입력 폴더").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.input_folder, width=58).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm, text="찾기", command=self.choose_input).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="출력 폴더").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.output_folder, width=58).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(frm, text="찾기", command=self.choose_output).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="밝기(%)").grid(row=2, column=0, sticky="w", **pad)
        brightness_frame = ttk.Frame(frm)
        brightness_frame.grid(row=2, column=1, sticky="we", **pad)

        self.scale = ttk.Scale(
            brightness_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.brightness,
            command=self.update_brightness_label,
        )
        self.scale.pack(side="left", fill="x", expand=True)

        self.brightness_label = ttk.Label(brightness_frame, text="95%", width=8)
        self.brightness_label.pack(side="left", padx=10)

        ttk.Checkbutton(
            frm,
            text="하위 폴더까지 처리",
            variable=self.recursive
        ).grid(row=3, column=1, sticky="w", **pad)

        self.start_btn = ttk.Button(frm, text="시작", command=self.start_processing)
        self.start_btn.grid(row=4, column=1, sticky="e", **pad)

        ttk.Label(frm, text="진행 상태").grid(row=5, column=0, sticky="nw", **pad)
        self.log = tk.Text(frm, height=11, width=72, state="disabled")
        self.log.grid(row=5, column=1, columnspan=2, sticky="nsew", **pad)

        status_bar = ttk.Label(frm, textvariable=self.status, relief="sunken", anchor="w")
        status_bar.grid(row=6, column=0, columnspan=3, sticky="we", padx=10, pady=(12, 0))

        frm.columnconfigure(1, weight=1)

    def choose_input(self):
        folder = filedialog.askdirectory(title="입력 폴더 선택")
        if folder:
            self.input_folder.set(folder)

    def choose_output(self):
        folder = filedialog.askdirectory(title="출력 폴더 선택")
        if folder:
            self.output_folder.set(folder)

    def update_brightness_label(self, _=None):
        self.brightness_label.config(text=f"{self.brightness.get():.0f}%")

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_busy(self, busy: bool):
        self.start_btn.config(state="disabled" if busy else "normal")

    def start_processing(self):
        input_dir = Path(self.input_folder.get().strip())
        output_dir = Path(self.output_folder.get().strip())
        brightness_percent = float(self.brightness.get())
        recursive = bool(self.recursive.get())

        if not input_dir.exists() or not input_dir.is_dir():
            messagebox.showerror("오류", "입력 폴더를 올바르게 선택하세요.")
            return

        if not self.output_folder.get().strip():
            messagebox.showerror("오류", "출력 폴더를 선택하세요.")
            return

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

        self.set_busy(True)
        self.status.set("처리 중...")
        self.write_log(f"입력 폴더: {input_dir}")
        self.write_log(f"출력 폴더: {output_dir}")
        self.write_log(f"밝기: {brightness_percent:.0f}%")
        self.write_log(f"하위 폴더 처리: {'예' if recursive else '아니오'}")
        self.write_log("")

        thread = threading.Thread(
            target=self.process_images,
            args=(input_dir, output_dir, brightness_percent, recursive),
            daemon=True
        )
        thread.start()

    def process_images(self, input_dir, output_dir, brightness_percent, recursive):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            if recursive:
                files = [p for p in input_dir.rglob("*") if p.is_file() and is_image_file(p)]
            else:
                files = [p for p in input_dir.iterdir() if p.is_file() and is_image_file(p)]

            total = len(files)
            self.root.after(0, self.write_log, f"처리할 이미지 수: {total}")

            if total == 0:
                self.root.after(0, self.status.set, "처리할 이미지가 없습니다.")
                self.root.after(0, self.set_busy, False)
                return

            for i, src in enumerate(files, start=1):
                try:
                    if recursive:
                        dst = output_dir / src.relative_to(input_dir)
                    else:
                        dst = output_dir / src.name

                    brighten_image(src, dst, brightness_percent)
                    self.root.after(0, self.write_log, f"[{i}/{total}] 완료: {src.name}")
                except Exception as e:
                    self.root.after(0, self.write_log, f"[{i}/{total}] 실패: {src.name} -> {e}")

            self.root.after(0, self.status.set, "완료")
            self.root.after(0, self.write_log, "")
            self.root.after(0, self.write_log, "끝")
        except Exception as e:
            self.root.after(0, messagebox.showerror, "오류", str(e))
            self.root.after(0, self.status.set, "오류 발생")
        finally:
            self.root.after(0, self.set_busy, False)


if __name__ == "__main__":
    root = tk.Tk()
    app = BrightenApp(root)
    root.mainloop()
