from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image

# 지원 확장자
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in EXTS


def apply_exposure_gamma_clahe(
    img: Image.Image,
    exposure_ev: float,
    gamma_value: float,
    clahe_clip: float,
) -> Image.Image:
    """
    1) Exposure: 밝기 전체를 선형 증폭
       -1 EV = 0.5배, +1 EV = 2배

    2) Gamma: 중간톤 조정
       gamma < 1.0 -> 밝아짐
       gamma > 1.0 -> 어두워짐

    3) CLAHE: 국부 대비 강화
       작은 명암 차이를 더 잘 보이게 함
    """
    has_alpha = "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info)

    if has_alpha:
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        rgb_img = rgba.convert("RGB")
    else:
        alpha = None
        rgb_img = img.convert("RGB")

    # RGB -> BGR -> LAB
    rgb = np.array(rgb_img, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    # 1) Exposure (EV)
    l = l.astype(np.float32)
    exposure_factor = 2.0 ** float(exposure_ev)
    l = l * exposure_factor
    l = np.clip(l, 0, 255)

    # 2) Gamma
    # normalized 0..1 -> gamma correction
    # gamma < 1 => brighter, gamma > 1 => darker
    gamma_value = max(0.05, float(gamma_value))
    l_norm = l / 255.0
    l_norm = np.power(l_norm, gamma_value)
    l = np.clip(l_norm * 255.0, 0, 255).astype(np.uint8)

    # 3) CLAHE
    clahe_clip = max(1.0, float(clahe_clip))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab2 = cv2.merge((l, a, b))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

    result = Image.fromarray(rgb2, mode="RGB")

    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(alpha)

    return result


def process_single_image(
    src_path: Path,
    dst_path: Path,
    exposure_ev: float,
    gamma_value: float,
    clahe_clip: float,
):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        result = apply_exposure_gamma_clahe(img, exposure_ev, gamma_value, clahe_clip)
        result.save(dst_path)


class BrightenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("이미지 노출/감마/CLAHE 조절")
        self.root.geometry("740x520")
        self.root.resizable(False, False)

        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()

        # 기본값: HTE/어두운 이미지 기준
        self.exposure_ev = tk.DoubleVar(value=0.8)   # +0.8EV
        self.gamma_value = tk.DoubleVar(value=0.9)    # 조금 밝게
        self.clahe_clip = tk.DoubleVar(value=2.5)     # 국부 대비 강화

        self.recursive = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="준비됨")

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=14, pady=14)

        ttk.Label(frm, text="입력 폴더").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.input_folder, width=62).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm, text="찾기", command=self.choose_input).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="출력 폴더").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.output_folder, width=62).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(frm, text="찾기", command=self.choose_output).grid(row=1, column=2, **pad)

        # Exposure
        ttk.Label(frm, text="노출(EV)").grid(row=2, column=0, sticky="w", **pad)
        exposure_frame = ttk.Frame(frm)
        exposure_frame.grid(row=2, column=1, sticky="we", **pad)

        self.exposure_scale = ttk.Scale(
            exposure_frame,
            from_=-2.0,
            to=3.0,
            orient="horizontal",
            variable=self.exposure_ev,
            command=self.update_labels,
        )
        self.exposure_scale.pack(side="left", fill="x", expand=True)

        self.exposure_label = ttk.Label(exposure_frame, text="+0.8 EV", width=10)
        self.exposure_label.pack(side="left", padx=10)

        # Gamma
        ttk.Label(frm, text="감마").grid(row=3, column=0, sticky="w", **pad)
        gamma_frame = ttk.Frame(frm)
        gamma_frame.grid(row=3, column=1, sticky="we", **pad)

        self.gamma_scale = ttk.Scale(
            gamma_frame,
            from_=0.5,
            to=1.8,
            orient="horizontal",
            variable=self.gamma_value,
            command=self.update_labels,
        )
        self.gamma_scale.pack(side="left", fill="x", expand=True)

        self.gamma_label = ttk.Label(gamma_frame, text="0.90", width=10)
        self.gamma_label.pack(side="left", padx=10)

        # CLAHE
        ttk.Label(frm, text="CLAHE").grid(row=4, column=0, sticky="w", **pad)
        clahe_frame = ttk.Frame(frm)
        clahe_frame.grid(row=4, column=1, sticky="we", **pad)

        self.clahe_scale = ttk.Scale(
            clahe_frame,
            from_=1.0,
            to=5.0,
            orient="horizontal",
            variable=self.clahe_clip,
            command=self.update_labels,
        )
        self.clahe_scale.pack(side="left", fill="x", expand=True)

        self.clahe_label = ttk.Label(clahe_frame, text="2.5", width=10)
        self.clahe_label.pack(side="left", padx=10)

        ttk.Checkbutton(
            frm,
            text="하위 폴더까지 처리",
            variable=self.recursive
        ).grid(row=5, column=1, sticky="w", **pad)

        self.start_btn = ttk.Button(frm, text="시작", command=self.start_processing)
        self.start_btn.grid(row=6, column=1, sticky="e", **pad)

        ttk.Label(frm, text="진행 상태").grid(row=7, column=0, sticky="nw", **pad)
        self.log = tk.Text(frm, height=12, width=82, state="disabled")
        self.log.grid(row=7, column=1, columnspan=2, sticky="nsew", **pad)

        status_bar = ttk.Label(frm, textvariable=self.status, relief="sunken", anchor="w")
        status_bar.grid(row=8, column=0, columnspan=3, sticky="we", padx=10, pady=(12, 0))

        frm.columnconfigure(1, weight=1)

        self.update_labels()

    def choose_input(self):
        folder = filedialog.askdirectory(title="입력 폴더 선택")
        if folder:
            self.input_folder.set(folder)

    def choose_output(self):
        folder = filedialog.askdirectory(title="출력 폴더 선택")
        if folder:
            self.output_folder.set(folder)

    def update_labels(self, _=None):
        ev = self.exposure_ev.get()
        if ev >= 0:
            self.exposure_label.config(text=f"+{ev:.1f} EV")
        else:
            self.exposure_label.config(text=f"{ev:.1f} EV")
        self.gamma_label.config(text=f"{self.gamma_value.get():.2f}")
        self.clahe_label.config(text=f"{self.clahe_clip.get():.1f}")

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

        exposure_ev = float(self.exposure_ev.get())
        gamma_value = float(self.gamma_value.get())
        clahe_clip = float(self.clahe_clip.get())
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
        self.write_log(f"노출: {exposure_ev:+.1f} EV")
        self.write_log(f"감마: {gamma_value:.2f}")
        self.write_log(f"CLAHE: {clahe_clip:.1f}")
        self.write_log(f"하위 폴더 처리: {'예' if recursive else '아니오'}")
        self.write_log("")

        thread = threading.Thread(
            target=self.process_images,
            args=(input_dir, output_dir, exposure_ev, gamma_value, clahe_clip, recursive),
            daemon=True
        )
        thread.start()

    def process_images(self, input_dir, output_dir, exposure_ev, gamma_value, clahe_clip, recursive):
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

                    process_single_image(src, dst, exposure_ev, gamma_value, clahe_clip)
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
