#!/usr/bin/env python3
"""Interactive two-image point picker for track-map/BEV homography alignment."""

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from layout_rendering import (
    blend_layout_lines,
    extract_layout_line_mask,
    warp_layout_line_mask,
)


class ZoomPointCanvas(ttk.Frame):
    def __init__(self, master, title, image_bgr, count, marker_radius, on_change):
        super().__init__(master)
        self.title = title
        self.image_bgr = image_bgr
        self.image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        self.image_w, self.image_h = self.image.size
        self.count = count
        self.marker_radius = marker_radius
        self.on_change = on_change
        self.points = []
        self.fit_scale = 1.0
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.drag_origin = None
        self.drag_offset = None
        self.drag_point = None
        self.photo = None
        self.initialized = False
        self.render_job = None

        header = ttk.Frame(self)
        header.pack(fill="x", padx=4, pady=(2, 4))
        ttk.Label(header, text=title, font=("Sans", 12, "bold")).pack(side="left")
        self.counter = ttk.Label(header, text=f"0/{count}")
        self.counter.pack(side="left", padx=10)
        ttk.Button(header, text="−", width=3, command=lambda: self.zoom_by(1 / 1.25)).pack(side="right")
        ttk.Button(header, text="+", width=3, command=lambda: self.zoom_by(1.25)).pack(side="right", padx=3)
        ttk.Button(header, text="화면 맞춤", command=self.fit).pack(side="right", padx=3)
        ttk.Button(header, text="전체 초기화", command=self.reset).pack(side="right", padx=3)
        ttk.Button(header, text="최근 점 취소", command=self.undo).pack(side="right", padx=3)

        self.canvas = tk.Canvas(self, background="#202020", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Button-1>", self._left_press)
        self.canvas.bind("<B1-Motion>", self._left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._left_release)
        self.canvas.bind("<ButtonPress-2>", self._pan_press)
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonPress-3>", self._pan_press)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._wheel_at(event.x, event.y, 1.15))
        self.canvas.bind("<Button-5>", lambda event: self._wheel_at(event.x, event.y, 1 / 1.15))

    @property
    def scale(self):
        return self.fit_scale * self.zoom

    def _on_configure(self, _event=None):
        if not self.initialized and self.canvas.winfo_width() > 20:
            self.fit()
            self.initialized = True
        else:
            self.schedule_render()

    def fit(self):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.fit_scale = min(cw / self.image_w, ch / self.image_h)
        self.zoom = 1.0
        self.offset_x = (cw - self.image_w * self.scale) / 2.0
        self.offset_y = (ch - self.image_h * self.scale) / 2.0
        self.render()

    def zoom_by(self, factor):
        self._wheel_at(
            self.canvas.winfo_width() / 2.0,
            self.canvas.winfo_height() / 2.0,
            factor,
        )

    def _mousewheel(self, event):
        self._wheel_at(event.x, event.y, 1.15 if event.delta > 0 else 1 / 1.15)

    def _wheel_at(self, x, y, factor):
        old_scale = self.scale
        image_x = (x - self.offset_x) / old_scale
        image_y = (y - self.offset_y) / old_scale
        self.zoom = float(np.clip(self.zoom * factor, 0.35, 8.0))
        new_scale = self.scale
        self.offset_x = x - image_x * new_scale
        self.offset_y = y - image_y * new_scale
        self._clamp_offsets()
        self.render()

    def _clamp_offsets(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        sw = self.image_w * self.scale
        sh = self.image_h * self.scale
        if sw <= cw:
            self.offset_x = (cw - sw) / 2.0
        else:
            self.offset_x = min(20.0, max(cw - sw - 20.0, self.offset_x))
        if sh <= ch:
            self.offset_y = (ch - sh) / 2.0
        else:
            self.offset_y = min(20.0, max(ch - sh - 20.0, self.offset_y))

    def _pan_press(self, event):
        self.drag_origin = (event.x, event.y)
        self.drag_offset = (self.offset_x, self.offset_y)

    def _pan_drag(self, event):
        if self.drag_origin is None:
            return
        self.offset_x = self.drag_offset[0] + event.x - self.drag_origin[0]
        self.offset_y = self.drag_offset[1] + event.y - self.drag_origin[1]
        self._clamp_offsets()
        self.render()

    def _canvas_to_image(self, x, y):
        return (
            (x - self.offset_x) / self.scale,
            (y - self.offset_y) / self.scale,
        )

    def _nearest_point(self, canvas_x, canvas_y, limit=12.0):
        if not self.points:
            return None
        distances = [
            np.hypot(
                self.offset_x + px * self.scale - canvas_x,
                self.offset_y + py * self.scale - canvas_y,
            )
            for px, py in self.points
        ]
        index = int(np.argmin(distances))
        return index if distances[index] <= limit else None

    def _left_press(self, event):
        nearest = self._nearest_point(event.x, event.y)
        if nearest is not None:
            self.drag_point = nearest
            return
        x, y = self._canvas_to_image(event.x, event.y)
        if not (0 <= x < self.image_w and 0 <= y < self.image_h):
            return
        if len(self.points) >= self.count:
            return
        self.points.append([float(x), float(y)])
        self._changed()

    def _left_drag(self, event):
        if self.drag_point is None:
            return
        x, y = self._canvas_to_image(event.x, event.y)
        self.points[self.drag_point] = [
            float(np.clip(x, 0, self.image_w - 1)),
            float(np.clip(y, 0, self.image_h - 1)),
        ]
        self._changed()

    def _left_release(self, _event):
        self.drag_point = None

    def undo(self):
        if self.points:
            self.points.pop()
            self._changed()

    def reset(self):
        if self.points:
            self.points.clear()
            self._changed()

    def _changed(self):
        self.counter.configure(text=f"{len(self.points)}/{self.count}")
        self._draw_points()
        self.on_change()

    def schedule_render(self):
        if self.render_job is not None:
            self.after_cancel(self.render_job)
        self.render_job = self.after(30, self.render)

    def render(self):
        self.render_job = None
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = max(1e-6, self.scale)
        x0 = max(0, int(np.floor(-self.offset_x / scale)))
        y0 = max(0, int(np.floor(-self.offset_y / scale)))
        x1 = min(self.image_w, int(np.ceil((cw - self.offset_x) / scale)))
        y1 = min(self.image_h, int(np.ceil((ch - self.offset_y) / scale)))
        self.canvas.delete("image")
        if x1 > x0 and y1 > y0:
            crop = self.image.crop((x0, y0, x1, y1))
            out_w = max(1, int(round((x1 - x0) * scale)))
            out_h = max(1, int(round((y1 - y0) * scale)))
            resample = Image.Resampling.NEAREST if scale >= 1.0 else Image.Resampling.LANCZOS
            crop = crop.resize((out_w, out_h), resample)
            self.photo = ImageTk.PhotoImage(crop)
            self.canvas.create_image(
                self.offset_x + x0 * scale,
                self.offset_y + y0 * scale,
                image=self.photo,
                anchor="nw",
                tags="image",
            )
            self.canvas.tag_lower("image")
        self._draw_points()

    def _draw_points(self):
        self.canvas.delete("point")
        r = self.marker_radius
        for index, (px, py) in enumerate(self.points, start=1):
            x = self.offset_x + px * self.scale
            y = self.offset_y + py * self.scale
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill="#ff3030", outline="white", width=1, tags="point"
            )
            self.canvas.create_text(
                x + r + 4, y - r - 3, text=str(index), anchor="sw",
                fill="#ff4040", font=("Sans", 10, "bold"), tags="point"
            )


class AlignmentPickerGui:
    def __init__(self, bev_bgr, track_bgr, count=4, marker_radius=4, fullscreen=True):
        self.bev_bgr = bev_bgr
        self.track_bgr = track_bgr
        self.count = count
        self.result = None
        self.preview_current = False

        self.root = tk.Tk()
        self.root.title("LiDAR BEV ↔ 트랙 이미지 4점 정합")
        self.root.minsize(1000, 650)
        if fullscreen:
            self.root.attributes("-fullscreen", True)
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", self._leave_fullscreen)

        instructions = (
            "양쪽 이미지에서 같은 물리 지점을 같은 번호 순서로 4개 선택하세요.  "
            "휠: 확대/축소  |  우클릭/가운데 드래그: 이동  |  기존 점 드래그: 위치 보정"
        )
        ttk.Label(self.root, text=instructions, anchor="center").pack(fill="x", padx=8, pady=6)

        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=2)
        self.bev_panel = ZoomPointCanvas(
            panes, "1. LiDAR BEV", bev_bgr, count, marker_radius, self._points_changed
        )
        self.track_panel = ZoomPointCanvas(
            panes, "2. 트랙 이미지", track_bgr, count, marker_radius, self._points_changed
        )
        panes.add(self.bev_panel, weight=1)
        panes.add(self.track_panel, weight=1)

        footer = ttk.Frame(self.root)
        footer.pack(fill="x", padx=8, pady=7)
        self.status = ttk.Label(footer, text="BEV 0/4 · 트랙 0/4")
        self.status.pack(side="left")
        ttk.Button(footer, text="취소", command=self.cancel).pack(side="right")
        self.complete_button = ttk.Button(
            footer, text="저장 및 완료", command=self.complete, state="disabled"
        )
        self.complete_button.pack(side="right", padx=6)
        self.preview_button = ttk.Button(
            footer, text="정합 미리보기", command=self.preview, state="disabled"
        )
        self.preview_button.pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    def _toggle_fullscreen(self, _event=None):
        value = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not value)

    def _leave_fullscreen(self, _event=None):
        self.root.attributes("-fullscreen", False)

    def _points_changed(self):
        bev_n = len(self.bev_panel.points)
        track_n = len(self.track_panel.points)
        ready = bev_n == self.count and track_n == self.count
        self.status.configure(text=f"BEV {bev_n}/{self.count} · 트랙 {track_n}/{self.count}")
        self.preview_button.configure(state="normal" if ready else "disabled")
        self.preview_current = False
        self.complete_button.configure(state="disabled")

    def _homography_preview(self):
        bev_points = np.asarray(self.bev_panel.points, dtype=np.float32)
        track_points = np.asarray(self.track_panel.points, dtype=np.float32)
        homography, _ = cv2.findHomography(track_points, bev_points, method=0)
        if homography is None:
            raise RuntimeError("선택한 점으로 호모그래피를 계산하지 못했습니다.")
        h, w = self.bev_bgr.shape[:2]
        line_mask = extract_layout_line_mask(self.track_bgr)
        line_alpha = warp_layout_line_mask(
            line_mask, homography, (w, h), supersample=4
        )
        preview = blend_layout_lines(self.bev_bgr, line_alpha, opacity=0.85)
        for idx, (x, y) in enumerate(bev_points, start=1):
            cv2.circle(preview, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
            cv2.putText(preview, str(idx), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        return preview

    def preview(self):
        try:
            image_bgr = self._homography_preview()
        except Exception as exc:
            messagebox.showerror("정합 오류", str(exc), parent=self.root)
            return
        popup = tk.Toplevel(self.root)
        popup.title("정합 미리보기 — 노란 선과 LiDAR BEV가 맞는지 확인")
        popup.geometry("1000x850")
        popup.transient(self.root)
        popup.grab_set()
        ttk.Label(
            popup,
            text="노란 트랙 선이 LiDAR BEV와 맞으면 닫고 ‘저장 및 완료’를 누르세요.",
        ).pack(fill="x", padx=8, pady=6)
        canvas = ZoomPointCanvas(
            popup, "정합 결과", image_bgr, 0, 2, lambda: None
        )
        canvas.pack(fill="both", expand=True, padx=5)
        ttk.Button(popup, text="확인하고 닫기", command=popup.destroy).pack(pady=7)
        self.preview_current = True
        self.complete_button.configure(state="normal")

    def complete(self):
        if not self.preview_current:
            messagebox.showwarning("미리보기 필요", "먼저 정합 미리보기를 확인하세요.", parent=self.root)
            return
        if not messagebox.askyesno(
            "정합 저장",
            "현재 4점과 호모그래피를 저장하고 완료할까요?",
            parent=self.root,
        ):
            return
        self.result = (
            [[int(round(x)), int(round(y))] for x, y in self.bev_panel.points],
            [[int(round(x)), int(round(y))] for x, y in self.track_panel.points],
        )
        self.root.destroy()

    def cancel(self):
        if messagebox.askyesno("취소", "저장하지 않고 정합 작업을 종료할까요?", parent=self.root):
            self.result = None
            self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result


def pick_corresponding_points(
    bev_bgr,
    track_bgr,
    count=4,
    marker_radius=4,
    fullscreen=True,
):
    return AlignmentPickerGui(
        bev_bgr,
        track_bgr,
        count=count,
        marker_radius=marker_radius,
        fullscreen=fullscreen,
    ).run()
