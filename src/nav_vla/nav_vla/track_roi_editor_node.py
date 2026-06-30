"""Track ROI editor for nav-vla.

Big top-down track image (track.png). Click to draw lines and rectangles for the
course geometry (start/reposition/stop/end), name them, and save to JSON with
both pixel coordinates and real /world coordinates.

Pixel<->world mapping is solved empirically: drive the ego vehicle to a few
recognizable features, click the same feature on the image, and bind it to the
current /world_pose. >=3 spread-out pairs give a full affine transform (handles
rotation / scale / flip / offset), bypassing texture-orientation guesswork.

Vehicle poses (heading-sensitive goals) are captured separately by
zone_capture_gui_node.

Usage:
    ros2 run nav_vla track_roi_editor_node
    ros2 run nav_vla track_roi_editor_node --ros-args \
        -p image_path:=/abs/track.png -p out_path:=/abs/track_rois.json
"""

import json
import math
import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import rclpy
from rclpy.node import Node

from nav_vla.gz_pose import query_world_pose, resolve_gz_bin

DEFAULT_IMAGE = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/simulation_pkg/models/race_track/materials/textures/track.png"
)
DEFAULT_OUT = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/config/track_rois.json"
)
ROLES = ["출발", "재위치", "정차", "종료"]


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class RoiEditorNode(Node):
    def __init__(self):
        super().__init__("track_roi_editor_node")
        self.image_path = self.declare_parameter("image_path", DEFAULT_IMAGE).value
        self.out_path = self.declare_parameter("out_path", DEFAULT_OUT).value
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.poll_period = float(self.declare_parameter("poll_period", 0.3).value)
        self.gz_bin = resolve_gz_bin(self.declare_parameter("gz_bin", "").value)
        self.subsample = int(self.declare_parameter("subsample", 2).value)

        self.latest = None  # ego ground-truth world pose (x, y, yaw)
        # Poll the model's true world pose via gz CLI (reflects drags, no drift).
        self._poll_run = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def _poll_loop(self):
        while self._poll_run:
            pose = query_world_pose(self.gz_bin, self.model_name)
            if pose is not None:
                self.latest = pose
            time.sleep(self.poll_period)


class RoiEditorGUI:
    def __init__(self, node: RoiEditorNode):
        self.node = node
        self.scale = node.subsample  # image_px = canvas_px * scale
        self.affine = None  # 2x3 np array, pixel->world
        self.calib = []     # [{"px":[u,v], "world":[x,y]}]
        self.rois = []      # [{"name","role","geom","pixels":[[u,v]...]}]
        self.pending = []   # clicked pixels for the current shape
        self.pending_calib_px = None

        self.root = tk.Tk()
        self.root.title("nav-vla Track ROI Editor")

        try:
            full = tk.PhotoImage(file=node.image_path)
        except tk.TclError as exc:
            raise SystemExit(f"이미지 로드 실패: {node.image_path}\n{exc}")
        self.photo = full.subsample(self.scale, self.scale)
        self.img_w, self.img_h = full.width(), full.height()
        cw, ch = self.photo.width(), self.photo.height()

        # left: canvas
        left = ttk.Frame(self.root)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, width=cw, height=ch, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)

        # right: controls
        right = ttk.Frame(self.root, width=300)
        right.pack(side="right", fill="y")
        self._build_controls(right)

        self._redraw()
        self._tick()

    # ---- controls -----------------------------------------------------
    def _build_controls(self, p):
        self.live_var = tk.StringVar(value="world 대기중...")
        ttk.Label(p, textvariable=self.live_var, font=("monospace", 10)).pack(
            anchor="w", padx=6, pady=4
        )
        self.hover_var = tk.StringVar(value="hover: -")
        ttk.Label(p, textvariable=self.hover_var, font=("monospace", 9)).pack(
            anchor="w", padx=6
        )

        # mode: calib vs roi
        self.mode_var = tk.StringVar(value="roi")
        mode = ttk.LabelFrame(p, text="모드")
        mode.pack(fill="x", padx=6, pady=6)
        ttk.Radiobutton(mode, text="ROI 그리기", value="roi",
                        variable=self.mode_var, command=self._reset_pending).pack(anchor="w")
        ttk.Radiobutton(mode, text="캘리브레이션", value="calib",
                        variable=self.mode_var, command=self._reset_pending).pack(anchor="w")

        # ROI panel
        roi = ttk.LabelFrame(p, text="ROI")
        roi.pack(fill="x", padx=6, pady=6)
        gf = ttk.Frame(roi); gf.pack(fill="x", pady=2)
        ttk.Label(gf, text="geom").pack(side="left", padx=(2, 4))
        self.geom_var = tk.StringVar(value="line")
        ttk.Radiobutton(gf, text="선", value="line", variable=self.geom_var,
                        command=self._reset_pending).pack(side="left")
        ttk.Radiobutton(gf, text="사각형", value="rect", variable=self.geom_var,
                        command=self._reset_pending).pack(side="left")
        nf = ttk.Frame(roi); nf.pack(fill="x", pady=2)
        ttk.Label(nf, text="이름").pack(side="left", padx=(2, 4))
        self.name_var = tk.StringVar()
        ttk.Entry(nf, textvariable=self.name_var, width=16).pack(side="left")
        rf = ttk.Frame(roi); rf.pack(fill="x", pady=2)
        self.role_vars = {}
        for r in ROLES:
            v = tk.BooleanVar(value=False)
            self.role_vars[r] = v
            ttk.Checkbutton(rf, text=r, variable=v).pack(side="left")
        self.pending_var = tk.StringVar(value="점 0개 (선=2, 사각형=4 클릭)")
        ttk.Label(roi, textvariable=self.pending_var).pack(anchor="w", pady=2)
        ttk.Button(roi, text="현재 도형 취소", command=self._reset_pending).pack(anchor="w")

        # calibration panel
        cal = ttk.LabelFrame(p, text="캘리브 (차량 위치 ↔ 픽셀)")
        cal.pack(fill="x", padx=6, pady=6)
        ttk.Label(cal, text="① 캘리브 모드로 특징점 클릭\n② 차를 그 지점에 두고 ↓",
                  justify="left").pack(anchor="w", padx=2)
        ttk.Button(cal, text="클릭점 ↔ 현재 world 묶기",
                   command=self._add_calib).pack(fill="x", pady=2)
        self.calib_var = tk.StringVar(value="0쌍 (≥3 필요)")
        ttk.Label(cal, textvariable=self.calib_var).pack(anchor="w")
        ttk.Button(cal, text="캘리브 계산", command=self._solve_calib).pack(fill="x", pady=2)
        self.resid_var = tk.StringVar(value="미캘리브")
        ttk.Label(cal, textvariable=self.resid_var).pack(anchor="w")

        # saved list
        sv = ttk.LabelFrame(p, text="저장 목록")
        sv.pack(fill="both", expand=True, padx=6, pady=6)
        self.listbox = tk.Listbox(sv, height=8)
        self.listbox.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(sv, text="선택 삭제", command=self._delete_sel).pack(fill="x")

        # file
        ff = ttk.Frame(p); ff.pack(fill="x", padx=6, pady=6)
        ttk.Button(ff, text="JSON 저장", command=self._save).pack(side="left", expand=True, fill="x")
        ttk.Button(ff, text="불러오기", command=self._load).pack(side="left", expand=True, fill="x")
        self.status_var = tk.StringVar(value=os.path.basename(self.node.out_path))
        ttk.Label(p, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x", side="bottom"
        )

    # ---- coordinate helpers ------------------------------------------
    def _canvas_to_px(self, cx, cy):
        return (cx * self.scale, cy * self.scale)

    def _px_to_canvas(self, u, v):
        return (u / self.scale, v / self.scale)

    def _px_to_world(self, u, v):
        if self.affine is None:
            return None
        x, y = self.affine @ np.array([u, v, 1.0])
        return [round(float(x), 4), round(float(y), 4)]

    # ---- click handling ----------------------------------------------
    def _need(self):
        return 2 if self.geom_var.get() == "line" else 4

    def _on_click(self, ev):
        u, v = self._canvas_to_px(ev.x, ev.y)
        if self.mode_var.get() == "calib":
            self.pending_calib_px = [round(u, 1), round(v, 1)]
            self.resid_var.set(f"클릭 px=({u:.0f},{v:.0f}) → world 묶기 누르세요")
            self._redraw()
            return
        # roi mode: line = 2 clicks, rect = 4 clicks (arbitrary quad)
        need = self._need()
        self.pending.append([round(u, 1), round(v, 1)])
        self.pending_var.set(f"점 {len(self.pending)}개 / {need}")
        if len(self.pending) >= need:
            self._commit_roi()
        self._redraw()

    def _on_motion(self, ev):
        u, v = self._canvas_to_px(ev.x, ev.y)
        od = self._px_to_world(u, v)
        if od is None:
            self.hover_var.set(f"hover px=({u:.0f},{v:.0f})  world=미캘리브")
        else:
            self.hover_var.set(f"hover px=({u:.0f},{v:.0f})  world=({od[0]},{od[1]})")

    def _reset_pending(self):
        self.pending = []
        self.pending_calib_px = None
        self.pending_var.set(f"점 0개 / {self._need()}")
        self._redraw()

    def _commit_roi(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("입력 필요", "이름을 입력하세요.")
            self.pending = []
            self._redraw()
            return
        # role is optional: pure markers (e.g. crosswalk) save with role: []
        roles = [r for r, v in self.role_vars.items() if v.get()]
        self.rois.append({
            "name": name,
            "role": roles,
            "geom": self.geom_var.get(),
            "pixels": list(self.pending),
        })
        self.status_var.set("ROI 추가: " + name)
        self.pending = []
        self.pending_var.set(f"점 0개 / {self._need()}")
        self._refresh_list()

    # ---- calibration --------------------------------------------------
    def _add_calib(self):
        if self.pending_calib_px is None:
            messagebox.showwarning("순서", "먼저 캘리브 모드에서 이미지를 클릭하세요.")
            return
        if self.node.latest is None:
            messagebox.showwarning("pose 없음", "아직 /world_pose 수신 전입니다.")
            return
        x, y, _ = self.node.latest
        self.calib.append({
            "px": self.pending_calib_px,
            "world": [round(x, 4), round(y, 4)],
        })
        self.pending_calib_px = None
        self.calib_var.set(f"{len(self.calib)}쌍 (≥3 필요)")
        self.resid_var.set("쌍 추가됨 — 계산 누르세요")
        self._redraw()

    def _solve_calib(self):
        if len(self.calib) < 3:
            messagebox.showwarning("쌍 부족", "캘리브 쌍이 3개 이상 필요합니다.")
            return
        A = np.array([[c["px"][0], c["px"][1], 1.0] for c in self.calib])
        bx = np.array([c["world"][0] for c in self.calib])
        by = np.array([c["world"][1] for c in self.calib])
        cx, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
        self.affine = np.vstack([cx, cy])
        pred = A @ self.affine.T
        gt = np.vstack([bx, by]).T
        rms = float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))
        self.resid_var.set(f"캘리브 완료, RMS={rms:.3f} m")
        self.status_var.set("캘리브 완료")
        self._refresh_list()

    # ---- list / draw --------------------------------------------------
    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for r in self.rois:
            od = "✓" if self.affine is not None else "px"
            role_str = "/".join(r["role"]) or "-"
            self.listbox.insert(
                tk.END, f"{r['name']} [{role_str}·{r['geom']}·{od}]"
            )

    def _delete_sel(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        del self.rois[sel[0]]
        self._refresh_list()
        self._redraw()

    def _redraw(self):
        self.canvas.delete("ov")
        # calibration markers
        for c in self.calib:
            u, v = self._px_to_canvas(*c["px"])
            self.canvas.create_oval(u - 4, v - 4, u + 4, v + 4, outline="yellow",
                                    width=2, tags="ov")
        if self.pending_calib_px is not None:
            u, v = self._px_to_canvas(*self.pending_calib_px)
            self.canvas.create_oval(u - 5, v - 5, u + 5, v + 5, outline="orange",
                                    width=2, tags="ov")
        # committed ROIs: lines cyan, rectangles yellow
        for r in self.rois:
            color = "#ffd400" if r["geom"] == "rect" else "#00e5ff"
            self._draw_shape(r["geom"], r["pixels"], color, r["name"])
        # pending shape
        if self.pending:
            self._draw_shape(self.geom_var.get(), self.pending, "#ff3b3b", "")

    def _draw_shape(self, geom, pixels, color, label):
        pts = [self._px_to_canvas(*p) for p in pixels]
        for (u, v) in pts:
            self.canvas.create_oval(u - 3, v - 3, u + 3, v + 3, fill=color,
                                    outline=color, tags="ov")
        lx = ly = None
        if geom == "line":
            if len(pts) >= 2:
                (x1, y1), (x2, y2) = pts[0], pts[1]
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags="ov")
                lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
        else:  # rect: quad from up to 4 points
            for i in range(len(pts) - 1):
                self.canvas.create_line(*pts[i], *pts[i + 1], fill=color, width=2,
                                        tags="ov")
            if len(pts) == 4:  # close the quad
                self.canvas.create_line(*pts[3], *pts[0], fill=color, width=2, tags="ov")
            if pts:
                lx = sum(p[0] for p in pts) / len(pts)
                ly = sum(p[1] for p in pts) / len(pts)
        if label and lx is not None:
            self.canvas.create_text(lx, ly, text=label, fill=color,
                                    font=("sans", 9, "bold"), tags="ov")

    # ---- save / load --------------------------------------------------
    def _roi_with_world(self, r):
        out = dict(r)
        if self.affine is not None:
            out["world"] = [self._px_to_world(u, v) for (u, v) in r["pixels"]]
            if r["geom"] == "rect" and out["world"]:
                cx = round(sum(p[0] for p in out["world"]) / len(out["world"]), 4)
                cy = round(sum(p[1] for p in out["world"]) / len(out["world"]), 4)
                out["center_world"] = [cx, cy]
        return out

    def _save(self):
        data = {
            "image": os.path.basename(self.node.image_path),
            "image_size": [self.img_w, self.img_h],
            "frame": "world",
            "calibration": {
                "points": self.calib,
                "affine": self.affine.tolist() if self.affine is not None else None,
            },
            "rois": [self._roi_with_world(r) for r in self.rois],
        }
        os.makedirs(os.path.dirname(self.node.out_path), exist_ok=True)
        with open(self.node.out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.status_var.set(f"저장: {self.node.out_path}")

    def _load(self):
        if not os.path.exists(self.node.out_path):
            messagebox.showinfo("없음", "저장된 JSON이 없습니다.")
            return
        with open(self.node.out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.calib = data.get("calibration", {}).get("points", [])
        aff = data.get("calibration", {}).get("affine")
        self.affine = np.array(aff) if aff else None
        self.rois = [{"name": r["name"], "role": r["role"], "geom": r["geom"],
                      "pixels": r["pixels"]} for r in data.get("rois", [])]
        self.calib_var.set(f"{len(self.calib)}쌍 (≥3 필요)")
        self.resid_var.set("불러옴" + ("(캘리브 있음)" if self.affine is not None else ""))
        self._refresh_list()
        self._redraw()
        self.status_var.set("불러옴")

    # ---- loop ---------------------------------------------------------
    def _tick(self):
        if self.node.latest is None:
            self.live_var.set("world 대기중...")
        else:
            x, y, yaw = self.node.latest
            self.live_var.set(f"차량 x={x:+.3f} y={y:+.3f} yaw={math.degrees(yaw):+.0f}°")
        self.root.after(100, self._tick)

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = RoiEditorNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    gui = RoiEditorGUI(node)
    try:
        gui.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
