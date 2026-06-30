"""Vehicle pose capture GUI for nav-vla.

Drive (or drag) the ego vehicle to a spot, type a name, pick its role, and save
the vehicle's ground-truth WORLD pose (x, y, yaw) into config/zone_map.yaml.

This tool captures POINT poses only -- the value of using the real vehicle is
its true heading (yaw), which matters for goals like 출발/재위치/parking. Line
and rectangle course geometry is drawn separately in track_roi_editor_node.

Non-ego objects (obstacle spawns, traffic light) go in the `objects` section
with a `kind` (spawn / landmark).

Pose comes from `gz model -m <model> -p` (ground truth: reflects manual drags,
no drift), polled in the background.

Usage:
    ros2 run nav_vla zone_capture_gui_node
    ros2 run nav_vla zone_capture_gui_node --ros-args -p map_path:=/abs/zone_map.yaml
"""

import os
import math
import subprocess
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node

from nav_vla.gz_pose import query_world_pose, resolve_gz_bin

DEFAULT_MAP_PATH = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml"
)
ROLES = ["출발", "재위치", "정차", "종료"]
OBJECT_KINDS = ["spawn", "landmark"]


class ZoneCaptureNode(Node):
    def __init__(self):
        super().__init__("zone_capture_gui_node")
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.poll_period = float(self.declare_parameter("poll_period", 0.3).value)
        self.cmd_vel_topic = self.declare_parameter("cmd_vel_topic", "/cmd_vel").value
        self.map_path = self.declare_parameter("map_path", DEFAULT_MAP_PATH).value
        self.gz_world = self.declare_parameter("gz_world", "default").value
        self.gz_bin = resolve_gz_bin(self.declare_parameter("gz_bin", "").value)

        self.latest = None  # ego ground-truth world pose (x, y, yaw)
        # Poll the model's true world pose via gz CLI (reflects drags, no drift).
        self._poll_run = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

        # Stop control: pausing Gazebo physics is the only reliable stop while
        # other nodes keep publishing /cmd_vel. Zero velocity is a fallback.
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.stop_active = False
        self.create_timer(0.05, self._stop_timer)

        self.get_logger().info(
            f"zone_capture polling gz model={self.model_name}, writing {self.map_path}"
        )

    def _poll_loop(self):
        while self._poll_run:
            pose = query_world_pose(self.gz_bin, self.model_name)
            if pose is not None:
                self.latest = pose
            time.sleep(self.poll_period)

    def _stop_timer(self):
        if self.stop_active:
            self.cmd_pub.publish(Twist())

    def set_stop(self, active):
        self.stop_active = bool(active)
        if self.stop_active:
            self.cmd_pub.publish(Twist())

    def pause_sim(self, pause):
        """Pause/resume Gazebo physics. Returns (ok, message)."""
        service = f"/world/{self.gz_world}/control"
        req = f"pause: {'true' if pause else 'false'}"
        cmd = [
            self.gz_bin, "service", "-s", service,
            "--reqtype", "gz.msgs.WorldControl",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req", req,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            return False, f"gz 실행파일 없음: {self.gz_bin}"
        except subprocess.TimeoutExpired:
            return False, "gz service 응답 없음 (시뮬 실행 중인지 확인)"
        if out.returncode != 0:
            return False, (out.stderr or out.stdout or "gz service 실패").strip()
        return True, ("일시정지" if pause else "재개")

    # ---- map file I/O -------------------------------------------------
    def load_map(self):
        if not os.path.exists(self.map_path):
            data = {}
        else:
            with open(self.map_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        data.setdefault("frame", "world")
        data.setdefault("zones", {})
        data.setdefault("objects", {})
        return data

    def _write(self, data):
        os.makedirs(os.path.dirname(self.map_path), exist_ok=True)
        with open(self.map_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def save_entry(self, section, name, entry):
        data = self.load_map()
        existed = name in data[section]
        data[section][name] = entry
        self._write(data)
        return existed

    def delete_entry(self, section, name):
        data = self.load_map()
        if name in data.get(section, {}):
            del data[section][name]
            self._write(data)
            return True
        return False


class ZoneCaptureGUI:
    def __init__(self, node: ZoneCaptureNode):
        self.node = node

        self.root = tk.Tk()
        self.root.title("nav-vla Zone Capture")
        self.root.geometry("460x520")

        # --- top: live pose + stop -------------------------------------
        top = ttk.LabelFrame(self.root, text="차량 월드 좌표 (ground truth)")
        top.pack(fill="x", padx=8, pady=6)
        self.pose_var = tk.StringVar(value="차량 대기중...")
        ttk.Label(top, textvariable=self.pose_var, font=("monospace", 11)).pack(
            side="left", padx=8, pady=8
        )
        self.stop_var = tk.BooleanVar(value=False)
        self.stop_btn = tk.Button(
            top, text="■ 정지 (시뮬 일시정지)", width=18, command=self.toggle_stop,
            bg="#c0392b", fg="white",
            activebackground="#a93226", activeforeground="white",
        )
        self.stop_btn.pack(side="right", padx=6, pady=4)

        # --- name + category -------------------------------------------
        head = ttk.Frame(self.root)
        head.pack(fill="x", padx=8, pady=4)
        ttk.Label(head, text="이름").pack(side="left", padx=(6, 4))
        self.name_var = tk.StringVar()
        ttk.Entry(head, textvariable=self.name_var, width=16).pack(side="left")
        ttk.Label(head, text="대상").pack(side="left", padx=(12, 4))
        self.category_var = tk.StringVar(value="zone")
        ttk.Radiobutton(
            head, text="zone(주행)", value="zone",
            variable=self.category_var, command=self._on_category,
        ).pack(side="left")
        ttk.Radiobutton(
            head, text="object(물체)", value="object",
            variable=self.category_var, command=self._on_category,
        ).pack(side="left")

        # --- zone panel: roles + tol -----------------------------------
        self.zone_panel = ttk.Frame(self.root)
        role_box = ttk.LabelFrame(self.zone_panel, text="역할 (다중 선택)")
        role_box.pack(fill="x", pady=4)
        self.role_vars = {}
        for r in ROLES:
            v = tk.BooleanVar(value=False)
            self.role_vars[r] = v
            ttk.Checkbutton(role_box, text=r, variable=v).pack(
                side="left", padx=8, pady=4
            )
        tol = ttk.LabelFrame(self.zone_panel, text="도착 허용오차")
        tol.pack(fill="x", pady=4)
        ttk.Label(tol, text="tol pos / yaw").pack(side="left", padx=(8, 4))
        self.tol_pos_var = tk.StringVar(value="0.30")
        self.tol_yaw_var = tk.StringVar(value="0.20")
        ttk.Entry(tol, textvariable=self.tol_pos_var, width=7).pack(side="left", padx=4)
        ttk.Entry(tol, textvariable=self.tol_yaw_var, width=7).pack(side="left", padx=4)
        ttk.Button(
            self.zone_panel, text="현재 차량 위치 캡처 & 저장",
            command=self.capture_save_zone,
        ).pack(anchor="w", pady=4)
        self.zone_panel.pack(fill="x", padx=8, pady=4)

        # --- object panel ----------------------------------------------
        self.object_panel = ttk.Frame(self.root)
        of = ttk.LabelFrame(self.object_panel, text="object (ego가 가지 않는 물체)")
        of.pack(fill="x", pady=4)
        ttk.Label(of, text="kind").pack(side="left", padx=(8, 4))
        self.kind_var = tk.StringVar(value="spawn")
        ttk.OptionMenu(of, self.kind_var, "spawn", *OBJECT_KINDS).pack(side="left")
        ttk.Button(
            self.object_panel, text="현재 차량 위치 캡처 & 저장",
            command=self.capture_save_object,
        ).pack(anchor="w", pady=4)

        # --- saved list -------------------------------------------------
        saved = ttk.LabelFrame(self.root, text="저장 목록")
        saved.pack(fill="both", expand=True, padx=8, pady=6)
        self.listbox = tk.Listbox(saved, height=9)
        self.listbox.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        btns = ttk.Frame(saved)
        btns.pack(side="right", fill="y", padx=4, pady=6)
        ttk.Button(btns, text="새로고침", command=self.refresh_list).pack(pady=2)
        ttk.Button(btns, text="선택 삭제", command=self.delete_selected).pack(pady=2)

        # --- status bar -------------------------------------------------
        self.status_var = tk.StringVar(value=f"map: {self.node.map_path}")
        ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w"
        ).pack(fill="x", side="bottom")

        self.refresh_list()
        self._tick()

    # ---- UI logic ----------------------------------------------------
    def _on_category(self):
        if self.category_var.get() == "zone":
            self.object_panel.pack_forget()
            self.zone_panel.pack(fill="x", padx=8, pady=4, before=self.listbox.master)
        else:
            self.zone_panel.pack_forget()
            self.object_panel.pack(fill="x", padx=8, pady=4, before=self.listbox.master)

    def _tick(self):
        if self.node.latest is None:
            self.pose_var.set("차량 대기중...")
        else:
            x, y, yaw = self.node.latest
            self.pose_var.set(f"x={x:+.3f} y={y:+.3f} yaw={math.degrees(yaw):+.1f}°")
        self.root.after(100, self._tick)

    # ---- stop --------------------------------------------------------
    def toggle_stop(self):
        active = not self.stop_var.get()
        self.stop_btn.config(state="disabled")
        self.status_var.set("일시정지 요청 중..." if active else "재개 요청 중...")

        def worker():
            self.node.set_stop(active)
            ok, msg = self.node.pause_sim(active)
            self.root.after(0, lambda: self._apply_stop(active, ok, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_stop(self, active, ok, msg):
        self.stop_btn.config(state="normal")
        if not ok:
            self.node.set_stop(False)
            self.stop_var.set(False)
            self.stop_btn.config(text="■ 정지 (시뮬 일시정지)", bg="#c0392b")
            self.status_var.set("정지 실패: " + msg)
            messagebox.showwarning("정지 실패", msg)
            return
        self.stop_var.set(active)
        if active:
            self.stop_btn.config(text="▶ 재개 (정지 해제)", bg="#7f8c8d")
            self.status_var.set("시뮬 일시정지됨 — 캡처 후 재개하세요")
        else:
            self.node.set_stop(False)
            self.stop_btn.config(text="■ 정지 (시뮬 일시정지)", bg="#c0392b")
            self.status_var.set("시뮬 재개됨")

    # ---- capture -----------------------------------------------------
    def _name_pose(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("입력 필요", "이름을 입력하세요.")
            return None, None
        if self.node.latest is None:
            messagebox.showwarning("pose 없음", "아직 차량 pose 수신 전입니다.")
            return None, None
        return name, self.node.latest

    def _pose_dict(self, x, y, yaw):
        return {"x": round(x, 4), "y": round(y, 4), "yaw": round(yaw, 4)}

    def capture_save_zone(self):
        name, pose = self._name_pose()
        if name is None:
            return
        try:
            tol = {"pos": float(self.tol_pos_var.get()),
                   "yaw": float(self.tol_yaw_var.get())}
        except ValueError:
            messagebox.showwarning("값 오류", "tol 값이 숫자가 아닙니다.")
            return
        roles = [r for r, v in self.role_vars.items() if v.get()]
        x, y, yaw = pose
        entry = {"role": roles, "geom": "point",
                 "pose": self._pose_dict(x, y, yaw), "tol": tol}
        existed = self.node.save_entry("zones", name, entry)
        self.status_var.set(("덮어씀: " if existed else "저장: ") + name)
        self.refresh_list()

    def capture_save_object(self):
        name, pose = self._name_pose()
        if name is None:
            return
        x, y, yaw = pose
        entry = {"kind": self.kind_var.get(), "pose": self._pose_dict(x, y, yaw)}
        existed = self.node.save_entry("objects", name, entry)
        self.status_var.set(("덮어씀: " if existed else "저장: ") + name + " (object)")
        self.refresh_list()

    # ---- list --------------------------------------------------------
    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        data = self.node.load_map()
        for name, e in data.get("zones", {}).items():
            roles = e.get("role") or [e.get("type", "-")]
            self.listbox.insert(tk.END, f"[zone] {name}  ({'/'.join(roles) or '-'})")
        for name, e in data.get("objects", {}).items():
            self.listbox.insert(tk.END, f"[obj]  {name}  ({e.get('kind', '?')})")

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        raw = self.listbox.get(sel[0])
        section = "zones" if raw.startswith("[zone]") else "objects"
        name = raw.split("]", 1)[1].split()[0]
        if messagebox.askyesno("삭제", f"'{name}' 삭제할까요?"):
            self.node.delete_entry(section, name)
            self.refresh_list()
            self.status_var.set("삭제: " + name)

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = ZoneCaptureNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    gui = ZoneCaptureGUI(node)
    try:
        gui.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
