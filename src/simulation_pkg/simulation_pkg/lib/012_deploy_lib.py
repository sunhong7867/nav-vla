import os
import random
import re
import shutil
import subprocess
import tempfile
import shlex
import numpy as np
from datetime import datetime
from pathlib import Path
import time
from ament_index_python.packages import get_package_share_directory

# from simulation_pkg import get_pyc

random.seed(time.time())
"""
파일 수정 후 그냥 실행하면 pyc 파일 옮겨짐
"""


# huggingface.py의 sim.pt 파일 다운로드 코드
def check_and_download_model(file, destination_path):
    if not os.path.exists(destination_path):
        import shutil
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id='gogoring/simulation_ws', filename=file)
        shutil.copy(model_path, destination_path)


    
HOME = os.path.expanduser("~")


def _package_root():
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        source_candidate = parent / "src" / "simulation_pkg"
        if (source_candidate / "models").is_dir():
            return source_candidate
        if parent.name == "simulation_pkg" and (parent / "models").is_dir():
            return parent

    return Path(get_package_share_directory("simulation_pkg"))
    
def get_base_path(extra_dirs=None, repeat_last=False):
    base_path = _package_root()

    if repeat_last:
        base_path = base_path / "simulation_pkg"

    if extra_dirs:
        base_path = base_path.joinpath(*extra_dirs)
    
    return str(base_path)

def get_pkg(): # 패키지 경로
    return get_base_path()

def get_path(): # 노드 파일 경로
    return get_base_path(repeat_last=True)

def get_model(file_name=None): # model.sdf 파일 경로
    return get_base_path(["models", file_name, "model.sdf"])


def get_model_dir(file_name=None):
    return get_base_path(["models", file_name])

def get_data(file_name=None): # data 폴더 안의 파일 경로
    return get_base_path(["data", file_name], repeat_last=True)

def get_lib(file_name=None): # lib/pyc 폴더 안의 파일 경로
    return get_base_path(["lib", "pyc", file_name], repeat_last=True)

def get_time(is_img=True):
    now = datetime.now()
    now = now.strftime('%y%m%d_%H%M%S')
    if is_img:
        result = now + '.png' 
    return result    


def wait_for_gz_service(service_name, timeout_sec=40):
    deadline = time.time() + timeout_sec
    command = (
        "source /opt/ros/jazzy/setup.bash && "
        "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz service -l"
    )

    while time.time() < deadline:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
        )
        if service_name in result.stdout:
            return True
        time.sleep(1.0)

    return False


def wait_for_gz_model(model_name, timeout_sec=15):
    deadline = time.time() + timeout_sec
    command = "source /opt/ros/jazzy/setup.bash && " + " ".join(
        shlex.quote(part)
        for part in [
            "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz",
            "model",
            "-m",
            model_name,
            "-p",
            "--force-version",
            "8",
        ]
    )

    while time.time() < deadline:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and "Pose" in result.stdout:
            return True
        time.sleep(0.5)

    return False


def wait_for_gz_model_removed(model_name, timeout_sec=10):
    deadline = time.time() + timeout_sec
    command = "source /opt/ros/jazzy/setup.bash && " + " ".join(
        shlex.quote(part)
        for part in [
            "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz",
            "model",
            "-m",
            model_name,
            "-p",
            "--force-version",
            "8",
        ]
    )

    while time.time() < deadline:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or "Pose" not in result.stdout:
            return True
        time.sleep(0.5)

    return False


def remove_model(entity_name):
    if not wait_for_gz_service("/world/default/remove", timeout_sec=5):
        raise RuntimeError("Timed out waiting for /world/default/remove")

    if not wait_for_gz_model(entity_name, timeout_sec=1):
        print(f"[remove_model] {entity_name} is already absent")
        return

    remove_cmd = [
        "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz", "service",
        "-s", "/world/default/remove",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", f'name: "{entity_name}" type: MODEL',
    ]
    remove_cmd_shell = "source /opt/ros/jazzy/setup.bash && " + " ".join(
        shlex.quote(part) for part in remove_cmd
    )

    result = subprocess.run(
        ["bash", "-lc", remove_cmd_shell],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to remove {entity_name}: "
            f"{result.stderr.strip() or result.stdout.strip() or result.returncode}"
        )

    if not wait_for_gz_model_removed(entity_name, timeout_sec=8):
        raise RuntimeError(f"Timed out waiting for {entity_name} to be removed")

    print(f"[remove_model] removed {entity_name}")


def reset_model(entity_name, model_name, coordinates):
    remove_model(entity_name)
    time.sleep(0.5)
    load_model(entity_name, model_name, coordinates, skip_if_exists=False)



# 시뮬레이션 세팅
def load_model(entity_name, model_name, random_coordinates, skip_if_exists=True):
    
    x, y, z, roll, pitch, yaw = random_coordinates
    
    source_model_dir = Path(get_model_dir(model_name))
    temp_root_dir = Path(tempfile.mkdtemp(prefix=f"{model_name}_"))
    temp_model_dir = temp_root_dir / model_name
    shutil.copytree(source_model_dir, temp_model_dir)

    texture_dir = temp_model_dir / "materials" / "textures"
    meshes_dir = temp_model_dir / "meshes"

    if texture_dir.exists() and meshes_dir.exists():
        for texture_file in texture_dir.iterdir():
            if texture_file.is_file():
                shutil.copy2(texture_file, meshes_dir / texture_file.name)

    if meshes_dir.exists():
        for mtl_path in meshes_dir.glob("*.mtl"):
            mtl_text = mtl_path.read_text()
            mtl_text = mtl_text.replace("../materials/textures/", "")
            mtl_text = mtl_text.replace("..\\materials\\textures\\", "")
            mtl_path.write_text(mtl_text)

    model_file = temp_model_dir / "model.sdf"
    model_text = model_file.read_text()
    model_text = model_text.replace(
        f"model://{model_name}/",
        f"file://{temp_model_dir.as_posix()}/",
    )
    if model_name == "traffic":
        model_text = re.sub(
            r"\s*<plugin\b[^>]*>.*?</plugin>",
            "",
            model_text,
            flags=re.DOTALL,
        )
    model_file.write_text(model_text)

    if not wait_for_gz_service("/world/default/create"):
        raise RuntimeError("Timed out waiting for /world/default/create")

    if skip_if_exists and wait_for_gz_model(entity_name, timeout_sec=1):
        print(f"[load_model] {entity_name} already exists; skipping spawn")
        return

    create_cmd = [
        "ros2", "run", "ros_gz_sim", "create",
        "-file", str(model_file),
        "-name", entity_name,
        "-x", str(x),
        "-y", str(y),
        "-z", str(z),
        "-R", str(roll),
        "-P", str(pitch),
        "-Y", str(yaw),
    ]
    create_cmd_shell = "source /opt/ros/jazzy/setup.bash && " + " ".join(
        shlex.quote(part) for part in create_cmd
    )

    last_error = None
    for attempt in range(1, 6):
        try:
            result = subprocess.run(
                ["bash", "-lc", create_cmd_shell],
                check=True,
                capture_output=True,
                text=True,
            )
            print(
                f"[load_model] spawned {entity_name} at "
                f"x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw:.3f}"
            )
            if result.stdout.strip():
                print(f"[load_model] create stdout: {result.stdout.strip()}")
            if not wait_for_gz_model(entity_name, timeout_sec=5):
                print(
                    f"[load_model] warning: model list did not confirm {entity_name}; "
                    "not retrying because create returned success"
                )
            return
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            last_error = exc
            if attempt == 5:
                break
            print(
                f"[load_model] spawn retry {attempt}/5 for {entity_name}: "
                f"{getattr(exc, 'stderr', '') .strip() or getattr(exc, 'stdout', '') .strip() or str(exc)}"
            )
            time.sleep(2.0)

    if last_error is not None:
        raise RuntimeError(
            f"Failed to spawn {entity_name}: "
            f"{getattr(last_error, 'stderr', '').strip() or getattr(last_error, 'stdout', '').strip() or str(last_error)}"
        ) from last_error

def driving_ego():  
    # Fixed start pose captured from Gazebo at the desired lane position.
    random_x = 3.7
    random_y = 24.594280242919915
    
    z = 0.012649910524487494

    r = -0.000000030954
    p = 0.000000003609
    y = -1.570739951736

    return random_x, random_y, z, r, p, y


# Current driving mission pose set.
# Ego starts at lane2, heading toward negative Y. These fixed points keep the
# obstacle/light scene reproducible for chat/VLA demos and paper experiments.
DRIVING_TRACK_Z = 0.012649910524487494
DRIVING_TRACK_ROLL = -0.000000030954
DRIVING_TRACK_PITCH = 0.000000003609
DRIVING_TRACK_YAW_SOUTH = -1.570739951736

MISSION_TRAFFIC_LIGHT_POSE = (
    5.79853,
    -19.7073,
    0.051603,
    0.0,
    0.0,
    -1.54044,
)

MISSION_OBSTACLE_SPECS = [
    # Lane obstacles. Ego starts in lane2 heading toward negative Y.
    (
        "obstacle1",
        "hatchback_green",
        (-13.1327, 17.5616, 0.01265, 0.0, 0.0, -2.25061),
    ),
    (
        "obstacle2",
        "hatchback_red",
        (-13.1074, -9.33681, 0.01265, 0.0, 0.0, -1.48022),
    ),
    (
        "obstacle3",
        "hatchback_blue",
        (-17.6886, 3.04956, 0.01265, 0.0, 0.0, -1.47976),
    ),
    # Parking-space obstacles.
    (
        "obstacle4",
        "hatchback_yellow",
        (3.67792, -5.7237, 0.01265, 0.0, -0.0, 3.12854),
    ),
    (
        "obstacle5",
        "hatchback_green",
        (3.76991, 1.54264, 0.01265, 0.0, -0.0, 3.11838),
    ),
]

def old_obstacle_stand(): # 신호등
    x = 23.185000
    
    y = 4.388550
    
    y_min = 0.383322
    y_max = 5.920192    
    random_y = random.uniform(y_min, y_max)

    z = 0.0
    r = 0.0
    p = 0.0
    y = 0.0

    return x, y, z, r, p, y

def traffic_light_stand(): # 신호등
    return MISSION_TRAFFIC_LIGHT_POSE


def mission_obstacle_specs():
    return MISSION_OBSTACLE_SPECS

# 장애물 회피 차량 + 범위 지정 + 랜덤
model_types = ["prius_hybrid_ob1", "prius_hybrid_ob2", "prius_hybrid_ob3", "hatchback_green", "hatchback_yellow"] 

obstacle_coordinates1 = (12.251981, -15.909271, 0.00, 0.00, 0.00, 2.484252)
obstacle_coordinates_1= (-3.659642, 8.710748, 0.00, 0.00, 0.00, -0.013934)
obstacle_coordinates_2= (-3.659642, 2.037476, 0.00, 0.00, 0.00, -0.013934)

obstacle_coordinates2 = [(11.884767, 11.605120), (12.040719, 10.060495), (12.230394, 8.181866)]
obstacle_coordinates3 = [(16.106836, -0.111269), (16.281788, -1.844067), (16.366463, -3.446680)]

def obstacle_coord(coordinates):  
    x_obstacle, y_obstacle = random.choice(coordinates)
    
    z = 0.0
    r = 0.0
    p = 0.0

    y = 3.25


    return x_obstacle, y_obstacle, z, r, p, y

parking_start = [(-1.672862, -16.311572, 0.011641, -0.000000, 0.00, -3.133789),
                 (-1.681217, -15.244810, 0.011641, -0.000000, 0.00, -3.133795),  
                 (-0.772971, -15.237810, 0.011641, -0.000000, 0.00, -3.133797),  
                 (-0.764668, -16.302988, 0.011641, -0.000000, 0.00, -3.133797),
                ]

parking_ego = random.choice(parking_start)


# 주차 칸의 범위를 정의 (x_min, x_max, y_min, y_max, z_min, z_max, yaw_min, yaw_max)
parking_zones = {
    1: {"x_range": (2.633951, 3.147836), "y_range": (1.190888, 1.854545), "yaw_range": (-3.132635, -3.128233)},
    2: {"x_range": (2.543644, 3.121653 ), "y_range": (-1.986755, -1.258803), "yaw_range": (-3.132635, -3.128233)},
    3: {"x_range": (2.722283, 3.126416), "y_range": (-5.262367, -4.567991), "yaw_range": (-3.132635, -3.128233)},
    4: {"x_range": (2.583159, 3.261442), "y_range": (-8.441107, -7.685804), "yaw_range": (-3.132635, -3.128233)}
}   

def parking_coord(zone):  
    x_obstacle = random.uniform(zone["x_range"][0], zone["x_range"][1])
    y_obstacle = random.uniform(zone["y_range"][0], zone["y_range"][1])
    z = 0.0
    r = 0.0
    p = 0.0
    y = random.uniform(zone["yaw_range"][0], zone["yaw_range"][1])
    
    return x_obstacle, y_obstacle, z, r, p, y

# 주차 가능한 조합: 1번과 3번 / 2번과 4번에만 주차
valid_parking_pairs = [(1, 3), (2, 4)]
parking_pair = random.choice(valid_parking_pairs)
selected_zone1, selected_zone2 = parking_pair

# 주차할 칸을 무작위로 선택하고, 좌표를 생성
selected_zone_num = random.choice(parking_pair)
parking_car1 = parking_coord(parking_zones[selected_zone1])
parking_car2 = parking_coord(parking_zones[selected_zone2])
    


    
if __name__ == "__main__":
    import os
    import py_compile

    # 경로 설정
    username = os.getlogin()
    base_path = f"/home/{username}/ros2_autonomous_vehicle_simulation/src/simulation_pkg/simulation_pkg/lib"

    pyc_folder = os.path.join(base_path, "pyc")
    target_file = "012_deploy_lib.py"

    # 1. __pycache__ 폴더 삭제
    pycache_path = os.path.join(base_path, "__pycache__")
    if os.path.exists(pycache_path):
        import shutil
        shutil.rmtree(pycache_path)
        #print(f"Deleted: {pycache_path}")
        print("pyc 폴더 삭제")

    # 2. pyc 폴더 내 기존 pyc 파일 삭제
    pyc_file_to_delete = os.path.join(pyc_folder, "012_deploy_lib.cpython-310.pyc")
    if os.path.exists(pyc_file_to_delete):
        os.remove(pyc_file_to_delete)
        #print(f"Deleted: {pyc_file_to_delete}")
        print("기존 pyc 파일 삭제")

    # 3. 특정 파일 컴파일
    source_file_path = os.path.join(base_path, target_file)
    if os.path.exists(source_file_path):
        py_compile.compile(source_file_path, cfile=pyc_file_to_delete)
        #print(f"Compiled: {source_file_path} -> {pyc_file_to_delete}")
        print("끝")
    else:
        print(f"File not found: {source_file_path}")
