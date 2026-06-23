import importlib.util
from pathlib import Path

_LIB_DIR = Path(__file__).resolve().parent / "lib"

def get_py(file_name):
    file_path = _LIB_DIR / file_name

    spec = importlib.util.spec_from_file_location(f"{__name__}.{file_path.stem}", file_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    spec.loader.exec_module(module)
    return module

basic = get_py("012_deploy_lib.py")
