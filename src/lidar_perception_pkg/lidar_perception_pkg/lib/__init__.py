from importlib import import_module


class _MissingCompiledModule:
    def __init__(self, package_name, module_name, pyc_name):
        self._package_name = package_name
        self._module_name = module_name
        self._pyc_name = pyc_name

    def __getattr__(self, attr_name):
        raise ImportError(
            f"{self._package_name}.{self._module_name} source file is missing. "
            f"Only legacy bytecode '{self._pyc_name}' exists, and Python 3.12 "
            f"cannot import Python 3.10 .pyc files."
        )


try:
    lidar_perception_func_lib = import_module(f"{__package__}.lidar_perception_func_lib")
except ModuleNotFoundError:
    lidar_perception_func_lib = _MissingCompiledModule(
        "lidar_perception_pkg",
        "lidar_perception_func_lib",
        "lidar_perception_func_lib.cpython-310.pyc",
    )
