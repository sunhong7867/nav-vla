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
    decision_making_func_lib = import_module(f"{__package__}.decision_making_func_lib")
except ModuleNotFoundError:
    decision_making_func_lib = _MissingCompiledModule(
        "decision_making_pkg",
        "decision_making_func_lib",
        "decision_making_func_lib.cpython-310.pyc",
    )
