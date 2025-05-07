__version__ = "0.0.1"

def load_module_from_pyd(module_name):
    import os, sys, glob
    import importlib.util
    _module_root = os.path.abspath(os.path.dirname(__file__))
    _lib_root = os.path.join(_module_root, "lib")
    _lib_paths = glob.glob(os.path.join(_module_root, "lib", f"{module_name}*.pyd"))
    for _lib_path in _lib_paths:
        try:
            _spec = importlib.util.spec_from_file_location(module_name, _lib_path)
            _module = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_module)
        except Exception as e:
            print(e)
        finally:
            pass

load_module_from_pyd("PyEdgePadding")

import PyEdgePadding as edge_padding
__all__ = ["__version__", "edge_padding"]