import importlib
import importlib.util
import os
import sys

# 模块名（不含后缀）
_native_name = __package__ + ".PyEdgePadding" if __package__ else ".PyEdgePadding"

def edge_padding_uint8_custom_mask(input, mask):
    raise NotImplementedError

def edge_padding_uint8(input):
    raise NotImplementedError

try:
    # 尝试导入 native 扩展（cuda_inpaint.pyd / .so）
    _mod = importlib.import_module(_native_name)

    # 将模块的符号导入当前命名空间（让用户以统一方式使用）
    from_types = dir(_mod)
    for name in from_types:
        if not name.startswith("_"):
            globals()[name] = getattr(_mod, name)

    __all__ = [name for name in from_types if not name.startswith("_")]

except ImportError:
    pass