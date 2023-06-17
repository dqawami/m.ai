from ctypes import *
import os

location = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

_c_lib = CDLL(location + "/mesh.so")

class Underlying:
    c_lib = _c_lib

    def __init__(self, und: Structure):
        self.und = und

    def _underlying(self) -> Structure:
        return self.und

(CPU, GPU) = (0, 1)

def get_device_by_name(device: str) -> int:
    if device == "cpu":
        return CPU
    elif device == "gpu":
        return GPU
    else:
        raise ArgumentError("Device not recognized")
