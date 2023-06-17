from ctypes import *
from typing import List, Optional

from ..common.common import Underlying, get_device_by_name, _c_lib

class C_Activation(Structure):
    pass

class C_NeuralLayer(Structure):
    pass

class C_GPUBlock(Structure):
    pass

class C_AppliedFunction(Structure):
    _fields_ = [
        ("act", POINTER(C_Activation)),
        ("layer", POINTER(C_NeuralLayer))
    ]

class C_TensorAsList(Structure):
    _fields_ = [
        ("val", POINTER(c_float)),
        ("dim", POINTER(c_int)),
        ("num_dims", c_int)
    ]

class C_Tensor(Structure):
    _fields_ = [
        ("val", POINTER(c_float)),
        ("grad", POINTER(c_float)),
        ("dim", POINTER(c_int)),
        ("num_dims", c_int),
        ("funcs", POINTER(C_AppliedFunction)),
        ("num_funcs", c_int),
        ("max_funcs", c_int),
        ("device", c_int),
        ("gpu_block", POINTER(C_GPUBlock))
    ]

class Tensor(Underlying):
    def __init__(self, value: Optional[List] = None, use_grad: bool = False, 
                 device: str = "cpu", input_tensor: POINTER(C_Tensor) = None, 
                 is_shallow: bool = False):
        def _create_dim(value):
            if not isinstance(value[0], list):
                for i in value:
                    assert not isinstance(i, list)
                return [len(value)]

            for i in value:
                assert isinstance(i, list)
                assert len(value[0]) == len(i)

            return [len(value)] + _create_dim(value[0])


        def _create_c_value(dim: List[int], value: List):
            def _flatten_value(value: List, num_dims: int):
                if (num_dims == 1):
                    return value
                
                num_dims -= 1

                out = []
                for i in value:
                    out += _flatten_value(i, num_dims)

                return out

            num_elem = 1
            for d in dim:
                num_elem *= d

            temp = _flatten_value(value, len(dim))

            c_value = (c_float * num_elem)(*temp)

            return c_value

        self.und = None
        self.is_shallow = is_shallow

        if input_tensor is None and is_shallow:
            raise ArgumentError("Cannot declare a sliced tensor without providing an input tensor (internal)")

        if value is None and input_tensor is None:
            raise ArgumentError("Create a Tensor using either dimensons or an input tensor (internal)")

        device = get_device_by_name(device)

        self._get_slice = self.c_lib.get_slice
        self._get_slice.argtypes = [POINTER(C_Tensor), c_int, c_int, c_int]
        self._get_slice.restype = POINTER(C_Tensor)

        self._get_tensor_at_idx = self.c_lib.get_tensor_at_idx
        self._get_tensor_at_idx.argtypes = [POINTER(C_Tensor), c_int]
        self._get_tensor_at_idx.restype = POINTER(C_Tensor)

        self._get_dimensions = self.c_lib.get_dimensions
        self._get_dimensions.argtypes = [POINTER(C_Tensor)]
        self._get_dimensions.restype = POINTER(c_int)

        self._get_num_dimensions = self.c_lib.get_num_dimensions
        self._get_num_dimensions.argtypes = [POINTER(C_Tensor)]
        self._get_num_dimensions.restype = c_int

        self._get_tensor_as_list = self.c_lib.get_tensor_as_list
        self._get_tensor_as_list.argtypes = [POINTER(C_Tensor)]
        self._get_tensor_as_list.restype = C_TensorAsList

        self._check_tensor_dims = self.c_lib.check_tensor_dims
        self._check_tensor_dims.argtypes = [POINTER(C_Tensor),
                                            POINTER(C_Tensor), c_int]
        self._check_tensor_dims.restype = c_int

        self._grad = self.c_lib.enable_grad
        self._grad.argtypes = [POINTER(C_Tensor)]

        self._no_grad = self.c_lib.disable_grad
        self._no_grad.argtypes = [POINTER(C_Tensor)]

        self._shuffle = self.c_lib.shuffle
        self._shuffle.argtypes = [POINTER(C_Tensor)]

        self._shuffle_two = self.c_lib.shuffle_two
        self._shuffle_two.argtypes = [POINTER(C_Tensor), POINTER(C_Tensor)]
        self._shuffle_two.restype = c_int

        self._destroy_tensor = self.c_lib.destroy_tensor
        self._destroy_tensor.argtypes = [POINTER(C_Tensor)]

        self._destroy_tensor_shallow = self.c_lib.destroy_tensor_shallow
        self._destroy_tensor_shallow.argtypes = [POINTER(C_Tensor)]

        self._destroy_tensor_as_list = self.c_lib.destroy_tensor_as_list
        self._destroy_tensor_as_list.argtypes = [POINTER(C_TensorAsList)]

        if not input_tensor:
            dim = _create_dim(value)

            num_dims = len(dim)
            c_dim = (c_int * num_dims)(*dim)

            c_value = _create_c_value(dim, value)
            _init_tensor = self.c_lib.init_tensor
            _init_tensor.argtypes = [c_void_p, POINTER(c_int), c_int, 
                                     c_bool, c_int]
            _init_tensor.restype = POINTER(C_Tensor)
            input_tensor = _init_tensor(c_value, c_dim, num_dims, use_grad, 
                                        device)

        super().__init__(input_tensor)

    def __getitem__(self, i):
        if isinstance(i, int):
            out = self._get_tensor_at_idx(self.und, i)
            return Tensor(input_tensor=out, is_shallow=True)
        elif isinstance(i, slice):
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else self.und.dim
            step = i.step if i.step is not None else 1
            tensor = self._get_slice(self.und, start, stop, step)
            if tensor.contents.dim == 0:
                raise ArgumentError("Start and/or end is invalid, need to select better range for slice")
            return Tensor(input_tensor=tensor)
        else:
            raise ArgumentError("Incorrect type, make either int or slice")

    def __len__(self):
        return self.und.dim

    def __del__(self):
        if not self.und:
            if self.is_shallow:
                self._destroy_tensor_shallow(self.und)
            else:
                self._destroy_tensor(self.und)

    def get_dims(self) -> List[int]:
        return self._get_dimensions(self.und)[:self.get_num_dims()]

    def get_num_dims(self) -> int:
        return self._get_num_dimensions(self.und)
    
    def grad(self):
        self._grad(self._underlying())

    def no_grad(self):
        self._no_grad(self._underlying())

    def shuffle(self, other = None):
        if other is None:
            self._shuffle(self.und)
        elif isinstance(other, Tensor):
            if self._shuffle_two(self.und, other.und):
                raise ArgumentError("Inputted tensor must have same first dimension as this tensor to sort the same way")
            
    def tolist(self):
        def tolist_recursive(vals, dims, block_idx=0):
            if len(dims) == 0:
                return []

            cur_dim = dims[0]

            if len(dims) == 1:
                return vals[block_idx:block_idx + cur_dim]
            
            next_dim = dims[1]
            
            return [tolist_recursive(vals, dims[1:], 
                                     (block_idx + i) * next_dim) 
                                     for i in range(cur_dim)]

        tensor_as_list = self._get_tensor_as_list(self.und)
        dims = tensor_as_list.dim[:tensor_as_list.num_dims]

        out = tolist_recursive(tensor_as_list.val, dims)

        self._destroy_tensor_as_list(pointer(tensor_as_list))

        return out

def tensor_random(dim: List[int], use_grad: bool = False, device: str = "cpu"):
    num_dims = len(dim)
    c_dim = (c_int * num_dims)(*dim)

    _init_random_tensor = _c_lib.init_random_tensor
    _init_random_tensor.argtypes = [POINTER(c_int), c_int, c_bool, c_int]
    _init_random_tensor.restype = POINTER(C_Tensor)

    device = get_device_by_name(device)

    tensor = _init_random_tensor(c_dim, num_dims, use_grad, device)

    return Tensor(input_tensor=tensor)
