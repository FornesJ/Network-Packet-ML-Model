import torch
import ctypes
import os

# load shared library
so_file = libname = os.path.join(os.getcwd(), "socket_transfer.so")
socket_transfer = ctypes.CDLL(so_file)

# C struct
class CTensor(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("dim", ctypes.c_int),
                ("shape", ctypes.POINTER(ctypes.c_int)),
                ("buffer", ctypes.POINTER(ctypes.c_float))]

# get host_recv_buffer function
host_recv_buffer = socket_transfer.host_recv_buffer
host_recv_buffer.argtypes = [ctypes.POINTER(CTensor)]
host_recv_buffer.restype = ctypes.c_int

# get alloc_tensor function
alloc_tensor = socket_transfer.alloc_tensor
alloc_tensor.restype = ctypes.POINTER(CTensor)

# get free_tensor function
free_tensor = socket_transfer.free_tensor
free_tensor.argtypes = [ctypes.POINTER(CTensor)]

# get tensor struct
ctensor_ptr = alloc_tensor()
if not ctensor_ptr:
    raise RuntimeError("alloc_tensor returned NULL")

# get tensor data form dpu
status = host_recv_buffer(ctensor_ptr)
if status != 0:
    raise RuntimeError("host_recv_buffer failed")

size = ctensor_ptr.contents.size
dim = ctensor_ptr.contents.dim
shape_ptr = ctensor_ptr.contents.shape
buf_ptr = ctensor_ptr.contents.buffer

if not shape_ptr:
    raise RuntimeError("shape pointer is NULL")

# Sanity check
if not buf_ptr:
    raise RuntimeError("buffer pointer is NULL")

# Convert C array to a ctypes array
CArrayInt = ctypes.c_int * dim
c_int_array = ctypes.cast(shape_ptr, ctypes.POINTER(CArrayInt)).contents
shape = tuple(c_int_array)

CArrayFloat = ctypes.c_float * size
c_float_array = ctypes.cast(buf_ptr, ctypes.POINTER(CArrayFloat)).contents

# Create torch tensor and copy data from C buffer
tensor = torch.empty(size, dtype=torch.float32)
ctypes.memmove(tensor.data_ptr(),
               ctypes.addressof(c_float_array),
               size * ctypes.sizeof(ctypes.c_float))

# Free C memory now that we've copied it
free_tensor(ctensor_ptr)

# reshape tensor to original shape
tensor = tensor.reshape(shape)
print("Received tensor:", tensor)
print("Size: ", size)
print("shape: ", shape)
print("dim: ", dim)


