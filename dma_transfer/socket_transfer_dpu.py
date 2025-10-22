import torch
import ctypes
import os

so_file = libname = os.path.join(os.getcwd(), "socket_transfer.so")
socket_transfer = ctypes.CDLL(so_file)

# get dpu_send_buffer function
dpu_send_buffer = socket_transfer.dpu_send_buffer
dpu_send_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), 
                            ctypes.c_int, 
                            ctypes.c_int, 
                            ctypes.POINTER(ctypes.c_int), 
                            ctypes.c_char_p]
dpu_send_buffer.restype = ctypes.c_int # int return type

tensor = torch.randn((2, 4), dtype=torch.float32)
dim = tensor.dim()
shape = list(tensor.shape)
size = tensor.numel()
address = "127.0.0.1"

print(tensor)
print(dim)
print(shape)
print(size)
print(address)

if not tensor.is_contiguous():
    tensor = tensor.contiguous()

tensor_pointer = ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
int_array = ctypes.c_int * dim
shape_pointer = int_array(*shape)
address_pointer = ctypes.c_char_p(address.encode("utf-8"))

status = dpu_send_buffer(tensor_pointer, 
                        ctypes.c_int(size), 
                        ctypes.c_int(dim), 
                        shape_pointer, 
                        address_pointer)
if status != 0:
    print("Send buffer to host failed!")

