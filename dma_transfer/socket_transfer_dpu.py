import torch
import ctypes
import os

so_file = libname = os.path.join(os.getcwd(), "socket_transfer.so")
socket_transfer = ctypes.CDLL(so_file)

# get dpu_send_buffer function
dpu_send_buffer = socket_transfer.dpu_send_buffer
dpu_send_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p]
dpu_send_buffer.restype = ctypes.c_int # int return type

tensor = torch.randn((4, 4), dtype=torch.float32)
print(tensor)
size = tensor.numel()
address = "127.0.0.1"

if not tensor.is_contiguous():
    tensor = tensor.contiguous()

tensor_pointer = ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
address_pointer = ctypes.c_char_p(address.encode("utf-8"))

status = dpu_send_buffer(tensor_pointer, ctypes.c_int(size), address_pointer)

print(status)

