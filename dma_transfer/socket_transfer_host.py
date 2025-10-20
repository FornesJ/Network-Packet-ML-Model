import torch
import ctypes
import os

so_file = libname = os.path.join(os.getcwd(), "socket_transfer.so")
socket_transfer = ctypes.CDLL(so_file)

# get host_recv_buffer function
host_recv_buffer = socket_transfer.host_recv_buffer
host_recv_buffer.restype = ctypes.c_int # int return type

status = host_recv_buffer()

print(status)

