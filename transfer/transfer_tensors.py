import torch
import ctypes
import os
from dotenv import load_dotenv

class DPUSocket:
    def __init__(self, so_file, localhost=True):
        self.socket_transfer = ctypes.CDLL(so_file)

        # dpu socket
        self.socket_ptr = None

        # get send_dpu_buffer
        self.send_dpu_buffer = self.socket_transfer.send_dpu_buffer
        self.send_dpu_buffer.argtypes = [ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_float), 
                                        ctypes.c_int, 
                                        ctypes.c_int, 
                                        ctypes.POINTER(ctypes.c_int),
                                        ctypes.c_int]
        self.send_dpu_buffer.restype = ctypes.c_int # int return type

        # get alloc_dpu_sock
        self.alloc_dpu_sock = self.socket_transfer.alloc_dpu_sock
        self.alloc_dpu_sock.restype = ctypes.c_void_p

        # get open_dpu_socket
        self.open_dpu_socket = self.socket_transfer.open_dpu_socket
        self.open_dpu_socket.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.open_dpu_socket.restype = ctypes.c_int

        # get close_dpu_sock
        self.close_dpu_sock = self.socket_transfer.close_dpu_sock
        self.close_dpu_sock.argtypes = [ctypes.c_void_p]
        
        # wait_ready_signal
        self.wait_ready_signal = self.socket_transfer.wait_ready_signal
        self.wait_ready_signal.argtypes = [ctypes.c_void_p]
        self.wait_ready_signal.restype = ctypes.c_int

        # Load environment variables from the .env file
        load_dotenv()
        if localhost:
            self.address = os.getenv("LOCALHOST")
        else:
            self.address = os.getenv("HOST_IP")

    def open(self):
        if self.socket_ptr != None:
            raise RuntimeError("dpu socket is already open!")
        
        # convert to c type
        address_pointer = ctypes.c_char_p(self.address.encode("utf-8"))
        
        # allocate host buffer
        csock_ptr = self.alloc_dpu_sock()
        if not csock_ptr:
            raise RuntimeError("alloc_dpu_sock returned NULL")

        # open host socket
        if self.open_dpu_socket(csock_ptr, address_pointer) != 0:
            raise RuntimeError("open_dpu_socket failed!")
        
        self.socket_ptr = csock_ptr

    def close(self):
        self.close_dpu_sock(self.socket_ptr)

    def wait(self):
        self.wait_ready_signal(self.socket_ptr)

    def send(self, tensor, log=0):
        # get tensor dim, shape and size
        dim = tensor.dim()
        shape = list(tensor.shape)
        size = tensor.numel()

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # convert to ctype datatypes
        tensor_pointer = ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        int_array = ctypes.c_int * dim
        shape_pointer = int_array(*shape)

        # start socket transfer
        status = self.send_dpu_buffer(self.socket_ptr,
                                    tensor_pointer, 
                                    ctypes.c_int(size), 
                                    ctypes.c_int(dim), 
                                    shape_pointer,
                                    ctypes.c_int(log))
        if status != 0:
            print("Send tensor to host failed!")

class HostSocket:
    # C struct
    class CTensor(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int),
                    ("dim", ctypes.c_int),
                    ("shape", ctypes.POINTER(ctypes.c_int)),
                    ("buffer", ctypes.POINTER(ctypes.c_float))]
        
    def __init__(self, so_file):
        self.socket_transfer = ctypes.CDLL(so_file)

        # host socket
        self.socket_ptr = None
        
        # get recv_host_buffer function
        self.recv_host_buffer = self.socket_transfer.recv_host_buffer
        self.recv_host_buffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.CTensor), ctypes.c_int]
        self.recv_host_buffer.restype = ctypes.c_int

        # get alloc_tensor function
        self.alloc_tensor = self.socket_transfer.alloc_tensor
        self.alloc_tensor.restype = ctypes.POINTER(self.CTensor)

        # get free_tensor function
        self.free_tensor = self.socket_transfer.free_tensor
        self.free_tensor.argtypes = [ctypes.POINTER(self.CTensor)]

        # get alloc_host_sock
        self.alloc_host_sock = self.socket_transfer.alloc_host_sock
        self.alloc_host_sock.restype = ctypes.c_void_p

        # get open_host_socket
        self.open_host_socket = self.socket_transfer.open_host_socket
        self.open_host_socket.argtypes = [ctypes.c_void_p]
        self.open_host_socket.restype = ctypes.c_int

        # get close_host_sock
        self.close_host_sock = self.socket_transfer.close_host_sock
        self.close_host_sock.argtypes = [ctypes.c_void_p]

        # get send_ready_signal
        self.send_ready_signal = self.socket_transfer.send_ready_signal
        self.send_ready_signal.argtypes = [ctypes.c_void_p]
        self.send_ready_signal.restype = ctypes.c_int

    def open(self):
        if self.socket_ptr != None:
            raise RuntimeError("host socket is already open!")
        
        # allocate host buffer
        csock_ptr = self.alloc_host_sock()
        if not csock_ptr:
            raise RuntimeError("alloc_host_sock returned NULL")

        # open host socket
        if self.open_host_socket(csock_ptr) != 0:
            raise RuntimeError("open_host_socket failed!")
        
        self.socket_ptr = csock_ptr

    def close(self):
        self.close_host_sock(self.socket_ptr)

    def signal(self):
        self.send_ready_signal(self.socket_ptr)

    def receive(self, log=0):
        # create tensor struct
        ctensor_ptr = self.alloc_tensor()
        if not ctensor_ptr:
            raise RuntimeError("alloc_tensor returned NULL")

        # get tensor data form dpu

        status = self.recv_host_buffer(self.socket_ptr, ctensor_ptr, ctypes.c_int(log))
        if status != 0:
            self.free_tensor(ctensor_ptr)
            raise RuntimeError("recv_host_buffer failed")

        size = ctensor_ptr.contents.size
        dim = ctensor_ptr.contents.dim
        shape_ptr = ctensor_ptr.contents.shape
        buf_ptr = ctensor_ptr.contents.buffer

        if not shape_ptr:
            self.free_tensor(ctensor_ptr)
            raise RuntimeError("shape pointer is NULL")

        # Sanity check
        if not buf_ptr:
            self.free_tensor(ctensor_ptr)
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
        self.free_tensor(ctensor_ptr)

        # reshape tensor to original shape
        tensor = tensor.reshape(shape)

        return tensor





