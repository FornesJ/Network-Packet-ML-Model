import torch
import ctypes

class DPUSocket:
    def __init__(self, so_file, host_address):
        self.socket_transfer = ctypes.CDLL(so_file)
        self.dpu_send_buffer = self.socket_transfer.dpu_send_buffer
        self.dpu_send_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), 
                                        ctypes.c_int, 
                                        ctypes.c_int, 
                                        ctypes.POINTER(ctypes.c_int), 
                                        ctypes.c_char_p]
        self.dpu_send_buffer.restype = ctypes.c_int # int return type
        self.address = host_address
    
    def send(self, tensor):
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
        address_pointer = ctypes.c_char_p(self.address.encode("utf-8"))

        # start socket transfer
        status = self.dpu_send_buffer(tensor_pointer, 
                        ctypes.c_int(size), 
                        ctypes.c_int(dim), 
                        shape_pointer, 
                        address_pointer)
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
        
        # get host_recv_buffer function
        self.host_recv_buffer = self.socket_transfer.host_recv_buffer
        self.host_recv_buffer.argtypes = [ctypes.POINTER(self.CTensor)]
        self.host_recv_buffer.restype = ctypes.c_int

        # get alloc_tensor function
        self.alloc_tensor = self.socket_transfer.alloc_tensor
        self.alloc_tensor.restype = ctypes.POINTER(self.CTensor)

        # get free_tensor function
        self.free_tensor = self.socket_transfer.free_tensor
        self.free_tensor.argtypes = [ctypes.POINTER(self.CTensor)]

    def receive(self):
        # create tensor struct
        ctensor_ptr = self.alloc_tensor()
        if not ctensor_ptr:
            raise RuntimeError("alloc_tensor returned NULL")

        # get tensor data form dpu
        status = self.host_recv_buffer(ctensor_ptr)
        if status != 0:
            self.free_tensor(ctensor_ptr)
            raise RuntimeError("host_recv_buffer failed")

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





