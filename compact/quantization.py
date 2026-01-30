import copy
import torch
import torch.nn as nn
from config import Config
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import torch.ao.quantization as quant
conf = Config()

def dynamic_quantize(fp32_model: nn.Module, arch: str="x86") -> nn.Module:
    """
    Method for eager dynamic quantize for 
    Params:
        model (nn.Module): model to be quantized
    Reurn:
        quantized_rnn (nn.Module): Dynamic quantized model
    """
    if arch == "x86":
        backend = "fbgemm"
    elif arch == "arm":
        backend = "qnnpack"
    else:
        backend = ""
        raise ValueError("arch for backend must be ether 'x86' or 'arm'!")
    
    # Set backend engine and qconfig
    torch.backends.quantized.engine = backend

    # Copy FP32 model and set to inference on cpu
    model_to_quantize = copy.deepcopy(fp32_model)
    model_to_quantize.cpu()
    model_to_quantize.eval()

    int8_model = torch.quantization.quantize_dynamic(
        model_to_quantize,
        {nn.LSTM, nn.GRU, nn.Linear},       # only quantize safe ops
        dtype=torch.qint8
    )

    return int8_model


def static_quantization(
    fp32_model: nn.Module,
    calibration_loader: DataLoader,
    fp32_modules: list,
    example_input: torch.Tensor,
    arch: str="x86"
) -> nn.Module:
    
    if arch == "x86":
        backend = "fbgemm"
    elif arch == "arm":
        backend = "qnnpack"
    else:
        backend = ""
        raise ValueError("arch for backend must be ether 'x86' or 'arm'!")
    
    # Set backend engine and qconfig
    torch.backends.quantized.engine = backend
    qconfig = quant.get_default_qconfig(backend)

    # Create qconfig mapping
    qconfig_mapping = quant.QConfigMapping()
    qconfig_mapping = qconfig_mapping.set_global(qconfig) #
    for mod in fp32_modules:
        qconfig_mapping = qconfig_mapping.set_module_name(mod, None) # FP32

    # Copy FP32 model and set to inference on cpu
    model_to_quantize = copy.deepcopy(fp32_model)
    model_to_quantize.cpu()
    model_to_quantize.eval()

    # Prepare model
    prepared = prepare_fx(
        model_to_quantize,
        qconfig_mapping,
        example_inputs=(example_input,)
    )

    # Calibrate model
    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            prepared(x)
            if i*conf.batch_size > 1000:
                break
    
    # Quantized model with int8 weights
    return convert_fx(prepared)
