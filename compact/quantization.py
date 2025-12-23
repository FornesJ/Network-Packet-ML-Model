import torch
import torch.nn as nn
import platform
from config import Config
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import torch.ao.quantization as quant
conf = Config()

class QuantizeModel():
    """
    QuantizedModel class can be used for dynamic or static quantization of a nn.Module model
    """
    def __init__(self):
        """
        Constructor for QuantizeModel class
        Defines backend for x86 or arm processor
        """
        cpu = platform.processor()
        if cpu == "x86_64" or cpu == "AMD64":
            self.backend = "fbgemm"
        elif cpu == "arm64" or cpu == "aarch64":
            self.backend = "qnnpack"
        else:
            raise ValueError(f"Platform architecture must be 'x86_64'/'AMD64' or 'arm64'/'aarch64' but got {cpu}!")
    
    def eager_dynamic_quantize(self, model: nn.Module, model_type: str) -> nn.Module:
        """
        Method for eager dynamic quantize
        Params:
            model (nn.Module): model to be quantized
            model_type (str): type of model 'mlp' | 'lstm' | 'gru'
        Reurn:
            quantized_model (nn.Module): Dynamic quantized model
        """

        model.eval()

        # define layers to quantize
        if model_type == "mlp":
            qconfig = {nn.Linear}
        elif model_type == "lstm":
            qconfig = {nn.Linear, nn.LSTM}
        elif model_type == "gru":
            qconfig = {nn.Linear, nn.GRU}
        else:
            raise ValueError(f"model_type must be 'mlp', 'lstm' or 'gru'!")

        # dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig,       # only quantize safe ops
            dtype=torch.qint8
        )

        return quantized_model
    
    def static_quantization(self, model: nn.Module, calibration_loader: DataLoader) -> nn.Module:
        """
        Method for static quantization
        Params:
            model (nn.Module): model to be quantized
            calibration_loader (Dataloader): dataloader to calibrate model before quantization
        Return:
            quantized_model (nn.Module): Static quantized model
        """
        model.eval()

        qconfig = quant.get_default_qconfig(self.backend)

        x, _ = next(iter(calibration_loader))

        model_prepared = prepare_fx(
            model,
            {"": qconfig},
            x
        )

        # Calibration
        with torch.no_grad():
            for x, _ in calibration_loader:
                model_prepared(x)

        quantized_model = convert_fx(model_prepared)

        return quantized_model



