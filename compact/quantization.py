import torch
import torch.nn as nn
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from config import Config
conf = Config()

class Quantize_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer = X86InductorQuantizer()
        self.quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    def calibrate(model, loader):
        torch.ao.quantization.move_exported_model_to_eval(model)
        with torch.no_grad():
            for (data, labels) in loader:
                if not data.is_cuda or not labels.is_cuda:
                    data, labels = data.to(conf.device), labels.to(conf.device)
                model(data)
    
    def forward(self, model, calibration_loader):
        prepared_model = prepare_pt2e(model, self.quantizer) # prepare model by folding batch norm into rnn/linear layers

        self.calibrate(prepared_model, calibration_loader) # calibration data to collect activation stats needed for activation quantization

        quantized_model = convert_pt2e(prepared_model) # converts calibrated model to a quantized model

        return quantized_model

