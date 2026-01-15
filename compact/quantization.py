import torch
import torch.nn as nn
import platform
from config import Config
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import torch.ao.quantization as quant
conf = Config()

def dynamic_quantize(model: nn.Module) -> nn.Module:
    """
    Method for eager dynamic quantize for 
    Params:
        model (nn.Module): model to be quantized
        model_type (str): type of model 'mlp' | 'rnn' | 'cnn'
    Reurn:
        quantized_rnn (nn.Module): Dynamic quantized model
    """
    model.eval()
    quantized_rnn = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.GRU, nn.Linear},       # only quantize safe ops
        dtype=torch.qint8
    )

    return quantized_rnn


def quantization_fx(
    model: nn.Module,
    calibration_loader: DataLoader,
    model_type: str
) -> nn.Module:

    model.eval()
    model.cpu()

    if model_type == "rnn":
        # nn.LSTM and nn.GRU do not support static quantization
        # dynamic quantization on rnn layers
        model.rnn = dynamic_quantize(model.rnn)

    backend = "fbgemm"
    torch.backends.quantized.engine = backend

    qconfig = quant.get_default_qconfig(backend)

    qconfig_mapping = (
        quant.QConfigMapping()
            .set_global(qconfig)
            .set_object_type(nn.Linear, qconfig)
            .set_object_type(nn.Conv1d, qconfig)
            .set_object_type(nn.ReLU, qconfig)
            .set_object_type(nn.LeakyReLU, qconfig)
            .set_object_type(nn.MaxPool1d, qconfig)
            .set_object_type(nn.LayerNorm, None)
    )

    # Example input
    example_inputs, _ = next(iter(calibration_loader))
    example_inputs = (example_inputs,)

    # Prepare
    prepared_model = prepare_fx(
        model,
        qconfig_mapping,
        example_inputs
    )

    # Calibration
    with torch.no_grad():
        for x, _ in calibration_loader:
            prepared_model(x)

    # Convert
    quantized_model = convert_fx(prepared_model)

    return quantized_model
