import os
import sys
sys.path.append(os.path.join(os.getcwd().replace("benchmark", "")))

import pandas as pd
import re
from config import Config
conf = Config()


def parse_value(value):
    """
    Convert values to float when possible.
    Handles cases like '5.55/8' by extracting the first number.
    Returns None if conversion is not possible.
    """
    if pd.isna(value):
        return None

    # Extract first numeric value (handles '5.55/8', '71.93', etc.)
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if match:
        return float(match.group())
    return None


def compare_models(baselines, compress, base_path, comp_path, output_csv):
    merge_list = []
    for base, comp in zip(baselines, compress):
        baseline_csv = os.path.join(base_path, base + ".csv")
        compressed_csv = os.path.join(comp_path, comp + ".csv")

        # Load CSVs
        baseline = pd.read_csv(baseline_csv)
        compressed = pd.read_csv(compressed_csv)

        # Parse numeric values
        baseline["NumericValue"] = baseline["Value"].apply(parse_value)
        compressed["NumericValue"] = compressed["Value"].apply(parse_value)

        # Merge on Section + Metric
        merged = pd.merge(
            baseline,
            compressed,
            on=["Section", "Metric"],
            suffixes=("_Baseline", "_Compressed")
        )

        # Compute numeric differences
        merged["Abs. Diff"] = (
            merged["NumericValue_Compressed"] - merged["NumericValue_Baseline"]
        )

        merged["Change (%)"] = (
            merged["Abs. Diff"] / merged["NumericValue_Baseline"] * 100
        )

        # Format precision
        merged["Change (%)"] = merged["Change (%)"].round(2)
        merged["Change (%)"] = merged["Change (%)"].map("({:.2f}%)".format)

        # Select final columns
        result = merged[[
            "Section",
            "Metric",
            "Value_Compressed",
            "Change (%)"
        ]]
        merge_list.append(result)

    result = merge_list[0][[
        "Section",
        "Metric"
    ]]

    for i in range(len(merge_list)):
        result[compress[i] + "_Compressed"] = merge_list[i]["Value_Compressed"]
        result[compress[i] + " Change (%)"] = merge_list[i]["Change (%)"]
    
    # Save output
    result.to_csv(output_csv, index=False)
    print(f"Comparison saved to: {output_csv}")


if __name__ == "__main__":
    #names = [
    #    ("mlp_4", "result_light_mlp_1"), ("mlp_4", "light_mlp_1"), ("mlp_4", "light_mlp_4"),
    #    ("lstm_4", "result_light_lstm_1"), ("lstm_4", "light_lstm_1"), ("lstm_4", "light_lstm_4"),
    #    ("gru_4", "result_light_gru_1"), ("gru_4", "light_gru_1"), ("gru_4", "light_gru_4"),
    #    ("cnn_4", "result_light_cnn_1"), ("cnn_4", "light_cnn_1"), ("cnn_4", "light_cnn_4")
    #]
    names = [
        ("mlp_4", "mlp_4"),
        ("lstm_4", "lstm_4"),
        ("gru_4", "gru_4"),
        ("cnn_4", "cnn_4")
    ]
    #for (base, comp) in names:
    params = {
        "quant": True,
        "prune": False,
        "model": "mlp_4",
        "comp_model": "result_light_mlp_1",
        "full": "large_model",
        "pq": "pruned_quantized_model",
        "comp": "compressed_model",
        "split": "split_model"
    }
    type_ = params["pq"]

    named_comp = ""

    # prune model
    if params["prune"]:
        named_comp += "pruned_"

    # quantize model
    if params["quant"]:
        named_comp += "quant_"

    base_path = os.path.join(conf.benchmark_dpu, params["full"])
    comp_path = os.path.join(conf.benchmark_dpu, type_)
    out_csv = os.path.join(conf.benchmark_comp, type_, "dpu_" + named_comp + "models" + ".csv")

    compare_models(
        baselines=["mlp_4", "lstm_4", "gru_4", "cnn_4"],
        compress=[named_comp + "mlp_4", named_comp + "lstm_4", named_comp + "gru_4", named_comp + "cnn_4"],
        base_path=base_path,
        comp_path=comp_path,
        output_csv=out_csv
    )
