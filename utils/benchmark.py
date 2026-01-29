import os
from IPython.display import display
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tracemalloc
import psutil
from torch.profiler import profiler
from torch.profiler import profile, ProfilerActivity, record_function
from utils.metrics import evaluate_metrics
from sklearn.metrics import confusion_matrix
from utils.plot import plot_fpr_tpr_roc_auc, plot_confusion_matrix
from data.dataset import get_label_dict
from config import Config
conf = Config()


class Benchmark:
    def __init__(self, model, loader, batch_size, model_name, result_path, runs):
        self.model = model
        self.loader = loader
        self.batch_size = batch_size
        self.model_name = model_name
        self.result_path = result_path
        self.runs = runs
        self.results = {}
        self.results["Info"] = {
            "Name": model_name, 
            "Batch Size": batch_size,
            "Samples": runs,
            "Location": "DPU" if conf.location == "dpu" else "Host"
        }
        self.df = pd.DataFrame()
    
    def __call__(self, plot=False, plot_path=""):
        self.warmup()
        self.memory_usage()
        self.latency()
        self.throughput()
        self.cpu_usage()
        self.metrics(plot=plot, plot_path=plot_path)
    
    def load_model(self):
        self.model.load()
    
    def get_data_frame(self):
        rows = []
        for section, metrics in self.results.items():
            for metric, value in metrics.items():
                rows.append({"Section": section, "Metric": metric, "Value": value})
        self.df = pd.DataFrame(rows)

    def print_result(self):
        if self.df.empty:
            self.get_data_frame()
        display(self.df)

    def save(self):
        if self.df.empty:
            self.get_data_frame()
        self.df.to_csv(self.result_path, index=False)
    
    def warmup(self, warmup=10):
        self.model.model.eval()
        for i, (data, _) in enumerate(self.loader):
            with torch.no_grad():
                self.model.model(data)
            if i == warmup:
                break
        
        print("Warmup Done!")
    
    def latency(self):
        self.model.model.eval()

        times = []
        for data, _ in self.loader:
            with torch.no_grad():
                start = time.perf_counter()
                self.model.model(data)
                end = time.perf_counter()
                times.append(end - start)
        
        self.results["Latency"] = {
            "Total (ms)": f"{sum(times)*1000:.3f}",
            "Avg. (ms)": f"{sum(times)/len(times)*1000:.3f}",
            "Min (ms)": f"{min(times)*1000:.3f}",
            "Max (ms)": f"{max(times)*1000:.3f}"
        }

        print("Latency Benchmark Done!")

    def throughput(self, seconds=10):
        self.model.model.eval()

        count = 0
        start = time.perf_counter()

        while time.perf_counter() - start < seconds:
            data, _ = next(iter(self.loader))
            with torch.no_grad():
                self.model.model(data)
                count += self.batch_size
        
        elapsed = time.perf_counter() - start

        self.results["Throughput"] = {
            "Runtime (s)": f"{elapsed:.2f}",
            "Samples/s": f"{count / elapsed:.2f}"
        }

        print("Throughput Benchmark Done!")
    
    def cpu_usage(self):
        self.model.model.eval()

        process = psutil.Process()
        start_cpu = process.cpu_times()
        start_time = time.time()

        for data, _ in self.loader:
            with torch.no_grad():
                self.model.model(data)

        end_cpu = process.cpu_times()
        end_time = time.time()

        cpu_used = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
        elapsed = end_time - start_time

        self.results["CPU"] = {
            "Runtime (s)": f"{elapsed:.2f}",
            "Avg. (cores)": f"{cpu_used / elapsed:.2f}/{psutil.cpu_count()}"
        }

        print("CPU Benchmark Done!")

    def memory_usage(self):
        self.model.model.eval()

        mem_size = get_model_memory_size(self.model.model)

        with torch.profiler.profile(
                activities=[profiler.ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=False
            ) as prof:
                for i, (data, _) in enumerate(self.loader):
                    with torch.no_grad():
                        self.model.model(data)
                    if i + 1 >= 10:
                        break
        
        # peak memory during profiling
        memory_readings = [e.cpu_memory_usage for e in prof.key_averages()]
        avg_mem = (sum(memory_readings) / len(memory_readings)) / 1024**2 #MB
        peak_mem = max(memory_readings) / 1024**2 #MB

        self.results["Memory"] = {
            "Avg. (MB)": f"{avg_mem:.3f}",
            "Peak (MB)": f"{peak_mem:.3f}",
            "Model (MB)": f"{mem_size:.3f}"
        }

        print("Memory Benchmark Done!")

    def metrics(self, plot=False, plot_path=""):
        self.model.model.eval()
        y_true, y_logits = [], []

        for data, labels in self.loader:
            with torch.no_grad():
                logits = self.model.model(data)
            
            y_true.append(labels)
            y_logits.append(logits)

        y_true, y_logits = torch.cat(y_true, dim=0), torch.cat(y_logits, dim=0)

        y_probs = F.softmax(y_logits, dim=1)
        y_preds = torch.argmax(y_probs, dim=1).cpu()
        y_probs = y_probs.cpu()
        y_true = y_true.cpu()

        metrics = evaluate_metrics(y_true, y_preds, y_probs, num_classes=y_logits.size(1))

        self.results["Score"] = {
            "Macro-F1": f"{metrics['f1_macro']:.2f}",
            "Weighted-F1": f"{metrics['f1_weighted']:.2f}",
            "Micro-F1": f"{metrics['f1_micro']:.2f}",
            "Macro ROC AUC": f"{metrics['roc_auc_macro']:.2f}"
        }

        print("Metrics Benchmark Done!")

        if plot:
            roc_auc_path = os.path.join(plot_path, "roc_auc_" + self.model_name + ".png")
            cm_path = os.path.join(plot_path, "cm_" + self.model_name + ".png")
            
            cm = confusion_matrix(y_true, y_preds, normalize="true")
            label_dict = get_label_dict(conf.datasets)
            class_names = list(label_dict.keys())

            plot_confusion_matrix(cm, class_names, cm_path)
            plot_fpr_tpr_roc_auc(metrics, roc_auc_path)






class SplitBenchmark(Benchmark):
    def __init__(self, model, loader, batch_size, model_name, result_path, socket, split="dpu"):
        super().__init__(model, loader, batch_size, model_name, result_path)
        self.socket = socket
        self.split = split

    def open(self):
        self.socket.open()

    def close(self):
        self.socket.close()

    def send(self, features, labels, time=False):
        if self.split == "dpu":
            self.socket.send(features, t_time=time)
            self.socket.send(labels.to(dtype=torch.float))
            self.socket.wait()
        else:
            self.socket.signal()

    def receive(self, time=False):
        if self.split == "dpu":
            data, labels = next(iter(self.loader))
        else:
            data = self.socket.receive(t_time=time)
            labels = self.socket.receive().to(dtype=torch.long)
        
        return data, labels

    def latency(self, warmup=10, runs=100):
        self.model.model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                data, labels = self.receive()
                features = self.model.model(data)
                self.send(features, labels)

        times = []
        with torch.no_grad():
            for _ in range(runs):
                data, labels = self.receive()
                start = time.perf_counter()
                features = self.model.model(data)
                end = time.perf_counter()
                self.send(features, labels)
                times.append(end - start)
        
        self.results.append(f"Model inference latency (ms):")
        self.results.append(f"Avg.: {sum(times)/len(times)*1000:.3f}ms")
        self.results.append(f"Min: {min(times)*1000:.3f}ms")
        self.results.append(f"Max: {max(times)*1000:.3f}ms\n")

    def throughput(self, seconds=10):
        self.model.model.eval()

        count = 0
        start = time.perf_counter()

        with torch.no_grad():
            while time.perf_counter() - start < seconds:
                data, labels = self.receive()
                features = self.model.model(data)
                self.send(features, labels)
                count += data.size(0)

        elapsed = time.perf_counter() - start
        
        self.results.append(f"Model inference throughput (samples/sec):")
        self.results.append(f"Throughput: {count / elapsed:.2f} samples/sec\n")

    def cpu_usage(self, warmup=10, runs=50):
        self.model.model.eval()

        process = psutil.Process()

        with torch.no_grad():
            for _ in range(warmup):
                data, labels = self.receive()
                features = self.model.model(data)
                self.send(features, labels)
        
        start_cpu = process.cpu_times()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(runs):
                data, labels = self.receive()
                features = self.model.model(data)
                self.send(features, labels)

        end_cpu = process.cpu_times()
        end_time = time.time()

        cpu_used = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
        elapsed = end_time - start_time

        self.results.append(f"Model inference CPU core usage (number of logical cores):")
        self.results.append(f"runtime: {elapsed:.2f} seconds")
        self.results.append(f"Avg.: {cpu_used / elapsed:.2f}/{psutil.cpu_count()} cores\n")

    def memory_usage(self, warmup=10, runs=10):
        self.model.model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                data, labels = self.receive()
                features = self.model.model(data)
                self.send(features, labels)
        
        with torch.no_grad():
            with torch.profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                for _ in range(runs):
                    data, labels = self.receive()
                    features = self.model.model(data)
                    self.send(features, labels)

        # peak memory during profiling
        memory_readings = [e.cpu_memory_usage for e in prof.key_averages()]
        avg_mem = (sum(memory_readings) / len(memory_readings)) / 1024**2 #MB
        peak_mem = max(memory_readings) / 1024**2 #MB
        
        self.results.append(f"Memory usage (MB):")
        self.results.append(f"Avg.: {avg_mem:.3f}MB")
        self.results.append(f"Peak: {peak_mem:.3f}MB\n")


    def metrics(self, runs=100):
        self.model.model.eval()
        if self.split == "dpu":
            for _ in range(runs):
                data, labels = self.receive()
                with torch.no_grad():
                    features = self.model.model(data)
                self.send(features, labels)
        else:
            y_true, y_logits = [], []

            for _ in range(runs):
                data, labels = self.receive()
                with torch.no_grad():
                    logits = self.model.model(data)
                self.send(logits, labels)

                y_true.append(labels)
                y_logits.append(logits)
            
            y_true, y_logits = torch.cat(y_true, dim=0), torch.cat(y_logits, dim=0)
            
            y_probs = F.softmax(y_logits, dim=1)
            y_preds = torch.argmax(y_probs, dim=1).cpu()
            y_probs = y_probs.cpu()
            y_true = y_true.cpu()

            metrics = evaluate_metrics(y_true, y_preds, y_probs, num_classes=y_logits.size(1))
            #acc = (pred.argmax(dim=1) == labels).float().mean()

            self.results.append(f"Model Macro-F1, Micro-F1 and Macro ROC AUC scores:")
            self.results.append(f"Macro-F1: {metrics['f1_macro']:.2f}")
            self.results.append(f"Micro-F1: {metrics['f1_micro']:.2f}")
            self.results.append(f"Macro ROC AUC: {metrics['roc_auc_macro']:.2f}\n\n\n")

    def transfer_time(self, runs=100):
        self.model.model.eval()

        if self.split == "dpu":
            with torch.no_grad():
                for _ in range(runs):
                    data, labels = self.receive(time=True)
                    features = self.model.model(data)
                    self.send(features, labels, time=True)
        else:
            times = []
            with torch.no_grad():
                for _ in range(runs):
                    (data, time), labels = self.receive(time=True)
                    features = self.model.model(data)
                    self.send(features, labels, time=True)
                    times.append(time)
            
            self.results.append(f"Split Model transfer time (ms):")
            self.results.append(f"Avg.: {sum(times)/len(times)*1000:.3f}ms")
            self.results.append(f"Min: {min(times)*1000:.3f}ms")
            self.results.append(f"Max: {max(times)*1000:.3f}ms\n")
                

    


def get_model_memory_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # Total size in bytes
    total_size_bytes = param_size + buffer_size
    
    # Convert to human-readable format (MB)
    size_all_mb = total_size_bytes / (1024**2)
    
    return size_all_mb