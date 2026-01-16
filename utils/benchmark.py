import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
from torch.profiler import profiler
from utils.metrics import evaluate_metrics


class Benchmark:
    def __init__(self, model, loader, batch_size, model_name, result_path):
        self.model = model
        self.loader = loader
        self.batch_size = batch_size
        self.model_name = model_name
        self.result_path = result_path
        self.results = [f"Benchmark - {model_name} model:\n"]
    
    def __call__(self):
        self.memory_usage()
        self.latency()
        self.throughput()
        self.cpu_usage()
        self.metrics()
    
    def load_model(self):
        self.model.load()

    def print_result(self):
        for line in self.results:
            print(line)

    def save(self):
        with open(self.result_path, "w") as f:
            for line in self.results:
                f.writelines(line + "\n")

    def latency(self, warmup=10, runs=100):
        self.model.model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                data, labels = next(iter(self.loader))
                _ = self.model.model(data)

        times = []
        with torch.no_grad():
            for _ in range(runs):
                data, labels = next(iter(self.loader))
                start = time.perf_counter()
                _ = self.model.model(data)
                end = time.perf_counter()
                times.append(end - start)
        
        self.results.append(f"Model inference latency on one batch (batch size = {self.batch_size}):")
        self.results.append(f"Avg latency: {sum(times)/len(times)*1000:.3f}ms")
        self.results.append(f"Min latency: {min(times)*1000:.3f}ms")
        self.results.append(f"Max latency: {max(times)*1000:.3f}ms\n")

    def throughput(self, seconds=10):
        self.model.model.eval()

        count = 0
        start = time.perf_counter()

        with torch.no_grad():
            while time.perf_counter() - start < seconds:
                data, labels = next(iter(self.loader))
                _ = self.model.model(data)
                count += data.size(0)

        elapsed = time.perf_counter() - start
        
        self.results.append(f"Model inference throughput (batch size = {self.batch_size}):")
        self.results.append(f"Throughput: {count / elapsed:.2f} samples/sec\n")

    def cpu_usage(self, warmup=10, runs=50):
        self.model.model.eval()

        process = psutil.Process()

        with torch.no_grad():
            for _ in range(warmup):
                data, labels = next(iter(self.loader))
                _ = self.model.model(data)

        start_cpu = process.cpu_times()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(runs):
                data, labels = next(iter(self.loader))
                _ = self.model.model(data)

        end_cpu = process.cpu_times()
        end_time = time.time()

        cpu_used = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
        elapsed = end_time - start_time

        self.results.append(f"Model inference CPU usage (number of logical cores) during runtime:")
        self.results.append(f"CPU runtime: {elapsed:.2f} seconds")
        self.results.append(f"Average CPU usage: {cpu_used / elapsed:.2f}/{psutil.cpu_count()} cores\n")

    def memory_usage(self, warmup=10, runs=10):
        self.model.model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                data, labels = next(iter(self.loader))
                _ = self.model.model(data)
        
        with torch.no_grad():
            with torch.profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                for _ in range(runs):
                    data, labels = next(iter(self.loader))
                    _ = self.model.model(data)

        # peak memory during profiling
        memory_readings = [e.cpu_memory_usage for e in prof.key_averages()]
        avg_mem = (sum(memory_readings) / len(memory_readings)) / 1024**2 #MB
        peak_mem = max(memory_readings) / 1024**2 #MB
        
        self.results.append(f"Memory usage (MB):")
        self.results.append(f"Avg memory usage: {avg_mem:.3f}MB")
        self.results.append(f"Peak memory usage: {peak_mem:.3f}MB\n")


    def metrics(self, runs=100):
        self.model.model.eval()
        y_true, y_logits = [], []

        for _ in range(runs):
            data, labels = next(iter(self.loader))
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

        self.results.append(f"Model ({self.model_name}) Macro-F1, Micro-F1 and Macro ROC AUC scores:")
        self.results.append(f"Macro-F1 score: {metrics['f1_macro']:.2f}")
        self.results.append(f"Micro-F1 score: {metrics['f1_micro']:.2f}")
        self.results.append(f"Macro ROC AUC score: {metrics['roc_auc_macro']:.2f}\n\n\n")








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
        
        self.results.append(f"Model inference latency on one batch (batch size = {self.batch_size}):")
        self.results.append(f"Avg latency: {sum(times)/len(times)*1000:.3f}ms")
        self.results.append(f"Min latency: {min(times)*1000:.3f}ms")
        self.results.append(f"Max latency: {max(times)*1000:.3f}ms\n")

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
        
        self.results.append(f"Model inference throughput (batch size = {self.batch_size}):")
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

        self.results.append(f"Model inference CPU usage (number of logical cores) during runtime:")
        self.results.append(f"CPU runtime: {elapsed:.2f} seconds")
        self.results.append(f"Average CPU usage: {cpu_used / elapsed:.2f}/{psutil.cpu_count()} cores\n")

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
        self.results.append(f"Avg memory usage: {avg_mem:.3f}MB")
        self.results.append(f"Peak memory usage: {peak_mem:.3f}MB\n")


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

            self.results.append(f"Model ({self.model_name}) Macro-F1, Micro-F1 and Macro ROC AUC scores:")
            self.results.append(f"Macro-F1 score: {metrics['f1_macro']:.2f}")
            self.results.append(f"Micro-F1 score: {metrics['f1_micro']:.2f}")
            self.results.append(f"Macro ROC AUC score: {metrics['roc_auc_macro']:.2f}\n")

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
            
            self.results.append(f"Split Model transfer time from dpu to host (batch size = {self.batch_size}):")
            self.results.append(f"Avg transfer time: {sum(times)/len(times)*1000:.3f}ms")
            self.results.append(f"Min transfer time: {min(times)*1000:.3f}ms")
            self.results.append(f"Max transfer time: {max(times)*1000:.3f}ms\n")
                

    
