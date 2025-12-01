import torch
import time
import psutil
from torch.profiler import profiler


class Benchmark:
    def __init__(self, model, model_path, loader, batch_size, model_name):
        self.model = model
        self.model.load(model_path)
        self.loader = loader
        self.batch_size = batch_size
        self.model_name = model_name
        self.results = [f"Benchmark - {model_name} model:\n"]

    def latency(self, warmup=10, runs=100):
        self.model.model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                data, labels = next(iter(self.loader))
                pred = self.model.model(data)

        times = []
        with torch.no_grad():
            for _ in range(runs):
                data, labels = next(iter(self.loader))
                start = time.perf_counter()
                pred = self.model.model(data)
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
                pred = self.model.model(data)
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
                pred = self.model.model(data)

        start_cpu = process.cpu_times()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(runs):
                data, labels = next(iter(self.loader))
                pred = self.model.model(data)

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
                pred = self.model.model(data)
        
        with torch.no_grad():
            with torch.profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                for _ in range(runs):
                    data, labels = next(iter(self.loader))
                    pred = self.model.model(data)

        # peak memory during profiling
        memory_readings = [e.cpu_memory_usage for e in prof.key_averages()]
        avg_mem = (sum(memory_readings) / len(memory_readings)) / 1024**2 #MB
        peak_mem = max(memory_readings) / 1024**2 #MB
        
        self.results.append(f"Memory usage (MB):")
        self.results.append(f"Avg memory usage: {avg_mem:.3f}MB")
        self.results.append(f"Peak memory usage: {peak_mem:.3f}MB\n")


    def accuracy(self, runs=100):
        self.model.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for _ in range(runs):
                data, labels = next(iter(self.loader))
                _, pred = self.model.model(data)

                y_true.append(labels)
                y_pred.append(pred)
        
        y_true, y_pred = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)
        
        acc = (pred.argmax(dim=1) == labels).float().mean()

        self.results.append(f"Model ({self.model_name}) inference accuracy (%):")
        self.results.append(f"Accuracy: {acc*100:.2f}%\n\n\n")


