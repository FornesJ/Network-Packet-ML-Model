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

    def cpu_usage(self, duration=5):
        self.model.model.eval()

        process = psutil.Process()
        cpu_readings = []
        thread_readings = []

        end_time = time.time() + duration
        with torch.no_grad():
            while time.time() < end_time:
                data, labels = next(iter(self.loader))
                pred = self.model.model(data)

                cpu_percent = process.cpu_percent(interval=0.0)
                num_threads = process.num_threads()
                cpu_readings.append(cpu_percent / num_threads)
                thread_readings.append(num_threads)
        
        self.results.append(f"CPU usage per thread:")
        self.results.append(f"Avg CPU usage: {sum(cpu_readings)/len(cpu_readings):.1f}%")
        self.results.append(f"Max CPU usage: {max(cpu_readings):.1f}%")
        self.results.append(f"Number of threads used: {sum(thread_readings)/len(thread_readings)}\n")

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
                pred = self.model.model(data)

                y_true.append(labels)
                y_pred.append(pred)
        
        y_true, y_pred = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)
        
        acc = (pred.argmax(dim=1) == labels).float().mean()

        self.results.append(f"Model ({self.model_name}) inference accuracy (%):")
        self.results.append(f"Accuracy: {acc*100:.2f}%\n\n\n")
