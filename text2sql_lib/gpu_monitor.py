import threading
import datetime
from pynvml import *

def bytes_to_mb(b):
    return round(b / 1024 / 1024, 2)

def log_gpu_metrics():
    """Return a list of dicts with current GPU stats for all GPUs."""
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    gpu_stats = []

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        try:
            name = nvmlDeviceGetName(handle).decode()
            mem = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            power = nvmlDeviceGetPowerUsage(handle) / 1000
            power_limit = nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
            fan_speed = nvmlDeviceGetFanSpeed(handle)
            clock_gpu = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
            clock_mem = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)
            perf_state = nvmlDeviceGetPerformanceState(handle)
            try:
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                proc_count = len(procs)
            except NVMLError:
                proc_count = -1

            gpu_stats.append({
                "timestamp": timestamp,
                "gpu_index": i,
                "gpu_name": name,
                "temperature_c": temp,
                "power_w": power,
                "power_limit_w": power_limit,
                "gpu_util_percent": util.gpu,
                "vram_util_percent": util.memory,
                "vram_used_mb": bytes_to_mb(mem.used),
                "vram_total_mb": bytes_to_mb(mem.total),
                "fan_speed_percent": fan_speed,
                "clock_gpu_mhz": clock_gpu,
                "clock_mem_mhz": clock_mem,
                "perf_state": perf_state,
                "active_processes": proc_count,
            })
        except Exception as e:
            gpu_stats.append({
                "timestamp": timestamp,
                "gpu_index": i,
                "error": str(e)
            })
    nvmlShutdown()
    return gpu_stats

class GPUMonitorThread(threading.Thread):
    def __init__(self, interval_sec=0.5):
        super().__init__()
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.samples = []

    def run(self):
        import time
        while not self.stop_event.is_set():
            metrics = log_gpu_metrics()
            self.samples.extend(metrics)
            time.sleep(self.interval_sec)

    def stop(self):
        self.stop_event.set()