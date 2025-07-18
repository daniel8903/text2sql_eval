from pynvml import *
import datetime
import csv
import os
import time

LOG_FILE = "gpu_metrics_log.csv"
LOG_INTERVAL_SECONDS = 2  # Logging alle 5 Sekunden

def bytes_to_mb(b):
    return round(b / 1024 / 1024, 2)

def init_csv():
    # Schreibe Header, falls Datei noch nicht existiert
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "gpu_index",
                "gpu_name",
                "temperature_c",
                "power_w",
                "power_limit_w",
                "gpu_util_percent",
                "vram_util_percent",
                "vram_used_mb",
                "vram_total_mb",
                "fan_speed_percent",
                "clock_gpu_mhz",
                "clock_mem_mhz",
                "perf_state",
                "active_processes"
            ])

def log_gpu_data():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
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

            # Anzahl aktiver Prozesse
            try:
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                proc_count = len(procs)
            except NVMLError:
                proc_count = -1

            writer.writerow([
                timestamp,
                i,
                name,
                temp,
                power,
                power_limit,
                util.gpu,
                util.memory,
                bytes_to_mb(mem.used),
                bytes_to_mb(mem.total),
                fan_speed,
                clock_gpu,
                clock_mem,
                perf_state,
                proc_count
            ])

    nvmlShutdown()

def main():
    init_csv()
    print(f"🚀 Starte Logging nach '{LOG_FILE}' alle {LOG_INTERVAL_SECONDS} Sekunden (Strg+C zum Abbrechen)")
    try:
        while True:
            log_gpu_data()
            time.sleep(LOG_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n🛑 Logging beendet.")

if __name__ == "__main__":
    main()
