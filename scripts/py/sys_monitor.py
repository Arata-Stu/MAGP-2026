#!/usr/bin/env python3
import psutil, os, time, subprocess

use_tegrastats = False
if os.path.exists('/usr/bin/tegrastats'):
    use_tegrastats = True

def get_gpu_usage_jetson():
    try:
        out = subprocess.check_output(['tegrastats', '--interval', '500'], text=True, timeout=1)
        last_line = out.strip().split('\n')[-1]
        if "GR3D_FREQ" in last_line:
            gpu_part = [s for s in last_line.split() if "GR3D_FREQ" in s]
            if gpu_part: return f"GPU: {gpu_part[0].split('@')[0]}"
        elif "GPU" in last_line:
             gpu_part = [s for s in last_line.split() if "GPU" in s]
             if gpu_part: return f"{gpu_part[0].split('@')[0]}"
        return "GPU: N/A (tegrastats error)"
    except Exception:
        return "GPU: N/A"

def main():
    while True:
        try:
            print("\033[2J\033[H", end='') # 画面クリア
            print("=== System Monitor ===\n")
            
            print(f"CPU Usage: {psutil.cpu_percent():5.1f}%")
            core_usages = psutil.cpu_percent(percpu=True)
            core_strs = [f"{u:5.1f}%" for u in core_usages]
            print("Cores: " + " | ".join(core_strs))
            
            mem = psutil.virtual_memory()
            print(f"\nMemory: {mem.percent:5.1f}% ({mem.used//(1024**2):,} / {mem.total//(1024**2):,} MB)")
            
            swap = psutil.swap_memory()
            print(f"Swap:    {swap.percent:5.1f}% ({swap.used//(1024**2):,} / {swap.total//(1024**2):,} MB)")
            
            if use_tegrastats:
                print(f"Jetson:  {get_gpu_usage_jetson()}")
            
            print("\n(tmuxペインを閉じるには Ctrl+B -> X)")
            
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nSysMonitor stopped.")
            break 
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

if __name__ == '__main__':
    main()