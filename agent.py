import time
import os
import subprocess
import random

# ANSI Colors for a sick terminal UI
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header():
    os.system('clear' if os.name == 'posix' else 'cls')
    print(Colors.CYAN + Colors.BOLD + "="*60)
    print(" 🧠 DREAMLOOP OMNI-AGENT CONSOLE v1.0")
    print("="*60 + Colors.RESET)

def simulate_thinking(text, delay=1.5):
    print(Colors.YELLOW + f"[{time.strftime('%H:%M:%S')}] ⚙️  {text}..." + Colors.RESET)
    time.sleep(delay)

def run_agent_loop():
    print_header()
    
    # ---------------------------------------------------------
    # CYCLE 1: STANDARD OBSERVATION (Daytime Driving)
    # ---------------------------------------------------------
    print(Colors.BOLD + "\n--- CYCLE 1: ACTIVE MODE ---" + Colors.RESET)
    simulate_thinking("Observing environment streams (Camera 1: Forward)", 1)
    simulate_thinking("Running baseline YOLOv8 perception model", 2)
    print(Colors.GREEN + f"[{time.strftime('%H:%M:%S')}] ✅ STATUS: CLEAR. Confidence: 94%. No hazards detected." + Colors.RESET)
    time.sleep(1)

    # ---------------------------------------------------------
    # CYCLE 2: IDLE STATE DETECTED
    # ---------------------------------------------------------
    print(Colors.BOLD + "\n--- CYCLE 2: IDLE DETECTED ---" + Colors.RESET)
    simulate_thinking("Vehicle parked. CPU/GPU utilization drops below 10%", 1.5)
    print(Colors.MAGENTA + Colors.BOLD + f"[{time.strftime('%H:%M:%S')}] 🌙 TRIGGERING DREAM SEQUENCE" + Colors.RESET)
    simulate_thinking("Scanning memory bank for standard driving logs", 1)
    print(Colors.CYAN + f"[{time.strftime('%H:%M:%S')}] 📂 Retrieved: baseline.mp4 (Location: Highway 101, Sunny)" + Colors.RESET)
    time.sleep(1)

    # ---------------------------------------------------------
    # CYCLE 3: THE DREAM (Calling your gen.py script!)
    # ---------------------------------------------------------
    print(Colors.BOLD + "\n--- CYCLE 3: SYNTHETIC HALLUCINATION ---" + Colors.RESET)
    simulate_thinking("Agent Planner routing request to Cosmos/Helios World Model", 2)
    
    edge_cases = ["blizzard", "rain"]
    chosen_scenario = random.choice(edge_cases)
    
    print(Colors.MAGENTA + f"[{time.strftime('%H:%M:%S')}] 💉 Injecting Edge Case: {chosen_scenario.upper()} (Intensity: 10/10)" + Colors.RESET)
    
    # Actually call the gen.py script we wrote earlier!
    # We use --dry-run here so it doesn't crash your laptop during the live demo
    command = f"python3 gen.py --scenario {chosen_scenario} --intensity 10 --time_of_day night --input_video ./data/baseline.mp4 --output_dir ./outputs/dream_run --dry-run"
    
    print(Colors.CYAN + "\n[Executing Sub-Process: Generative World Model]..." + Colors.RESET)
    time.sleep(1)
    
    # Run the generator
    subprocess.run(command, shell=True)
    
    # ---------------------------------------------------------
    # CYCLE 4: CONSOLIDATION (Training)
    # ---------------------------------------------------------
    print(Colors.BOLD + "\n--- CYCLE 4: MEMORY CONSOLIDATION ---" + Colors.RESET)
    simulate_thinking("Video generation complete. Extracting frames and mapping labels", 2)
    simulate_thinking("Fine-tuning internal perception weights on synthetic edge case", 2.5)
    print(Colors.GREEN + Colors.BOLD + f"[{time.strftime('%H:%M:%S')}] 🧠 UPGRADE COMPLETE: Agent is now robust to {chosen_scenario.upper()} conditions." + Colors.RESET)
    
    print(Colors.CYAN + Colors.BOLD + "\n" + "="*60)
    print(" ✨ DREAMLOOP CYCLE FINISHED. RETURNING TO STANDBY.")
    print("="*60 + Colors.RESET + "\n")

if __name__ == "__main__":
    run_agent_loop()