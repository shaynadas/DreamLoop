import subprocess
import argparse
import sys
import os

def run_command(command, step_name):
    print("\n" + "="*60)
    print(f"🚀 STARTING STEP: {step_name}")
    print(f"💻 Command: {' '.join(command)}")
    print("="*60)
    
    try:
        # Runs the command and pipes the output to the terminal in real-time
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FATAL ERROR IN STEP: {step_name}")
        print(f"Pipeline crashed. Exiting...")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="DreamLoop Master Pipeline Orchestrator")
    parser.add_argument("--scenario", default="pedestrian_cross", choices=["pedestrian_cross", "collision", "jaywalker", "bike"])
    parser.add_argument("--frames_dir", default="./data/frames", help="Path to original JSON frames")
    parser.add_argument("--baseline_video", default="./data/baseline.mp4", help="Path to original clear video")
    args = parser.parse_args()

    # --- FOLDER SETUP ---
    shared_data_dir = "./shared_data"
    dream_output_dir = "./outputs/dream_run"
    yolo_labels_dir = "./data/injected_labels"
    dataset_dir = "./dreamloop_dataset"
    
    os.makedirs(shared_data_dir, exist_ok=True)
    os.makedirs(dream_output_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # --- STEP 1: Person A's 3D Math Injection ---
    run_command([
        "python3", "inject.py",
        "--data_dir", args.frames_dir,
        "--output_dir", shared_data_dir,
        "--scenario", args.scenario
    ], "1. 3D Hazard Injection & Wireframe Generation")

    # --- STEP 2: Person B's Generative AI World Model ---
    # We dynamically inject the scenario name into the prompt
    prompt = f"Dashcam footage. Severe winter blizzard, heavy snow, poor visibility, a {args.scenario.replace('_', ' ')} in the road."
    run_command([
        "python3", "gen.py",
        "--scenario", args.scenario,
        "--intensity", "10",
        "--time_of_day", "night",
        "--input_video", args.baseline_video,
        "--output_dir", dream_output_dir,
        "--prompt", prompt
    ], "2. Helios V2V Generative Hallucination")

    # --- STEP 3: The Integration Wire (JSON to YOLO) ---
    # inject.py saves boxes.json in the output_dir
    boxes_json_path = os.path.join(shared_data_dir, "boxes.json")
    run_command([
        "python3", "json_to_yolo.py",
        "--json_file", boxes_json_path,
        "--output_dir", yolo_labels_dir
    ], "3. Converting 3D Coordinates to 2D YOLO Labels")

    # --- STEP 4: The Data Plumber ---
    # Assuming gen.py outputs a video named after the scenario
    dream_video_path = os.path.join(dream_output_dir, f"{args.scenario}.mp4")
    run_command([
        "python3", "prepare_yolo_data.py",
        "--dream_video", dream_video_path,
        "--baseline_labels", yolo_labels_dir,
        "--output_dir", dataset_dir
    ], "4. Formatting Synthetic Dataset")

    # --- STEP 5: The Brain Training ---
    run_command([
        "python3", "train_yolo.py",
        "--epochs", "30"
    ], "5. Fine-Tuning YOLO on Synthetic Dream Data")

    print("\n" + "🌟"*30)
    print("✅ DREAMLOOP PIPELINE COMPLETE!")
    print("🌟"*30)
    print(f"1. Your trained model is in the runs/train/ folder.")
    print(f"2. Your synthetic video is at: {dream_video_path}")
    print("Plug them into demo.py and go win this hackathon.")

if __name__ == "__main__":
    main()