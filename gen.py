import argparse
import os
import subprocess

def build_prompt(scenario, intensity, time_of_day):
    """
    This generates the massive, highly-specific text prompt needed for Helios.
    """
    base_prompt = (
        "Dashcam footage from a moving vehicle. The road layout and traffic "
        "remain physically consistent with the original video. "
    )
    
    # Weather Logic
    if scenario == "blizzard":
        weather = f"Severe winter blizzard, {intensity}/10 intensity, heavy snow accumulation on the windshield, thick falling snow obscuring the road ahead. "
    elif scenario == "rain":
        weather = f"Torrential downpour, {intensity}/10 intensity, massive puddles on the asphalt, water splashing, heavy rain hitting the windshield. "
    else:
        weather = "Clear weather. "

    # Lighting Logic
    lighting = f"Time of day is {time_of_day}, poor visibility, highly realistic cinematic lighting, glaring headlights from oncoming traffic."
    
    return base_prompt + weather + lighting

def main():
    parser = argparse.ArgumentParser(description="DreamLoop Helios V2V Generator")
    
    # Inputs that you (Person B) will dictate
    parser.add_argument("--scenario", type=str, choices=["blizzard", "rain", "clear"], required=True, help="The weather edge-case")
    parser.add_argument("--intensity", type=int, default=8, help="Scale of 1-10")
    parser.add_argument("--time_of_day", type=str, default="night", choices=["day", "night", "dusk"])
    
    # File routing
    parser.add_argument("--input_video", type=str, required=True, help="Path to the baseline clean video")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the dream")
    
    # THE LIFESAVER FLAG
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running the GPU models")

    args = parser.parse_args()

    # 1. Generate the prompt
    final_prompt = build_prompt(args.scenario, args.intensity, args.time_of_day)
    
    # 2. Ensure output directory exists (prevents crashes at the end of a 10 min render)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Construct the massive Helios command (based on Helios-Distilled V2V docs)
    # We use a multi-line string for readability
    helios_command = f"""python infer_helios.py \\
    --base_model_path "BestWishYsh/Helios-Distilled" \\
    --transformer_path "BestWishYsh/Helios-Distilled" \\
    --sample_type "v2v" \\
    --video_path "{args.input_video}" \\
    --prompt "{final_prompt}" \\
    --num_frames 240 \\
    --guidance_scale 1.0 \\
    --is_enable_stage2 \\
    --pyramid_num_inference_steps_list 2 2 2 \\
    --is_amplify_first_chunk \\
    --output_folder "{args.output_dir}" \\
    --enable_low_vram_mode \\
    --group_offloading_type "leaf_level"
    """

    print("\n" + "="*50)
    print("🚀 DREAMLOOP AGENT: JOB QUEUED")
    print("="*50)
    print(f"Scenario:  {args.scenario.upper()} (Intensity: {args.intensity}/10)")
    print(f"Prompt:    {final_prompt}")
    print(f"Output to: {args.output_dir}")
    print("="*50 + "\n")

    if args.dry_run:
        print("[DRY RUN MODE] - The following command would be executed on the DGX:\n")
        print(helios_command)
    else:
        print("[EXECUTE MODE] - Firing up the DGX...\n")
        # Clean up the string formatting before running
        clean_command = " ".join(helios_command.split())
        subprocess.run(clean_command, shell=True)

if __name__ == "__main__":
    main()