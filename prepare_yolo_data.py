import cv2
import os
import shutil
import argparse

def setup_yolo_folders(base_dir="yolo_dataset"):
    """Creates the strict folder structure YOLO requires for training."""
    dirs = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base_dir

def main():
    parser = argparse.ArgumentParser(description="Extracts frames and syncs labels for YOLO training.")
    parser.add_argument("--dream_video", type=str, required=True, help="Path to the Helios output video (e.g., blizzard.mp4)")
    parser.add_argument("--baseline_labels", type=str, required=True, help="Folder containing the original .txt YOLO labels")
    parser.add_argument("--output_dir", type=str, default="dreamloop_dataset", help="Where to save the YOLO dataset")
    args = parser.parse_args()

    print("="*50)
    print("🛠️  DREAMLOOP: DATASET FORMATTER")
    print("="*50)

    # 1. Setup Folders
    dataset_dir = setup_yolo_folders(args.output_dir)
    train_img_dir = os.path.join(dataset_dir, "images/train")
    train_lbl_dir = os.path.join(dataset_dir, "labels/train")

    # 2. Open the Helios Video
    cap = cv2.VideoCapture(args.dream_video)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {args.dream_video}")
        return

    frame_count = 0
    saved_count = 0
    missing_labels = 0

    print("Extracting frames and mapping labels... this might take a minute.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # YOLO expects filenames like: frame_0000.jpg, frame_0001.jpg
        base_filename = f"frame_{frame_count:04d}"
        img_filename = f"{base_filename}.jpg"
        txt_filename = f"{base_filename}.txt"

        # Check if we have the baseline label for this specific frame
        baseline_txt_path = os.path.join(args.baseline_labels, txt_filename)
        
        if os.path.exists(baseline_txt_path):
            # 1. Save the new snowy frame
            cv2.imwrite(os.path.join(train_img_dir, img_filename), frame)
            
            # 2. Copy the baseline label to the new dataset folder
            shutil.copy(baseline_txt_path, os.path.join(train_lbl_dir, txt_filename))
            saved_count += 1
        else:
            # If there's no label for this frame in the baseline, we skip it
            missing_labels += 1

        frame_count += 1

    cap.release()

    print("\n✅ DATASET GENERATION COMPLETE")
    print(f"Total Video Frames Processed: {frame_count}")
    print(f"Successfully paired frames+labels: {saved_count}")
    if missing_labels > 0:
        print(f"Skipped {missing_labels} frames (no matching baseline label found)")
    print(f"\nYour dataset is ready at: ./{dataset_dir}/")
    print("="*50)

if __name__ == "__main__":
    main()