import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="DreamLoop Integration Wire: JSON to YOLO")
    parser.add_argument("--json_file", type=str, required=True, help="Path to boxes_pedestrian.json")
    parser.add_argument("--output_dir", type=str, default="./data/baseline_labels", help="Where to save YOLO txt files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Open teammate's JSON file
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    print("="*50)
    print("🔌 DREAMLOOP WIRE: Converting 3D JSON to 2D YOLO Labels")
    print("="*50)

    # Convert their JSON structure into YOLO .txt format
    count = 0
    for frame_data in data:
        # Assuming their JSON has 'frame_id' and a list of 'boxes'
        frame_id = frame_data.get('frame_id', count)
        boxes = frame_data.get('boxes', [])
        
        # Create a YOLO text file for this specific frame
        txt_filename = os.path.join(args.output_dir, f"frame_{int(frame_id):04d}.txt")
        
        with open(txt_filename, 'w') as out_file:
            for box in boxes:
                # YOLO format: class x_center y_center width height
                # Assuming class 0 is pedestrian, class 2 is car
                cls_id = box.get('class_id', 0)
                x = box.get('x', 0.5)
                y = box.get('y', 0.5)
                w = box.get('w', 0.1)
                h = box.get('h', 0.2)
                
                out_file.write(f"{cls_id} {x} {y} {w} {h}\n")
        count += 1

    print(f"✅ Conversion complete! Saved {count} YOLO label files to {args.output_dir}")
    print("These are ready to be fed into prepare_yolo_data.py!")

if __name__ == "__main__":
    main()