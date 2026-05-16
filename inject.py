import json
import os
import copy
import math
import argparse

CLIP_ID = "0079aad7-0fc5-4722-804d-e7c8c1b84263_570745200000_570765200000"
DATA_DIR = "./Cosmos-Drive-Dreams/data/all_object_info"
OUTPUT_DIR = "./injected"

def make_pedestrian_matrix(x, y, z, heading):
    """Build a 4x4 object_to_world matrix from x, y, z, heading (radians)."""
    c = math.cos(heading)
    s = math.sin(heading)
    return [
        [c,  -s,  0.0, x],
        [s,   c,  0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ]

def inject_pedestrian(frames_dir, output_dir, num_frames=121):
    os.makedirs(output_dir, exist_ok=True)

    # Pedestrian walks from right side of road into car's path
    # Start position: 5m ahead, 4m to the right
    # End position:   5m ahead, 0m (center of lane)
    start_x, start_y = 5.0, -4.0
    end_x,   end_y   = 5.0,  0.0
    z = 0.0  # ground level

    # Pedestrian dimensions: l=0.5, w=0.5, h=1.75
    lwh = [0.5, 0.5, 1.75]

    # heading: walking left-to-right = 90 degrees
    heading = math.pi / 2

    files = sorted([
        f for f in os.listdir(frames_dir)
        if CLIP_ID in f and f.endswith(".json")
    ])

    print(f"Found {len(files)} frames")

    boxes_out = []

    for i, fname in enumerate(files):
        fpath = os.path.join(frames_dir, fname)
        with open(fpath) as f:
            frame_data = json.load(f)

        # Linear interpolation of position across frames
        t = i / max(len(files) - 1, 1)
        x = start_x + t * (end_x - start_x)
        y = start_y + t * (end_y - start_y)

        # Build new pedestrian entry
        ped_id = "injected:pedestrian:001"
        frame_data[ped_id] = {
            "object_to_world": make_pedestrian_matrix(x, y, z, heading),
            "object_lwh": lwh,
            "object_is_moving": True,
            "object_type": "Pedestrian",
            "aux_info": {
                "trackline_id": ped_id,
                "category": "pedestrian",
                "egomotion_label_class_id": "injected",
                "mounted": False,
                "has_trailer": False,
                "has_protrusion": False,
                "automobile_type": "",
                "truck_type": "",
                "bus_type": "",
                "puller_type": "",
                "rider_type": "",
                "alive": True,
                "parent_obstacle_label_id": "",
                "lidar_sensor": ""
            }
        }

        # Save modified frame
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w") as f:
            json.dump(frame_data, f)

        # Record box for boxes.json
        boxes_out.append({
            "frame": i,
            "filename": fname,
            "x": x,
            "y": y,
            "z": z,
            "l": lwh[0],
            "w": lwh[1],
            "h": lwh[2],
            "heading": heading
        })

    # Write boxes.json for Person B and D to consume
    boxes_path = os.path.join(output_dir, "boxes.json")
    with open(boxes_path, "w") as f:
        json.dump(boxes_out, f, indent=2)

    print(f"Done. Injected pedestrian into {len(files)} frames.")
    print(f"Output: {output_dir}")
    print(f"boxes.json: {boxes_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    inject_pedestrian(args.data_dir, args.output_dir)