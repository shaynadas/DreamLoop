import json
import os
import math
import argparse


def make_transform_matrix(x, y, z, heading):
    """Build a 4x4 object_to_world matrix from x, y, z, heading (radians)."""
    c = math.cos(heading)
    s = math.sin(heading)
    return [
        [c,   -s,  0.0, x],
        [s,    c,  0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ]


def make_object_entry(obj_id, x, y, z, heading, lwh, obj_type, category):
    return {
        "object_to_world": make_transform_matrix(x, y, z, heading),
        "object_lwh": lwh,
        "object_is_moving": True,
        "object_type": obj_type,
        "aux_info": {
            "trackline_id": obj_id,
            "category": category,
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


def get_json_files(frames_dir):
    """Auto-detect and return sorted list of all JSON label files in a directory."""
    files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith(".json")
    ])
    if len(files) == 0:
        raise FileNotFoundError(f"No JSON files found in {frames_dir}")
    print(f"Found {len(files)} frames in {frames_dir}")
    return files


def inject_scenario(frames_dir, output_dir, scenario):
    """
    Inject a hazard scenario into all frames in frames_dir.

    scenario options:
        pedestrian_cross  - pedestrian walks into car path from the right
        collision         - vehicle comes head-on
        jaywalker         - pedestrian crosses from the left
        bike              - cyclist cuts across path
    """
    os.makedirs(output_dir, exist_ok=True)
    files = get_json_files(frames_dir)
    num_frames = len(files)

    # Define scenario parameters
    scenarios = {
        "pedestrian_cross": {
            "obj_id":    "injected:pedestrian:001",
            "obj_type":  "Pedestrian",
            "category":  "pedestrian",
            "lwh":       [0.5, 0.5, 1.75],
            "start":     (5.0, -4.0, 0.0),
            "end":       (5.0,  0.0, 0.0),
            "heading":   math.pi / 2,
        },
        "collision": {
            "obj_id":    "injected:collision_vehicle:001",
            "obj_type":  "Automobile",
            "category":  "automobile",
            "lwh":       [4.5, 2.0, 1.6],
            "start":     (30.0, 0.0, 0.0),
            "end":       (0.0,  0.0, 0.0),
            "heading":   math.pi,
        },
        "jaywalker": {
            "obj_id":    "injected:pedestrian:002",
            "obj_type":  "Pedestrian",
            "category":  "pedestrian",
            "lwh":       [0.5, 0.5, 1.75],
            "start":     (8.0,  4.0, 0.0),
            "end":       (8.0, -1.0, 0.0),
            "heading":   -math.pi / 2,
        },
        "bike": {
            "obj_id":    "injected:cyclist:001",
            "obj_type":  "Cyclist",
            "category":  "cyclist",
            "lwh":       [1.8, 0.6, 1.5],
            "start":     (10.0, -5.0, 0.0),
            "end":       (6.0,   1.0, 0.0),
            "heading":   math.pi / 3,
        },
    }

    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {list(scenarios.keys())}")

    cfg = scenarios[scenario]
    sx, sy, sz = cfg["start"]
    ex, ey, ez = cfg["end"]

    boxes_out = []

    for i, fname in enumerate(files):
        fpath = os.path.join(frames_dir, fname)
        with open(fpath) as f:
            frame_data = json.load(f)

        # Linear interpolation across frames
        t = i / max(num_frames - 1, 1)
        x = sx + t * (ex - sx)
        y = sy + t * (ey - sy)
        z = sz + t * (ez - sz)

        frame_data[cfg["obj_id"]] = make_object_entry(
            cfg["obj_id"], x, y, z,
            cfg["heading"], cfg["lwh"],
            cfg["obj_type"], cfg["category"]
        )

        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w") as f:
            json.dump(frame_data, f)

        boxes_out.append({
            "frame":    i,
            "filename": fname,
            "scenario": scenario,
            "x": x, "y": y, "z": z,
            "l": cfg["lwh"][0],
            "w": cfg["lwh"][1],
            "h": cfg["lwh"][2],
            "heading": cfg["heading"]
        })

    boxes_path = os.path.join(output_dir, "boxes.json")
    with open(boxes_path, "w") as f:
        json.dump(boxes_out, f, indent=2)

    print(f"Done. Scenario '{scenario}' injected into {num_frames} frames.")
    print(f"Output dir : {output_dir}")
    print(f"boxes.json : {boxes_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject hazard scenarios into RDS-HQ label files.")
    parser.add_argument("--data_dir",   required=True,  help="Path to folder of all_object_info JSON files")
    parser.add_argument("--output_dir", required=True,  help="Where to save modified JSON files + boxes.json")
    parser.add_argument("--scenario",   default="pedestrian_cross",
                        choices=["pedestrian_cross", "collision", "jaywalker", "bike"],
                        help="Which hazard to inject (default: pedestrian_cross)")
    args = parser.parse_args()

    inject_scenario(args.data_dir, args.output_dir, args.scenario)
