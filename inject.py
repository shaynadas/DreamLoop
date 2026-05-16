import json
import os
import math
import argparse
import cv2
import numpy as np
import torch


def make_transform_matrix(x, y, z, heading):
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
    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".json")])
    if not files:
        raise FileNotFoundError(f"No JSON files found in {frames_dir}")
    print(f"Found {len(files)} frames in {frames_dir}")
    return files


def world_to_birdseye(x, y, img_size=800, scale=10.0):
    cx, cy = img_size // 2, img_size // 2
    px = int(cx + y * scale)
    py = int(cy - x * scale)
    px = max(0, min(img_size - 1, px))
    py = max(0, min(img_size - 1, py))
    return px, py


def render_frame(boxes_so_far, frame_idx, total_frames, img_size=800, scale=10.0):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    for i in range(0, img_size, 50):
        cv2.line(img, (i, 0), (i, img_size), (30, 30, 30), 1)
        cv2.line(img, (0, i), (img_size, i), (30, 30, 30), 1)

    ego_px, ego_py = world_to_birdseye(0, 0, img_size, scale)
    cv2.rectangle(img, (ego_px - 10, ego_py - 20), (ego_px + 10, ego_py + 20), (0, 255, 0), -1)
    cv2.putText(img, "EGO", (ego_px - 15, ego_py + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    for b in boxes_so_far[:-1]:
        px, py = world_to_birdseye(b["x"], b["y"], img_size, scale)
        cv2.circle(img, (px, py), 2, (80, 80, 255), -1)

    if boxes_so_far:
        b = boxes_so_far[-1]
        px, py = world_to_birdseye(b["x"], b["y"], img_size, scale)
        color = (0, 100, 255) if b.get("category") == "pedestrian" else (0, 200, 255)
        w_px = max(4, int(b["w"] * scale))
        l_px = max(4, int(b["l"] * scale))
        cv2.rectangle(img, (px - w_px, py - l_px), (px + w_px, py + l_px), color, -1)
        cv2.putText(img, b.get("scenario", "injected"), (px + 8, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(img, f"Frame {frame_idx}/{total_frames}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.arrowedLine(img, (ego_px, ego_py), (ego_px, ego_py - 60), (0, 255, 0), 2, tipLength=0.3)

    return img


def save_mp4(boxes_out, output_dir, scenario, fps=30, img_size=800, scale=10.0):
    mp4_path = os.path.join(output_dir, f"{scenario}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (img_size, img_size))
    total = len(boxes_out)
    for i in range(total):
        frame_img = render_frame(boxes_out[:i+1], i, total, img_size, scale)
        writer.write(frame_img)
    writer.release()
    print(f"MP4 saved  : {mp4_path}")
    return mp4_path


def save_pt(boxes_out, output_dir, scenario, img_size=800, scale=10.0):
    class_map = {"pedestrian": 0, "cyclist": 0, "automobile": 1}
    rows = []
    for b in boxes_out:
        class_id = class_map.get(b.get("category", "pedestrian"), 0)
        px, py = world_to_birdseye(b["x"], b["y"], img_size, scale)
        cx = px / img_size
        cy = py / img_size
        bw = max(0.01, (b["w"] * scale * 2) / img_size)
        bh = max(0.01, (b["l"] * scale * 2) / img_size)
        rows.append([b["frame"], class_id, cx, cy, bw, bh])
    tensor = torch.tensor(rows, dtype=torch.float32)
    pt_path = os.path.join(output_dir, f"{scenario}_labels.pt")
    torch.save(tensor, pt_path)
    print(f"PT saved   : {pt_path}  shape={tuple(tensor.shape)}")
    return pt_path


def inject_scenario(frames_dir, output_dir, scenario):
    os.makedirs(output_dir, exist_ok=True)
    files = get_json_files(frames_dir)
    num_frames = len(files)

    scenarios = {
        "pedestrian_cross": {
            "obj_id":   "injected:pedestrian:001",
            "obj_type": "Pedestrian",
            "category": "pedestrian",
            "lwh":      [0.5, 0.5, 1.75],
            "start":    (5.0, -4.0, 0.0),
            "end":      (5.0,  0.0, 0.0),
            "heading":  math.pi / 2,
        },
        "collision": {
            "obj_id":   "injected:collision_vehicle:001",
            "obj_type": "Automobile",
            "category": "automobile",
            "lwh":      [4.5, 2.0, 1.6],
            "start":    (30.0, 0.0, 0.0),
            "end":      (0.0,  0.0, 0.0),
            "heading":  math.pi,
        },
        "jaywalker": {
            "obj_id":   "injected:pedestrian:002",
            "obj_type": "Pedestrian",
            "category": "pedestrian",
            "lwh":      [0.5, 0.5, 1.75],
            "start":    (8.0,  4.0, 0.0),
            "end":      (8.0, -1.0, 0.0),
            "heading":  -math.pi / 2,
        },
        "bike": {
            "obj_id":   "injected:cyclist:001",
            "obj_type": "Cyclist",
            "category": "cyclist",
            "lwh":      [1.8, 0.6, 1.5],
            "start":    (10.0, -5.0, 0.0),
            "end":      (6.0,   1.0, 0.0),
            "heading":  math.pi / 3,
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
            "category": cfg["category"],
            "x": x, "y": y, "z": z,
            "l": cfg["lwh"][0],
            "w": cfg["lwh"][1],
            "h": cfg["lwh"][2],
            "heading": cfg["heading"]
        })

    boxes_path = os.path.join(output_dir, "boxes.json")
    with open(boxes_path, "w") as f:
        json.dump(boxes_out, f, indent=2)
    print(f"JSON saved : {boxes_path}")

    save_mp4(boxes_out, output_dir, scenario)
    save_pt(boxes_out, output_dir, scenario)

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject hazard scenarios into RDS-HQ label files.")
    parser.add_argument("--data_dir",   required=True,
                        help="Path to folder of all_object_info JSON files")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save modified JSONs, boxes.json, .mp4, and .pt")
    parser.add_argument("--scenario",   default="pedestrian_cross",
                        choices=["pedestrian_cross", "collision", "jaywalker", "bike"],
                        help="Which hazard to inject (default: pedestrian_cross)")
    args = parser.parse_args()

    inject_scenario(args.data_dir, args.output_dir, args.scenario)
