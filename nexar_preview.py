"""
nexar_preview.py
----------------
Downloads the first frame of N Nexar videos and saves them as
preview images so you can pick the best one without downloading
full video files.

USAGE:
    python nexar_preview.py            # previews first 20 videos
    python nexar_preview.py --count 40 # previews first 40 videos
    python nexar_preview.py --start 20 # starts from index 20
"""

import argparse
import os
import cv2
import tempfile
from huggingface_hub import list_repo_files, hf_hub_download

REPO = 'nexar-ai/nexar_collision_prediction'
PREVIEW_DIR = './nexar_previews'

def grab_first_frame(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        # Resize to thumbnail
        h, w = frame.shape[:2]
        thumb_w = 480
        thumb_h = int(h * thumb_w / w)
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        cv2.imwrite(out_path, thumb)
        return True
    return False

def main(start, count):
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    print("Fetching file list from Hugging Face...")
    files = list(list_repo_files(REPO, repo_type='dataset'))
    mp4s = [f.strip() for f in files if f.strip().startswith('train/positive') and f.strip().endswith('.mp4')]
    print(f"Found {len(mp4s)} train videos total.")

    batch = mp4s[start:start + count]
    print(f"Previewing videos {start} to {start + len(batch) - 1}\n")

    results = []
    for i, filename in enumerate(batch):
        idx = start + i
        print(f"[{idx}] {filename} ... ", end='', flush=True)

        try:
            # Download to temp location
            local = hf_hub_download(
                repo_id=REPO,
                filename=filename,
                repo_type='dataset',
                local_dir='./nexar_temp'
            )
            out_img = os.path.join(PREVIEW_DIR, f"preview_{idx:04d}.jpg")
            ok = grab_first_frame(local, out_img)
            if ok:
                print(f"saved → {out_img}")
                results.append((idx, filename, out_img))
            else:
                print("could not read frame")
        except Exception as e:
            print(f"SKIP ({e})")

    print(f"\n── Done ──────────────────────────────────────")
    print(f"Previews saved to: {os.path.abspath(PREVIEW_DIR)}")
    print(f"\nTo use a video in the sim, run:")
    print(f"  python nexar_preview.py shows index numbers")
    print(f"  Then download the full video:")
    print(f"  python get_nexar_video.py  (edit index in script)")
    print()
    print("Index  |  Filename")
    print("-------|--------------------------------------------")
    for idx, fname, img in results:
        print(f"  {idx:4d} |  {fname}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index (default 0)')
    parser.add_argument('--count', type=int, default=20, help='How many to preview (default 20)')
    args = parser.parse_args()
    main(args.start, args.count)
