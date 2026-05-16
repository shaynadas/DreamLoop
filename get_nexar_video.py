# from huggingface_hub import list_repo_files, hf_hub_download

# print("Finding train videos...")
# files = list(list_repo_files('nexar-ai/nexar_collision_prediction', repo_type='dataset'))

# # Only grab from train folder (public, no login needed), strip spaces
# train_mp4s = [f.strip() for f in files if f.strip().startswith('train/') and f.strip().endswith('.mp4')]

# print(f"Found {len(train_mp4s)} train videos.")
# if not train_mp4s:
#     print("No train videos found. All files:")
#     for f in files[:20]:
#         print(" ", repr(f))
# else:
#     target = train_mp4s[0]
#     print(f"Downloading: {target}")
#     path = hf_hub_download(
#         repo_id='nexar-ai/nexar_collision_prediction',
#         filename=target,
#         repo_type='dataset',
#         local_dir='./nexar_videos'
#     )
#     print(f"\nSaved to: {path}")
#     print(f"\nRun the sim with:")
#     print(f'  python dreamloop_sim.py --video "{path}"')

from huggingface_hub import list_repo_files, hf_hub_download

print("Finding positive train videos...")
files = list(list_repo_files('nexar-ai/nexar_collision_prediction', repo_type='dataset'))

train_mp4s = [f.strip() for f in files if f.strip().startswith('train/positive') and f.strip().endswith('.mp4')]

print(f"Found {len(train_mp4s)} positive videos.")
if not train_mp4s:
    print("No videos found. All files:")
    for f in files[:20]:
        print(" ", repr(f))
else:
    target = train_mp4s[13] 
    print(f"Downloading: {target}")
    path = hf_hub_download(
        repo_id='nexar-ai/nexar_collision_prediction',
        filename=target,
        repo_type='dataset',
        local_dir='./nexar_videos'
    )
    print(f"\nSaved to: {path}")
    print(f"\nRun the sim with:")
    print(f'  python dreamloop_sim.py --video "{path}"')