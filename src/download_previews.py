import pandas as pd
import requests
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(repo_root, "data", "SpotGenTrack", "Data Sources", "spotify_tracks.csv"))
sample = df[df["preview_url"].notna()]
preview_path = os.path.join(f"/net/scc1/scratch/{os.environ['USER']}", "previews")
os.makedirs(preview_path, exist_ok=True)

for index, url in sample["preview_url"].items():
    track_id = df["id"][index]
    out_path = os.path.join(preview_path, f"{track_id}.mp3")
    if os.path.exists(out_path):
        continue
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"skipping {track_id}: {e}")
        continue
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(response.content)
    os.replace(tmp_path, out_path)

