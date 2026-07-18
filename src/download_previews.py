import pandas as pd
import requests
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(repo_root, "data", "SpotGenTrack", "Data Sources", "spotify_tracks.csv"))
sample = df[df["preview_url"].notna()]
preview_path = os.path.join(f"/scratch/{os.environ['USER']}", "previews")
os.makedirs(preview_path, exist_ok=True)

for index, url in sample["preview_url"].items():
    response = requests.get(url)
    response.raise_for_status()
    track_id = df["id"][index]
    with open(os.path.join(preview_path, f"{track_id}.mp3"), "wb") as f:
        f.write(response.content)

