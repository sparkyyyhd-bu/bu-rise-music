import kagglehub
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = kagglehub.dataset_download(
    "saurabhshahane/spotgen-music-dataset",
    output_dir=os.path.join(repo_root, "data")
)

print("Path to dataset files:", path)