import torch
import torchaudio
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class AudioDataset(Dataset):
    def __init__(self, directory, output_directory):
        self.paths = [
            entry.path
            for entry in os.scandir(directory)
            if entry.is_file() and entry.name.endswith(".mp3")
        ]
        self.output_directory = output_directory
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        output_name = os.path.splitext(os.path.basename(path))[0] + "_mel.pt"
        # song.mp3 -> song_mel.pt
        if os.path.exists(os.path.join(self.output_directory, output_name)):
            return None, None, None
        try:
            waveform, sample_rate = torchaudio.load(path)
        except Exception as e:
            print(f"skipping {path}: {e}")
            return None, None, None
        return waveform, sample_rate, output_name

preview_dir = f"/scratch/{os.environ['USER']}/previews"
output_directory = os.path.join("/scratch",os.environ["USER"],"mel_spectrograms")
os.makedirs(output_directory, exist_ok=True)

dataset = AudioDataset(preview_dir, output_directory)

def collate_fn(batch):
    batch = [sample for sample in batch if sample[0] is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

loader = DataLoader(
    dataset,
    batch_size=1    ,
    num_workers=8,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
    collate_fn=collate_fn,
)

mel_transforms = {}
amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

with torch.inference_mode():
    for batch in loader:
        if batch is None:
            continue
        waveform, sample_rates, output_name = batch
        sample_rate = int(sample_rates[0])
        waveform = waveform.to(
            device,
            non_blocking=True,
        )
        if (sample_rate not in mel_transforms):
            mel_transforms[sample_rate] = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_fft=2048,
                        hop_length=512,
                        n_mels=128
                    ).to(device)
        
        mel_spectrograms = mel_transforms[sample_rate](waveform)
        mel_spectrograms_db = amplitude_to_db(mel_spectrograms)
        mel_to_save = mel_spectrograms_db[0].cpu()
        torch.save(mel_to_save, os.path.join(output_directory, output_name[0]))


