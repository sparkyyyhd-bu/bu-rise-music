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
    def __init__(self, directory):
        self.paths = [
            entry.path
            for entry in os.scandir(directory)
            if entry.is_file()
        ]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        waveform, sample_rate = torchaudio.load(path)       
        output_name = os.path.splitext(os.path.basename(path))[0] + "_mel.pt"
        # song.mp3 -> song_mel.pt
        return waveform, sample_rate, output_name

preview_dir = f"/scratch/{os.environ['USER']}/previews"
output_directory = os.path.join("/scratch",os.environ["USER"],"mel_spectrograms")
os.makedirs(output_directory, exist_ok=True)

dataset = AudioDataset(preview_dir)

loader = DataLoader(
    dataset,
    batch_size=1    ,
    num_workers=8,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
)

mel_transforms = {}
amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

with torch.inference_mode():
    for waveform, sample_rates, output_name in loader:
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


