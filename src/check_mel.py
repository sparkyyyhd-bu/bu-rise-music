import torch
import random
import os
import matplotlib.pyplot as plt

src_dir = f"/net/scc1/scratch/{os.environ['USER']}/mel_spectrograms"
n = 3
files = os.listdir(src_dir)
samples = random.sample(range(0,len(files)),n)
filenames = [files[i] for i in samples]
output_dir = f"/net/scc1/scratch/{os.environ['USER']}/figures/mel_spectrograms"
os.makedirs(output_dir, exist_ok = True)

for filename in filenames:
    mel_spec_db = torch.load(os.path.join(src_dir, filename))
    if mel_spec_db.dim() == 3:
        mel_spec_db = mel_spec_db.mean(dim=0)
    output_name = os.path.splitext(filename)[0]
    plt.figure(figsize=(10,4))
    plt.imshow(mel_spec_db.numpy(), cmap='magma', aspect='auto')
    plt.title('Spectrogram (dB)')
    plt.ylabel('Frequency Bin')
    plt.xlabel('Frame')
    plt.savefig(os.path.join(output_dir, output_name))

