import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

class preprocessing:
    def __init__(self, dir_path, crop_height_low, crop_height_high, crop_sec):
        self.dir_path = dir_path
        self.audio_path = os.path.join(dir_path, "wav_file")
        self.image_path = os.path.join(dir_path, "spectogram")
        self.crop_height_low = crop_height_low
        self.crop_height_high = crop_height_high
        self.crop_sec = crop_sec
        self.save_path = os.path.join(dir_path, "input")
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'resized'), exist_ok=True)
        os.chdir(dir_path)

    def Fourier(self, audio):
        y, sr = librosa.load(audio, sr=44100, mono=True, offset=0.0, duration=60)
        stft_result = librosa.stft(y, n_fft=2048, win_length=2048, hop_length=512)
        return y, sr, stft_result

    def spectrogram(self, y, sr, stft_result, name):
        os.chdir(self.image_path)
        D = np.abs(stft_result)
        S_dB = librosa.amplitude_to_db(D, ref=np.max)
        plt.figure(figsize=(10,7), frameon=False, dpi=150)
        librosa.display.specshow(S_dB, sr=sr, n_fft=2048, win_length=2048, hop_length=512, y_axis='mel', x_axis='time')
        plt.axis('off')
        plt.savefig(f"{name}.png", bbox_inches='tight', pad_inches=0)
        plt.close("all")

    def process_audio(self):
        wav_files = os.listdir(self.audio_path)
        for f in wav_files:
            path = os.path.join(self.audio_path, f)
            y, sr, stft_result = self.Fourier(path)
            self.spectrogram(y, sr, stft_result, f[:-4])
            print(f'{f} file Done')

    def load_image(self):
        os.chdir(self.image_path)
        image_list = os.listdir(self.image_path)
        image_constant = Image.open(image_list[0])
        return image_constant, image_list

    def define_constants(self, image_constant):
        width, height = image_constant.size
        sr = 44100
        height_high = int((self.crop_height_high * height) / (sr / 2))
        height_low = int((self.crop_height_low * height) / (sr / 2))
        width_crop = int((width * self.crop_sec) / 60)
        return width, height, height_high, height_low, width_crop

    def crop_images(self, image_list, width, height, height_high, height_low, width_crop):
        for i in image_list:
            crop_image = Image.open(os.path.join(self.image_path, i))
            image_height_crop = crop_image.crop((0, height - height_high, 1162, height - height_low))
            os.mkdir(f'{i[:-4]}')
            os.chdir(os.path.join(self.image_path, f'{i[:-4]}'))
            coordinate = 0
            c = 0
            while coordinate < width:
                image_width_crop = image_height_crop.crop((coordinate, height - height_high,
                                                           coordinate + width_crop, height - height_low))
                coordinate += width_crop
                if coordinate > width:
                    break
                image_width_crop.save(f'{i[:-4]}_{c}.png')
                c += 1
            print(f"Finish the {i} crop")
            os.chdir(self.image_path)

    def folder_path(self, folder):
        return os.path.join(self.image_path, folder)

    def resizing(self, image):
        resizing = Image.open(image)
        trans = resizing.resize((244,244), Image.LANCZOS)
        return trans

    def process_images(self):
        save = os.path.join('r', self.save_path, 'resized')
        folder = glob.glob('**/')
        for f in folder:
            os.chdir(self.folder_path(f))
            print(os.getcwd())
            photo_list = os.listdir()
            for p in photo_list:
                result = self.resizing(p)
                result.save(os.path.join(save, f'{p[:-4]}.png'))
        return save

