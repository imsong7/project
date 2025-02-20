# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:11:34 2023

@author: bodyn
"""
#%%
import os
import ffmpy
import pydub
import glob
import shutil
from pydub import AudioSegment

import sys

sys.path.append('/opt/homebrew/bin/ffmpeg')
sys.path.append('/opt/homebrew/bin/ffprobe')
#%%

class fileconvert:
    def __init__(self, inputdir):
        self.inputdir = inputdir
        os.chdir(inputdir)

    def converter(self):
        for filename in os.listdir(self.inputdir):
            actual_fileformat = filename[-4:]
            if filename.endswith(".mp4"):
                os.system('ffmpeg -i {} -vn -ar {} -ac {} -ab {} {}.wav'.format(filename, '44.1k', 1, 16, filename))
                print(filename)
            elif filename.endswith(".m4a"):
                sound = AudioSegment.from_file(filename, format='m4a')
                sound.export(f'{filename}.wav', format='wav')
                print(filename)
            elif filename.endswith(".mov"):
                os.system('ffmpeg -i {} -vn -ar {} -ac {} -ab {} {}.wav'.format(filename, '44.1k', 1, 16, filename))
            else:
                print('Format is different')
                continue

        wav_files = glob.glob('*.wav')
        os.mkdir("new_wav_file")
        for file in wav_files:
            shutil.move(file, 'new_wav_file')

# if __name__ == "__main__":
#     conv = fileconvert("/Users/songsooyeon/Desktop/SD/매미탐사_8월1일~15일/origin_data")
#     conv.converter()
# # Example Usage
# converter = Converter("/path/to/input/directory")
# converter.convert_files()
