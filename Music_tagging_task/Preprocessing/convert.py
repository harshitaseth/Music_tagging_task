import glob
import numpy as np
from os import path
from tqdm import tqdm
from pydub import AudioSegment
from scipy.io import wavfile

# files                                                                         
src = glob.glob("../Data/musimotion_youtube_subtestset/*")
dst = "../Data/wavs"

#convert wav to mp3  
for file_ in src:
    sound = AudioSegment.from_mp3(file_)
    sound.export(dst+"/"+file_.split("/")[-1].split(".")[0]+".wav", format="wav")

src  = glob.glob("./Data/wavs/*")
dst = "../Data/specs_new/"
for file_ in tqdm(src):
    sr, y = wavfile.read(file_)
    path = dst +file_.split("/")[-1].split(".")[0]+".npy"
    np.save(path, y, allow_pickle=True, fix_imports=True)
