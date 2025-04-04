# -*- coding: utf-8 -*-
"""Copia di LaSAFT with GPoCM - Stella Jang Example

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uNpdkVqcMM3nLKnwahyb7xPNzBJtxmXP

# LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation

We used the jupyter notebook form [colab notebook of UMX(Open Unmix)](https://colab.research.google.com/drive/1mijF0zGWxN-KaxTnd0q6hayAlrID5fEQ) as our template.

# Installation and Imports (RUN THESE CELLS FIRST)
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install musdb
# !pip install norbert
# !pip install librosa
# !pip install youtube-dl
# !pip install museval
# !pip install pydub
# !pip install pytorch_lightning==0.9.0 
# !pip install soundfile
# !pip install wandb
# !git clone https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT

import torch
import torch.nn as nn
import numpy as np
import scipy
import librosa
import youtube_dl
import os
import soundfile as sf
from google.colab import files
from IPython.display import Audio, display

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Commented out IPython magic to ensure Python compatibility.
# %cd Conditioned-Source-Separation-LaSAFT

"""# Define Model

"""

from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm_lasaft import DCUN_TFC_GPoCM_LaSAFT_Framework


args = {}

# FFT params
args['n_fft'] = 2048
args['hop_length'] = 1024
args['num_frame'] = 128

# SVS Framework
args['spec_type'] = 'complex'
args['spec_est_mode'] = 'mapping'

# Other Hyperparams
args['optimizer'] = 'adam'
args['lr'] = 0.001
args['dev_mode'] = False
args['train_loss'] = 'spec_mse'
args['val_loss'] = 'raw_l1'

# DenseNet Hyperparams

args ['n_blocks'] = 7
args ['input_channels'] = 4
args ['internal_channels'] = 24
args ['first_conv_activation'] = 'relu'
args ['last_activation'] = 'identity'
args ['t_down_layers'] = None
args ['f_down_layers'] = None
args ['tif_init_mode'] = None

# TFC_TDF Block's Hyperparams
args['n_internal_layers'] =5
args['kernel_size_t'] = 3
args['kernel_size_f'] = 3
args['tfc_tdf_activation'] = 'relu'
args['bn_factor'] = 16
args['min_bn_units'] = 16
args['tfc_tdf_bias'] = True
args['num_tdfs'] = 6
args['dk'] = 32

args['control_vector_type'] = 'embedding'
args['control_input_dim'] = 4
args['embedding_dim'] = 32
args['condition_to'] = 'decoder'

args['control_n_layer'] = 4
args['control_type'] = 'dense'
args['pocm_type'] = 'matmul'
args['pocm_norm'] = 'batch_norm'


model = DCUN_TFC_GPoCM_LaSAFT_Framework(**args)

"""# Load Pretrained Parameters"""

model = model.load_from_checkpoint('pretrained/gpocm_lasaft.ckpt')

"""# Villain - Stella Jang"""

from IPython.display import HTML
url = "ghpn99s8I-U" #@param {type:"string"}
start = 67 #@param {type:"number"}
stop = 77 #@param {type:"number"}
embed_url = "https://www.youtube.com/embed/%s?rel=0&start=%d&end=%d&amp;controls=0&amp;showinfo=0" % (url, start, stop)
HTML('<iframe width="560" height="315" src=' + embed_url + 'frameborder="0" allowfullscreen></iframe>')

def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading...')


ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '44100',
    }],
    'outtmpl': '%(title)s.wav',
    'progress_hooks': [my_hook],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    status = ydl.download([url])

audio, rate = librosa.load(info.get('title', None) + '.wav', sr=44100, mono=False)
audio = audio[:, start*rate:stop*rate]
print(audio.shape)
display(Audio(audio, rate=rate))

def separate_all (audio):
  print('vocals')
  separated = model.separate_track(audio.T, 'vocals') 
  vocals, sr=librosa.load('temp.wav', mono=False)
  display(Audio('temp.wav')) 

  print('drums')
  separated = model.separate_track(audio.T, 'drums') 
  drums, sr=librosa.load('temp.wav', mono=False)
  display(Audio('temp.wav')) 

  print('bass')
  separated = model.separate_track(audio.T, 'bass') 
  bass, sr=librosa.load('temp.wav', mono=False)
  display(Audio('temp.wav')) 

  print('other')
  separated = model.separate_track(audio.T, 'other') 
  other, sr=librosa.load('temp.wav', mono=False)
  display(Audio('temp.wav')) 

  print('v+d+b+o')
  librosa.output.write_wav('temp.wav', vocals+drums+bass+other, sr)
  display(Audio('temp.wav')) 

separate_all(audio)

"""# Vanishing Paycheck - Stella Jang

"""

url = "EVGVJQtwxCY" #@param {type:"string"}
start = 9 #@param {type:"number"}
stop = 19 #@param {type:"number"}
embed_url = "https://www.youtube.com/embed/%s?rel=0&start=%d&end=%d&amp;controls=0&amp;showinfo=0" % (url, start, stop)
HTML('<iframe width="560" height="315" src=' + embed_url + 'frameborder="0" allowfullscreen></iframe>')

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '44100',
    }],
    'outtmpl': '%(title)s.wav',
    'progress_hooks': [my_hook],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    status = ydl.download([url])

audio, rate = librosa.load(info.get('title', None) + '.wav', sr=44100, mono=False)
audio = audio[:, start*rate:stop*rate]
print(audio.shape)
display(Audio(audio, rate=rate))

separate_all(audio)

"""# La Vie En Rose - Stella Jang"""

url = "JU5LMG3WFBw" #@param {type:"string"}
start = 0 #@param {type:"number"}
stop = 219 #@param {type:"number"}
embed_url = "https://www.youtube.com/embed/%s?rel=0&start=%d&end=%d&amp;controls=0&amp;showinfo=0" % (url, start, stop)
HTML('<iframe width="560" height="315" src=' + embed_url + 'frameborder="0" allowfullscreen></iframe>')

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '44100',
    }],
    'outtmpl': '%(title)s.wav',
    'progress_hooks': [my_hook],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    status = ydl.download([url])

audio, rate = librosa.load(info.get('title', None) + '.wav', sr=44100, mono=False)
audio = audio[:, start*rate:stop*rate]
print(audio.shape)
display(Audio(audio, rate=rate))

separate_all(audio)