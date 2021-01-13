import torch
import torch.nn as nn
import numpy as np
import scipy
import librosa
import youtube_dl
import os
import soundfile as sf
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm_lasaft import DCUN_TFC_GPoCM_LaSAFT_Framework

#### Audio Utils
def audio_download(url):

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
        'outtmpl': os.path.join('%(id)s', '%(id)s.wav'),
        #'outtmpl': '%(title)s' + '.wav',
        'progress_hooks': [my_hook],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        status = ydl.download([url])
    
    return info

#### Model
def get_model_args():

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

    args['n_blocks'] = 7
    args['input_channels'] = 4
    args['internal_channels'] = 24
    args['first_conv_activation'] = 'relu'
    args['last_activation'] = 'identity'
    args['t_down_layers'] = None
    args['f_down_layers'] = None
    args['tif_init_mode'] = None

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

    return args

def get_model():

    args = get_model_args()

    model = DCUN_TFC_GPoCM_LaSAFT_Framework(**args)

    return model

def load_pretrained(model, pretrained='pretrained/gpocm_lasaft.ckpt'):
    
    model = model.load_from_checkpoint(pretrained)

    return model

### Separate
def separate_all(model, audio, ID):
    
    sr = 44100
    
    print('vocals')
    vocals = model.separate_track(audio.T, 'vocals', writePath=ID)

    print('drums')
    drums = model.separate_track(audio.T, 'drums', writePath=ID)

    print('bass')
    bass = model.separate_track(audio.T, 'bass', writePath=ID)

    print('other')
    other = model.separate_track(audio.T, 'other', writePath=ID)

    # audio mixture
    print('v+d+b+o')
    sf.write( os.path.join(ID, 'mixture.wav'),  vocals+drums+bass+other, sr, 'PCM_24')

### Main
def main(args):

    # load model arch and pretrained
    model = get_model()
    model = load_pretrained(model)

    if args.url is not None and args.a is None:
        # download audio
        url = args.url
        info = audio_download(url)

        ID = info.get('id', None)
        audioPath = os.path.join(ID, ID + '.wav')
    
    else:
        # load audio from local path
        ID = os.path.splitext(args.a)[0]
        
        if not os.path.exists(ID):
            os.makedirs(ID)
        
        audioPath = args.a

    audio, rate = librosa.load(audioPath, sr=44100, mono=False)

    # cut audio
    start = int(args.s)
    stop = int(args.e)

    if stop != 0 and isinstance(stop, int) and stop > start:
        audio = audio[:, start*rate:stop*rate]

    # separate audio
    separate_all(model, audio, ID)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    usage = 'python inference.py'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=usage, add_help=False)
    parser.add_argument('--a', default=None, help='local audio path')
    parser.add_argument('--url', default=None, help='youtube url')
    parser.add_argument('--s', default=0, help='start time')
    parser.add_argument('--e', help='end time')

    args = parser.parse_args()

    main(args)
