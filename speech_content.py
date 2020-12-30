import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import argparse
from glob import glob
import itertools
import time
from utils import *
from dataloader import *


import face_alignment
from resemblyzer import VoiceEncoder
import cv2


class SpeechContent(nn.Module):
    def __init__(self):
        super(SpeechContent, self).__init__()
        
        self.lstm = nn.LSTM(input_size=40, hidden_size=256, num_layers=3, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(1024+204, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 3*68),
        )

    def forward(self, _A_t, q):
        """
        params:
            _A_t: content embedding from AutoVC, with window [t, t+\tau], \tau = 30 frames. (n_wav_frames, 30, 40)
            q: initial static landmark, (1, 68*3)
        return:
            p_t: output from SpeechContent. (n_vid_frames, 204)
        """
        n_wav_frames = _A_t.size(0)
        n_frames = n_wav_frames // 4
        output, (_, _) = self.lstm(_A_t)  # (n_wav_frames, 30, 256)
        output = output[:, -1, :]  # (n_wav_frames, 256)
        output = output[:4 * n_frames, :]  # (4*n_frames, 256)
        x = torch.cat([output.view(n_frames, 1024), q.expand(n_frames, -1)], dim=1)  # (n_frames, 1024+204)
        delta_q_t = self.mlp(x)  # (n_frames, 204)
        p_t = q.expand_as(delta_q_t) + delta_q_t
        return p_t


