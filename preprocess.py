import cv2
from utils import *
import os
import ssl
import face_alignment
from resemblyzer import VoiceEncoder
import numpy as np
import argparse
import pynvml
pynvml.nvmlInit()


def get_gpu():
    l = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        if 0 < index < 4 or index == 6:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = meminfo.used / meminfo.total
            if used < 0.7:
                l.append(index)
    return l


def video_preprocess(filemame, targetname):
    wav, mel = wav_process(filemame)
    if len(mel) < 100 or len(mel) > 1500:
        return 0
    embed = encoder.embed_utterance(wav)
    n_frames = 0
    cap = cv2.VideoCapture(filemame)
    landmarks = []
    flag, params = True, None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print('\r# frames processed: %d' % n_frames, end='')
            n_frames += 1
            lands = fa.get_landmarks(frame)[0]  # x, y = int(lands[0][1]), int(lands[0][0]), frame[x, y]
            if lands is None:
                flag = False
                break
            lands, p = transform(lands, params)
            if params is None:
                params = p
            landmarks.append(lands)
        else:
            print()
            break
    cv2.destroyAllWindows()
    cap.release()
    res = {'embed': embed, 'mel': mel, 'landmarks': landmarks}
    if flag:
        np.save(targetname, res)
    return flag


def preprocess(foldername, prefix, targetfolder):
    id = 0
    for name in os.listdir(targetfolder):
        if (prefix + '_') in name:
            id += 1

    if not os.path.exists(targetfolder):
        os.mkdir(targetfolder)
    for root, dirs, files in os.walk(foldername):
        for name in files:
            if 'mp4' in name:
                print('\r# video processed: %d, processing: %s' % (id, name))
                id += video_preprocess(os.path.join(root, name), targetfolder + '/%s_%d.npy' % (prefix, id))


parser = argparse.ArgumentParser(description='')
parser.add_argument('--foldername', type=str, default='', help='the folder to preprocess')
parser.add_argument('--targetfolder', type=str, default='')
parser.add_argument('--prefix', type=str, default='')
args = parser.parse_args()
foldername = args.foldername
prefix = args.prefix
targetfolder = args.targetfolder
if os.path.exists(foldername):
    l = get_gpu()
    print(l)
    device = 'cuda:%d' % l[0]
    encoder = VoiceEncoder(device)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)
    preprocess(foldername, prefix, targetfolder)

