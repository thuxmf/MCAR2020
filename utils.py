import librosa
import os
import numpy as np
import torch
from pathlib import Path
from math import ceil, floor
from scipy.spatial import Delaunay
import cv2, face_alignment#, dlib


int16_max = (2 ** 15) - 1
audio_norm_target_dBFS = -30
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40

Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
         [57, 58], [58, 59], [59, 48]]

Lips = [[60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
         [66, 67], [67, 60]]

Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], \
        [33, 34], [34, 35]]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16]]
ranks = [0, 17, 5, 5, 9, 6, 6, 20]
faceLmarkLookup = Mouth + Lips + Nose + leftBrow + rightBrow + leftEye + rightEye + other
# ms_img = np.concatenate((np.load('mean_shape_img.npy'), np.zeros((68, 1))), axis=1)
ms_img = np.load('template.npy')


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def tensor_seq(tensor, seq_step):
    length = tensor.size(0)
    _tensor = [tensor[t:t+seq_step].unsqueeze(0) for t in range(length-seq_step+1)]
    _tensor = torch.cat(_tensor, dim=0)
    return _tensor


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        16000,
        n_fft=int(16000 * mel_window_length / 1000),
        hop_length=int(16000 * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def wav_process(filename, source_sr=16000):
    # Load the wav from disk if needed
    fpath = Path(filename)
    wav, sr = librosa.load(fpath, sr=None)

    # Resample the wav
    if sr is not None:
        wav = librosa.resample(wav, sr, source_sr)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)  # shape: (n_seconds * 16000, )
    mel = wav_to_mel_spectrogram(wav)  # shape: (n_seconds * 100, 40)

    return wav, mel


def laplace(inputs, circle_flag=True):
    length = inputs.size(1)
    if circle_flag:
        tmp = torch.cat([inputs[:, -1, :].unsqueeze(1), inputs, inputs[:, 0, :].unsqueeze(1)], dim=1)
    else:
        tmp = torch.cat([inputs[:, 1, :].unsqueeze(1), inputs, inputs[:, -2, :].unsqueeze(1)], dim=1)
    return inputs - (tmp[:, :length, :] + tmp[:, -length:, :]) * 0.5


def Laplacian(inputs):
    """face: 17, left brow: 5, right brow: 5, nose: 9, left eye: 6, right eye: 6, mouth: 20"""
    inputs = inputs.view(inputs.size(0), 68, -1)
    face = laplace(inputs[:, :17, :], False)
    left_brow = laplace(inputs[:, 17:22, :], False)
    right_brow = laplace(inputs[:, 22:27, :], False)
    nose = laplace(inputs[:, 27:36, :], False)

    left_eye = laplace(inputs[:, 36:42, :])
    right_eye = laplace(inputs[:, 42:48, :])
    mouth = laplace(inputs[:, 48:, :])

    return torch.cat([face, left_brow, right_brow, nose, left_eye, right_eye, mouth], dim=1)


def Laplacian_old(inputs):
    inputs = inputs.view(inputs.size(0), 68, -1)
    outputs = []
    i, j, hot = 0, 0, 0
    while i < 68:
        if i < hot-1:
            out = inputs[:, i, :] - (inputs[:, i+1, :]+inputs[:, i-1, :]) / 2
        elif i == hot:
            j += 1
            hot += ranks[j]
            out = inputs[:, i, :] - inputs[:, i+1, :]
        else:
            out = inputs[:, i, :] - inputs[:, i-1, :]
        outputs.append(out.unsqueeze(1))
        i += 1

    return torch.cat(outputs, dim=1)


def draw_line(input, start_x, start_y, end_x, end_y, color):
    delta_y, delta_x = float(end_y - start_y), float(end_x - start_x)
    if delta_x == 0:
        for y in range(min(start_y, end_y) + 1, max(start_y, end_y)):
            input[start_x, y] = color
    else:
        k = delta_y / delta_x
        if k <= 1 and k >= -1:
            for x in range(min(start_x, end_x) + 1, max(start_x, end_x)):
                y = int(float(x - start_x) * k + start_y)
                input[x, y] = color
        else:
            for y in range(min(start_y, end_y) + 1, max(start_y, end_y)):
                x = int(float(y - start_y) / k + start_x)
                input[x, y] = color


def draw(frame, lands, c=None):
    width, height = frame.shape[:2]
    lands = (lands + 0.5).astype(int)
    land_x, land_y = lands[:, 1], lands[:, 0]
    land_x = np.clip(land_x, 0, width)
    land_y = np.clip(land_y, 0, height)
    lands = np.concatenate((land_y[:, np.newaxis], land_x[:, np.newaxis]), axis=1)
    color = c if c is not None else [255, 0, 0]
    lands = np.clip(lands, 0, 223)
    for land in lands:
        land = [min(223, max(0, x)) for x in land]
        frame[land[1], land[0]] = color
    for refpts in faceLmarkLookup:
        draw_line(frame, lands[refpts[0], 1], lands[refpts[0], 0], lands[refpts[1], 1], lands[refpts[1], 0], color)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x in ('True')


def transform(origin, params=None):
    if params is not None:
        landmarks = np.dot(origin, params)
        return landmarks, None
    else:
        points1 = origin.astype(np.float64)
        points2 = ms_img.astype(np.float64)
        c1, c2 = np.mean(points1, 0), np.mean(points2, 0)
        points1_ = points1 - c1
        points2_ = points2 - c2
        s1 = np.std(points1_)
        s2 = np.std(points2_)
        points1_ /= s1
        points2_ /= s2

        U, _, Vt = np.linalg.svd(np.dot(points1_.T, points2_))
        R = np.dot(U, Vt)
        # print(np.sum((np.dot(points1_, R) - points2_) ** 2), np.sum((points1_ - points2_) ** 2))
        return np.dot(points1-c1, R)+np.array([111, 111, 0]), R


def traverse(a, b, c):
    a, b, c = sorted([a, b, c], key=lambda x: x[1])
    l = []
    x1, x2, y1, y2 = b[0]-a[0], c[0]-a[0], b[1]-a[1], c[1]-a[1]
    if abs(x1*y2-x2*y1) < 0.001:
        return l
    elif a[1] == b[1]:
        a, b = (b, a) if a[0] > b[0] else (a, b)
        l += [[x, a[1]] for x in range(floor(a[0]+0.01), ceil(b[0]-0.01)+1)]
        k1, k2 = (a[0]-c[0]) / (a[1]-c[1]), (b[0]-c[0]) / (b[1]-c[1])
        x_l, x_r = a[0], b[0]
        for y in range(a[1]+1, c[1]+1):
            x_l += k1
            x_r += k2
            l += [[x, y] for x in range(ceil(x_l-0.01), floor(x_r+0.01)+1)]
    elif b[1] == c[1]:
        b, c = (c, b) if b[0] > c[0] else (b, c)
        l += [[x, b[1]] for x in range(floor(b[0]+0.01), ceil(c[0]-0.01)+1)]
        k1, k2 = (a[0]-b[0]) / (a[1]-b[1]), (a[0]-c[0]) / (a[1]-c[1])
        x_l, x_r = a[0],  a[0]
        for y in range(a[1], c[1]):
            l += [[x, y] for x in range(ceil(x_l-0.01), floor(x_r+0.01)+1)]
            x_l += k1
            x_r += k2
    else:
        k1, k2, k3 = (a[0]-c[0]) / (a[1]-c[1]), (a[0]-b[0]) / (a[1]-b[1]), (b[0]-c[0]) / (b[1]-c[1])
        x_l, x_r = a[0] - k1, a[0] - k2
        for y in range(a[1], b[1]+1):
            x_l += k1
            x_r += k2
            l += [[x, y] for x in range(ceil(min(x_l, x_r)-0.01), floor(max(x_l, x_r)+0.01)+1)]
        for y in range(b[1]+1, c[1]+1):
            x_l += k1
            x_r += k3
            l += [[x, y] for x in range(ceil(min(x_l, x_r)-0.01), floor(max(x_l, x_r)+0.01)+1)]
    return l


def coord(v, a, b, c):
    b_, c_, v_ = b-a, c-a, v-a
    if c_[0]*b_[1]-c_[1]*b_[0] == 0:
        print(a, b, c)
    inv = 1 / (c_[0]*b_[1]-c_[1]*b_[0])
    lambda_b = (v_[1]*c_[0]-c_[1]*v_[0]) * inv
    lambda_c = (v_[0]*b_[1]-b_[0]*v_[1]) * inv
    lambda_a = 1 - lambda_b - lambda_c
    return lambda_a, lambda_b, lambda_c


def area_of_signed_triangle(pts):
    AB = pts[1, :] - pts[0, :]
    AC = pts[2, :] - pts[0, :]
    return .5 * np.cross(AB, AC)


def area_of_signed_polygon(pts):
    l = pts.shape[0]
    s = 0.
    for i in range(1, l-1):
        s += area_of_signed_triangle(pts[(0, i, i+1), :])
    return s


'''
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')
cap = cv2.VideoCapture('00051.mp4')
out = cv2.VideoWriter('4.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
n_frames = 0
params = None
while cap.isOpened():
    ret, frame = cap.read()
    if ret and n_frames < 50:
        n_frames += 1
        print('\r# frames processed: %d' % n_frames, end='')
        out_frame = np.zeros((224, 224, 3), dtype=np.uint8) + 255
        lands = fa.get_landmarks(frame)[0]
        lands, p = transform(lands, params)
        lands = (lands+0.5).astype(int)
        if params is None:
            params = p
        draw(out_frame, lands)
        out.write(out_frame)
    else:
        print()
        break

cap.release()
out.release()
cv2.destroyAllWindows()
'''
