from scipy.spatial import Delaunay
import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt
import cv2
import face_alignment
from model import *


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


model = Model()
params = torch.load('')
model.speech_content.load_state_dict(params['content'])
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')
start_frame = cv2.imread('start_frame.jpg')
out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
q = fa.get_landmarks(start_frame)[0]
A_t = np.load('0.npy', allow_pickle=True).item()['mel']
print(A_t.shape)
points = q[:, :2]
points = [[ceil(x-0.1) for x in point] for point in points]
points += [[0, 0], [223, 223], [0, 223], [223, 0], [0, 111], [111, 0], [111, 223], [223, 111]]
points = np.array(points)
tri = Delaunay(points)

landmarks = np.array(np.load('0.npy', allow_pickle=True).item()['landmarks'])[:, :, :2]
for t in range(len(landmarks)):
    print('\rprocessing frame %d' % t, end='')
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    lands = landmarks[t]
    lands = [[ceil(x-0.1) for x in land] for land in lands]
    lands += [[0, 0], [223, 223], [0, 223], [223, 0], [0, 111], [111, 0], [111, 223], [223, 111]]
    lands = np.array(lands)
    # transfer lands to np.array
    for indices in tri.simplices:
        a, b, c = lands[indices]
        a_, b_, c_ = points[indices]
        l = traverse(a, b, c)
        for v in l:
            l_a, l_b, l_c = coord(v, a, b, c)
            v = [ceil(x-0.1) for x in v]
            v_ = a_ * l_a + b_ * l_b + c_ * l_c  # coord in the start frame
            v_ = [ceil(x - 0.1) for x in v_]
            frame[v[1], v[0]] = start_frame[v_[1], v_[0]]
    out.write(frame)



'''
color = []
for index, sim in enumerate(points[tri.simplices]):
    cx, cy = center[index][0], center[index][1]
    x1, y1 = sim[0][0], sim[0][1]
    x2, y2 = sim[1][0], sim[1][1]
    x3, y3 = sim[2][0], sim[2][1]

    s = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5 + ((cx - x3) ** 2 + (cy - y3) ** 2) ** 0.5 + (
                (cx - x2) ** 2 + (cy - y2) ** 2) ** 0.5
    color.append(s)
color = np.array(color)

plt.figure(figsize=(10, 10))
plt.tripcolor(points[:, 0], points[:, 1], tri.simplices.copy(), facecolors=color, edgecolors='k')

plt.tick_params(labelbottom='off', labelleft='off', left='off', right='off', bottom='off', top='off')
ax = plt.gca()
plt.plot(83, 115, '*r')
plt.scatter(points[:, 0], points[:, 1], color='r')
#plt.show()
'''