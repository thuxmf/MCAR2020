import os
from glob import glob
import time
import itertools
from speech_content import *
from dataloader import *
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LandmarkLoss(nn.Module):
    """face: 17, left brow: 5, right brow: 5, nose: 9, left eye: 6, right eye: 6, mouth: 20"""
    def __init__(self, lambda_c, use_mouth_weight=False, use_motion_loss=False):
        super(LandmarkLoss, self).__init__()
        self.use_mouth_weight = use_mouth_weight
        self.use_motion_loss = use_motion_loss
        self.lambda_c = lambda_c

    def __call__(self, target, prediction):
        # calculate mouth weight
        w = torch.abs(target[:, 66*3+1] - target[:, 62*3+1])
        w = torch.tensor([1.]).to(device) / (w * 4. + .1)
        w = w.unsqueeze(1)
        mouth_weight = torch.ones(target.shape[0], 204).to(device)
        mouth_weight[:, 48*3:] = torch.cat([w]*60, dim=1)
        mouth_weight = mouth_weight.detach().clone().requires_grad_(False)

        if self.use_mouth_weight:
            loss = torch.mean(torch.abs(target - prediction) * mouth_weight)
        else:
            loss = F.l1_loss(prediction, target)

        if self.use_motion_loss:
            pred_motion = prediction[:-1] - prediction[1:]
            target_motion = target[:-1] - target[1:]
            loss += F.l1_loss(pred_motion, target_motion)

        loss += F.l1_loss(Laplacian(prediction, False), Laplacian(target, False)) * self.lambda_c

        return loss


class Model(object):
    def __init__(self, args):
        super(Model, self).__init__()
        self.phase = args.phase
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.decay_flag = args.decay_flag

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.lambda_c = args.lambda_c
        self.lambda_s = args.lambda_s
        self.mu_s = args.mu_s

        self.result_dir = args.result_dir
        self.resume = args.resume
        self.use_mouth_weight = args.use_mouth_weight
        self.use_motion_loss = args.use_motion_loss

        print()

        print("##### Information #####")
        print("# phase: ", self.phase)
        print("# dataset: ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch: ", self.iteration)
        print("# use_mouth_weight: ", 'True' if self.use_mouth_weight else 'False')
        print("# use_motion_loss: ", 'True' if self.use_motion_loss else 'False')

        print()

        print("##### Parameters #####")
        print("# lambda_c: ", self.lambda_c)
        print("# lambda_s: ", self.lambda_s)
        print("# mu_s: ", self.mu_s)

        print()

    def build_model(self):
        self.train_datafolder = DataFolder(os.path.join(self.dataset, 'train'))
        self.train_dataloader = DataLoader(self.train_datafolder, batch_size=self.batch_size, shuffle=True)
        self.test_datafolder = DataFolder(os.path.join(self.dataset, 'test'))
        self.test_dataloader = DataLoader(self.test_datafolder, batch_size=self.batch_size, shuffle=True)
        self.speech_content = SpeechContent()
        self.lms_loss = LandmarkLoss(self.lambda_c, self.use_mouth_weight, self.use_motion_loss)

        self.content_optim = torch.optim.Adam(itertools.chain(self.speech_content.parameters()), lr=self.lr,
                                              betas=(0.5, 0.999), weight_decay=self.weight_decay)

    def load_model(self, map_location=None):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        model_list.sort()
        start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
        model_path = os.path.join(os.path.join(self.result_dir, self.dataset, 'model'), self.dataset + '_params_%07d.pt' % start_iter)
        if map_location is None:
            params = torch.load(model_path)
        else:
            params = torch.load(model_path, map_location=map_location)
        return params, start_iter

    def train(self):
        self.speech_content.to(device)
        self.lms_loss.to(device)
        start_iter = 1
        if self.resume:
            params, start_iter = self.load_model()
            self.speech_content.load_state_dict(params['content'])

            print(" [*] resume Load SUCCESS")
            if start_iter > (self.iteration // 2):
                self.content_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * \
                                                            (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            self.speech_content.train()
            if step > (self.iteration // 2):
                self.content_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            train_data, test_data = None, None
            while train_data is None or train_data['mel'].size(1) > 2000:
                try:
                    train_data, _ = train_data_iter.next()
                except:
                    train_data_iter = iter(self.train_dataloader)
                    train_data, _ = train_data_iter.next()

            A_t = train_data['mel'].squeeze(0).float()
            A_t = torch.cat([A_t, torch.zeros(30-1, 40)], dim=0).to(device)
            _A_t = tensor_seq(A_t, 30)
            q = train_data['landmarks'][0].float().to(device)
            q = q.view(1, -1)
            hat_p_t = torch.cat(train_data['landmarks'], dim=0).view(-1, 204).float().to(device)  # (n, 204)

            self.content_optim.zero_grad()
            p_t = self.speech_content(_A_t, q)[:hat_p_t.size(0)]  # (n, 204)
            content_loss = self.lms_loss(hat_p_t, p_t)
            content_loss.backward()
            self.content_optim.step()

            if step % 10000 == 0:
                p_t, hat_p_t = p_t.view(-1, 68, 3).detach().cpu().numpy(), hat_p_t.view(-1, 68, 3).detach().cpu().numpy()
                p_t = self.solve_inv_lip(p_t)
                out = cv2.VideoWriter(os.path.join(self.result_dir, self.dataset, 'train', 'train_%d.mp4' % step), cv2.VideoWriter_fourcc(*'XVID'), 25, (224 * 2, 224))
                for t in range(len(train_data['landmarks'])):
                    frame_1, frame_2 = np.zeros((224, 224, 3), dtype=np.uint8) + 255, np.zeros((224, 224, 3),
                                                                                               dtype=np.uint8) + 255
                    lands = hat_p_t[t]
                    # print frame
                    draw(frame_1, lands, [0, 0, 0])  # ground truth, black
                    lands = p_t[t]
                    # print frame
                    draw(frame_2, lands)
                    frame = np.concatenate((frame_1, frame_2), axis=1)
                    out.write(frame)
                out.release()
                cv2.destroyAllWindows()
                params = {}
                params['content'] = self.speech_content.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

            if step % self.print_freq == 0:
                print("[%5d/%5d] time: %4.4f loss per frame: %.8f, # frames: %d" %
                      (step, self.iteration, time.time() - start_time, content_loss / hat_p_t.size(0), hat_p_t.size(0)))

            if step % self.save_freq == 0:
                params = {}
                params['content'] = self.speech_content.state_dict()
                torch.save(params, os.path.join(os.path.join(self.result_dir, self.dataset, 'model')
                                                             , self.dataset + '_params_%07d.pt' % step))

    def test(self):
        data_iter = iter(self.test_dataloader)
        params, _ = self.load_model('cpu')
        self.speech_content.load_state_dict(params['content'])
        self.speech_content.eval()
        print(" [*] model Load SUCCESS")
        id = 0
        while 1:
            try:
                test_data, _ = data_iter.next()
            except:
                break
            id += 1
            out = cv2.VideoWriter(os.path.join(self.result_dir, self.dataset, 'test', 'test_%d.mp4' % id), cv2.VideoWriter_fourcc(*'XVID'), 25, (224*2, 224))
            A_t = data['mel'].squeeze(0).float()
            A_t = torch.cat([A_t, torch.zeros(30 - 1, 40)], dim=0).to(device)
            _A_t = tensor_seq(A_t, 30)
            q = test_data['landmarks'][0].float().to(device)
            q = q.view(1, -1)
            hat_p_t = torch.cat(test_data['landmarks'], dim=0).view(-1, 204).float().to(device)
            p_t = self.speech_content(_A_t, q)[:hat_p_t.size(0)].view(-1, 204)
            loss = self.lms_loss(hat_p_t, p_t)

            p_t = p_t.view(-1, 68, 3).detach().cpu().numpy()
            p_t = self.solve_inv_lip(p_t)
            hat_p_t = hat_p_t.view(-1, 68, 3)

            for t in range(len(test_data['landmarks'])):
                frame_1, frame_2 = np.zeros((224, 224, 3), dtype=np.uint8) + 255, np.zeros((224, 224, 3),
                                                                                           dtype=np.uint8) + 255
                # print frame
                lands = hat_p_t[t]
                # print frame
                draw(frame_1, lands, [0, 0, 0])  # ground truth, black
                lands = p_t[t]
                # print frame
                draw(frame_2, lands)
                frame = np.concatenate((frame_1, frame_2), axis=1)
                out.write(frame)
            print('%d.mp4: ' % id, loss)
            np.save(os.path.join(self.result_dir, self.dataset, 'test', 'test_%d.npy' % id), p_t)
            out.release()
            cv2.destroyAllWindows()

    def solve_inv_lip(self, p_t_numpy):
        p_t_numpy = p_t_numpy.reshape(-1, 68, 3)
        for i in range(p_t_numpy.shape[0]):
            fls = p_t_numpy[i]
            mouth_area = area_of_signed_polygon(fls[list(range(60, 68)), :2])
            if mouth_area < 0:
                p_t_numpy[j, 65 * 3:66 * 3] = 0.5 * (p_t_numpy[j, 63 * 3:64 * 3] + p_t_numpy[j, 65 * 3:66 * 3])
                p_t_numpy[j, 63 * 3:64 * 3] = p_t_numpy[j, 65 * 3:66 * 3]
                p_t_numpy[j, 66 * 3:67 * 3] = 0.5 * (p_t_numpy[j, 62 * 3:63 * 3] + p_t_numpy[j, 66 * 3:67 * 3])
                p_t_numpy[j, 62 * 3:63 * 3] = p_t_numpy[j, 66 * 3:67 * 3]
                p_t_numpy[j, 67 * 3:68 * 3] = 0.5 * (p_t_numpy[j, 61 * 3:62 * 3] + p_t_numpy[j, 67 * 3:68 * 3])
                p_t_numpy[j, 61 * 3:62 * 3] = p_t_numpy[j, 67 * 3:68 * 3]
                p = max([i - 1, 0])
                p_t_numpy[j, 55 * 3 + 1:59 * 3 + 1:3] = p_t_numpy[j, 64 * 3 + 1:68 * 3 + 1:3] \
                                                                    + p_t_numpy[p, 55 * 3 + 1:59 * 3 + 1:3] \
                                                                    - p_t_numpy[p, 64 * 3 + 1:68 * 3 + 1:3]
                p_t_numpy[j, 59 * 3 + 1:60 * 3 + 1:3] = p_t_numpy[j, 60 * 3 + 1:61 * 3 + 1:3] \
                                                                    + p_t_numpy[p, 59 * 3 + 1:60 * 3 + 1:3] \
                                                                    - p_t_numpy[p, 60 * 3 + 1:61 * 3 + 1:3]
                p_t_numpy[j, 49 * 3 + 1:54 * 3 + 1:3] = p_t_numpy[j, 60 * 3 + 1:65 * 3 + 1:3] \
                                                                    + p_t_numpy[p, 49 * 3 + 1:54 * 3 + 1:3] \
                                                                    - p_t_numpy[p, 60 * 3 + 1:65 * 3 + 1:3]
        return p_t_numpy





