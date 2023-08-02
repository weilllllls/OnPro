import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from utils.rotation_transform import Rotation
from utils import my_transform as TL
from losses.loss import Supervised_NT_xent_n, Supervised_NT_xent_uni
from copy import deepcopy
from modules.OPE import OPELoss
from modules.APF import AdaptivePrototypicalFeedback


pdist = torch.nn.PairwiseDistance(p=2).cuda()

class TrainLearner(object):
    def __init__(self, model, buffer, optimizer, n_classes_num, class_per_task, input_size, args, fea_dim=128):
        self.model = model
        self.optimizer = optimizer
        self.oop_base = n_classes_num
        self.oop = args.oop
        self.n_classes_num = n_classes_num
        self.fea_dim = fea_dim
        self.classes_mean = torch.zeros((n_classes_num, fea_dim), requires_grad=False).cuda()
        self.class_per_task = class_per_task
        self.class_holder = []
        self.mixup_base_rate = args.mixup_base_rate
        self.ins_t = args.ins_t
        self.proto_t = args.proto_t

        self.buffer = buffer
        self.buffer_batch_size = args.buffer_batch_size
        self.buffer_per_class = 7

        self.OPELoss = OPELoss(self.class_per_task, temperature=self.proto_t)

        self.dataset = args.dataset
        if args.dataset == "cifar10":
            self.sim_lambda = 0.5
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.sim_lambda = 1.0
            self.total_samples = 5000
        elif args.dataset == "tiny_imagenet":
            self.sim_lambda = 1.0
            self.total_samples = 10000
        self.print_num = self.total_samples // 10

        hflip = TL.HorizontalFlipLayer().cuda()
        with torch.no_grad():
            resize_scale = (0.3, 1.0)
            color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
            resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[input_size[1], input_size[2], input_size[0]]).cuda()
            self.transform = torch.nn.Sequential(
                hflip,
                color_gray,
                resize_crop)

        self.APF = AdaptivePrototypicalFeedback(self.buffer, args.mixup_base_rate, args.mixup_p, args.mixup_lower, args.mixup_upper,
                                  args.mixup_alpha, self.class_per_task)

        self.scaler = GradScaler()

    def train_task0(self, task_id, train_loader):
        num_d = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach())

            with autocast():
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                x = x.requires_grad_()

                rot_x = Rotation(x)
                rot_x_aug = self.transform(rot_x)
                images_pair = torch.cat([rot_x, rot_x_aug], dim=0)

                rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)

                features, projections = self.model(images_pair, use_proj=True)
                projections = F.normalize(projections)

                # instance-wise contrastive loss in OCM
                features = F.normalize(features)
                dim_diff = features.shape[1] - projections.shape[1]  # 512 - 128
                dim_begin = torch.randperm(dim_diff)[0]
                dim_len = projections.shape[1]

                sim_matrix = torch.matmul(projections, features[:, dim_begin:dim_begin + dim_len].t())
                sim_matrix += torch.mm(projections, projections.t())

                ins_loss = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)
                
                if batch_idx != 0:
                    buffer_x, buffer_y = self.sample_from_buffer_for_prototypes()
                    buffer_x.requires_grad = True
                    buffer_x, buffer_y = buffer_x.cuda(), buffer_y.cuda()
                    buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)

                    proto_seen_loss, _, _, _ = self.cal_buffer_proto_loss(buffer_x, buffer_y, buffer_x_pair, task_id)
                else:
                    proto_seen_loss = 0

                z = projections[:rot_x.shape[0]]
                zt = projections[rot_x.shape[0]:]
                proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, True)

                OPE_loss = proto_new_loss + proto_seen_loss

                y_pred = self.model(self.transform(x))
                ce = F.cross_entropy(y_pred, y)

                loss = ce + ins_loss + OPE_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(
                    '==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + OPE {:.4f} = {:.6f}, {}%'
                    .format(batch_idx, ce, ins_loss, OPE_loss, loss, 100 * (num_d / self.total_samples)))

    def train_other_tasks(self, task_id, train_loader):
        num_d = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach())

            with autocast():
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                x = x.requires_grad_()
                buffer_batch_size = min(self.buffer_batch_size, self.buffer_per_class * len(self.class_holder))

                ori_mem_x, ori_mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=None)
                if batch_idx != 0:
                    mem_x, mem_y, mem_y_mix = self.APF(ori_mem_x, ori_mem_y, buffer_batch_size, self.classes_mean, task_id)
                    rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)
                    rot_sim_labels_r = torch.cat([mem_y + self.oop_base * i for i in range(self.oop)], dim=0)
                    rot_mem_y_mix = torch.zeros(rot_sim_labels_r.shape[0], 3).cuda()
                    rot_mem_y_mix[:, 0] = torch.cat([mem_y_mix[:, 0] + self.oop_base * i for i in range(self.oop)], dim=0)
                    rot_mem_y_mix[:, 1] = torch.cat([mem_y_mix[:, 1] + self.oop_base * i for i in range(self.oop)], dim=0)
                    rot_mem_y_mix[:, 2] = mem_y_mix[:, 2].repeat(self.oop)
                else:
                    mem_x = ori_mem_x
                    mem_y = ori_mem_y

                    rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)
                    rot_sim_labels_r = torch.cat([mem_y + self.oop_base * i for i in range(self.oop)], dim=0)

                mem_x = mem_x.requires_grad_()

                rot_x = Rotation(x)
                rot_x_r = Rotation(mem_x)
                rot_x_aug = self.transform(rot_x)
                rot_x_r_aug = self.transform(rot_x_r)
                images_pair = torch.cat([rot_x, rot_x_aug], dim=0)
                images_pair_r = torch.cat([rot_x_r, rot_x_r_aug], dim=0)

                all_images = torch.cat((images_pair, images_pair_r), dim=0)

                features, projections = self.model(all_images, use_proj=True)

                projections_x = projections[:images_pair.shape[0]]
                projections_x_r = projections[images_pair.shape[0]:]

                projections_x = F.normalize(projections_x)
                projections_x_r = F.normalize(projections_x_r)

                # instance-wise contrastive loss in OCM
                features_x = F.normalize(features[:images_pair.shape[0]])
                features_x_r = F.normalize(features[images_pair.shape[0]:])

                dim_diff = features_x.shape[1] - projections_x.shape[1]
                dim_begin = torch.randperm(dim_diff)[0]
                dim_begin_r = torch.randperm(dim_diff)[0]
                dim_len = projections_x.shape[1]

                sim_matrix = self.sim_lambda * torch.matmul(projections_x, features_x[:, dim_begin:dim_begin + dim_len].t())
                sim_matrix_r = self.sim_lambda * torch.matmul(projections_x_r, features_x_r[:, dim_begin_r:dim_begin_r + dim_len].t())

                sim_matrix += self.sim_lambda * torch.mm(projections_x, projections_x.t())
                sim_matrix_r += self.sim_lambda * torch.mm(projections_x_r, projections_x_r.t())

                loss_sim_r = Supervised_NT_xent_uni(sim_matrix_r, labels=rot_sim_labels_r, temperature=self.ins_t)
                loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)
                
                ins_loss = loss_sim_r + loss_sim

                y_pred = self.model(self.transform(mem_x))

                buffer_x = ori_mem_x
                buffer_y = ori_mem_y
                buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)
                proto_seen_loss, cur_buffer_z1_proto, cur_buffer_z2_proto, cur_buffer_z = self.cal_buffer_proto_loss(buffer_x, buffer_y, buffer_x_pair, task_id)

                z = projections_x[:rot_x.shape[0]]
                zt = projections_x[rot_x.shape[0]:]
                proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, True)

                OPE_loss = proto_new_loss + proto_seen_loss

                if batch_idx != 0:
                    ce = self.loss_mixup(y_pred, mem_y_mix)
                else:
                    ce = F.cross_entropy(y_pred, mem_y)

                loss = ce + ins_loss + OPE_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

            if num_d % self.print_num == 0 or batch_idx == 1:
                print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + OPE {:.4f} = {:.6f}, {}%'
                    .format(batch_idx, ce, ins_loss, OPE_loss, loss, 100 * (num_d / self.total_samples)))

    def train(self, task_id, train_loader):
        self.model.train()
        for epoch in range(1):
            if task_id == 0:
                self.train_task0(task_id, train_loader)
            else:
                self.train_other_tasks(task_id, train_loader)

    def test(self, i, task_loader):
        self.model.eval()
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list

    def test_model(self, loader, i):
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            pred = self.model(data)
            Pred = pred.data.max(1, keepdim=True)[1]
            num += data.size()[0]
            correct += Pred.eq(target.data.view_as(Pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def cal_buffer_proto_loss(self, buffer_x, buffer_y, buffer_x_pair, task_id):
        buffer_fea, buffer_z = self.model(buffer_x_pair, use_proj=True)
        buffer_z_norm = F.normalize(buffer_z)
        buffer_z1 = buffer_z_norm[:buffer_x.shape[0]]
        buffer_z2 = buffer_z_norm[buffer_x.shape[0]:]

        buffer_proto_loss, buffer_z1_proto, buffer_z2_proto = self.OPELoss(buffer_z1, buffer_z2, buffer_y, task_id)
        self.classes_mean = (buffer_z1_proto + buffer_z2_proto) / 2

        return buffer_proto_loss, buffer_z1_proto, buffer_z2_proto, buffer_z_norm

    def sample_from_buffer_for_prototypes(self):
        b_num = self.buffer.x.shape[0]
        if b_num <= self.buffer_batch_size:
            buffer_x = self.buffer.x
            buffer_y = self.buffer.y
            _, buffer_y = torch.max(buffer_y, dim=1)
        else:
            buffer_x, buffer_y, _ = self.buffer.sample(self.buffer_batch_size, exclude_task=None)

        return buffer_x, buffer_y

    def loss_mixup(self, logits, y):
        criterion = F.cross_entropy
        loss_a = criterion(logits, y[:, 0].long(), reduction='none')
        loss_b = criterion(logits, y[:, 1].long(), reduction='none')
        return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()
