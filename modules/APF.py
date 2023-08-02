import torch.nn as nn
import torch
import torch.nn.functional as F
from kornia.augmentation import RandomMixUpV2
import numpy as np
import itertools


class AdaptivePrototypicalFeedback(nn.Module):
    def __init__(self, buffer, mixup_base_rate, mixup_p, mixup_lower, mixup_upper, mixup_alpha,
                 class_per_task):
        super(AdaptivePrototypicalFeedback, self).__init__()
        self.buffer = buffer
        self.class_per_task = class_per_task
        self.mixup_base_rate = mixup_base_rate
        self.mixup_p = mixup_p
        self.mixup_lower = mixup_lower
        self.mixup_upper = mixup_upper
        self.mixup_alpha = mixup_alpha
        self.mixup = RandomMixUpV2(p=mixup_p, lambda_val=(mixup_lower, mixup_upper),
                                   data_keys=["input", "class"]).cuda()

    def forward(self, mem_x, mem_y, buffer_batch_size, classes_mean, task_id):
        base_rate = self.mixup_base_rate
        base_sample_num = int(buffer_batch_size * base_rate)

        indices = torch.from_numpy(np.random.choice(mem_x.size(0), base_sample_num, replace=False)).cuda()
        mem_x_base = mem_x[indices]
        mem_y_base = mem_y[indices]

        mem_x_base_mix, mem_y_base_mix = self.mixup(mem_x_base, mem_y_base)

        prob_sample_num = buffer_batch_size - base_sample_num
        if prob_sample_num != 0:
            nonZeroRows = torch.abs(classes_mean).sum(dim=1) > 0
            ZeroRows = torch.abs(classes_mean).sum(dim=1) == 0
            class_num = classes_mean.shape[0]
            nonZero_class = torch.arange(class_num)[nonZeroRows]
            Zero_class = torch.arange(class_num)[ZeroRows]

            classes_mean = classes_mean[nonZeroRows]

            dis = torch.pdist(classes_mean)  # K*(K-1)/2

            sample_p = F.softmax(1 / dis, dim=0)

            mix_x_by_prob, mix_y_by_prob = self.make_mix_pair(sample_p, prob_sample_num, nonZero_class, Zero_class,
                                                              task_id)

            mem_x = torch.cat([mem_x_base_mix, mix_x_by_prob])
            mem_y_mix = torch.cat([mem_y_base_mix, mix_y_by_prob])

            origin_mem_y, mix_mem_y, mix_lam = mem_y_mix[:, 0], mem_y_mix[:, 1], mem_y_mix[:, 2]
            new_mem_y = (1 - mix_lam) * origin_mem_y + mix_lam * mix_mem_y
            mem_y = new_mem_y
        else:
            mem_x = mem_x_base_mix
            origin_mem_y, mix_mem_y, mix_lam = mem_y_base_mix[:, 0], mem_y_base_mix[:, 1], mem_y_base_mix[:, 2]
            new_mem_y = (1 - mix_lam) * origin_mem_y + mix_lam * mix_mem_y
            mem_y = new_mem_y
            mem_y_mix = mem_y_base_mix

        return mem_x, mem_y, mem_y_mix

    def make_mix_pair(self, sample_prob, prob_sample_num, nonZero_class, Zero_class, current_task_id):
        start_i = 0
        end_i = (current_task_id + 1) * self.class_per_task
        sample_num_per_class_pair = (sample_prob * prob_sample_num).round()
        diff_num = int((prob_sample_num - sample_num_per_class_pair.sum()).item())
        if diff_num > 0:
            add_idx = torch.randperm(sample_num_per_class_pair.shape[0])[:diff_num]
            sample_num_per_class_pair[add_idx] += 1
        elif diff_num < 0:
            reduce_idx = torch.nonzero(sample_num_per_class_pair, as_tuple=True)[0]
            reduce_idx_ = torch.randperm(reduce_idx.shape[0])[:-diff_num]
            reduce_idx = reduce_idx[reduce_idx_]
            sample_num_per_class_pair[reduce_idx] -= 1

        assert sample_num_per_class_pair.sum() == prob_sample_num

        x_indices = torch.arange(self.buffer.x.shape[0])
        y_indices = torch.arange(self.buffer.y.shape[0])
        y = self.buffer.y
        _, y = torch.max(y, dim=1)

        class_x_list = []
        class_y_list = []
        class_id_map = {}
        for task_id in range(start_i, end_i):
            if task_id in Zero_class:
                continue
            indices = (y == task_id)
            if not any(indices):
                continue

            class_x_list.append(x_indices[indices])
            class_y_list.append(y_indices[indices])
            class_id_map[task_id] = len(class_y_list) - 1

        mix_images = []
        mix_labels = []

        for idx, class_pair in enumerate(itertools.combinations(nonZero_class.tolist(), 2)):
            n = int(sample_num_per_class_pair[idx].item())
            if n == 0:
                continue
            first_class_y = class_pair[0]
            second_class_y = class_pair[1]

            if first_class_y not in class_id_map:
                first_class_y = np.random.choice(list(class_id_map.keys()), 1)[0]
                first_class_y = int(first_class_y)
            if second_class_y not in class_id_map:
                second_class_y = np.random.choice(list(class_id_map.keys()), 1)[0]
                second_class_y = int(second_class_y)

            first_class_idx = class_id_map[first_class_y]
            second_class_idx = class_id_map[second_class_y]

            first_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[first_class_idx].tolist(), n)).long()
            second_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[second_class_idx].tolist(), n)).long()

            first_class_x = self.buffer.x[first_class_sample_idx]
            second_class_x = self.buffer.x[second_class_sample_idx]

            mix_pair, mix_lam = self.mixup_by_input_pair(first_class_x, second_class_x, n)
            mix_y = torch.zeros(n, 3)
            mix_y[:, 0] = first_class_y
            mix_y[:, 1] = second_class_y
            mix_y[:, 2] = mix_lam

            mix_images.append(mix_pair)
            mix_labels.append(mix_y)

        mix_images_by_prob = torch.cat(mix_images).cuda()
        mix_labels_by_prob = torch.cat(mix_labels).cuda()

        return mix_images_by_prob, mix_labels_by_prob

    def mixup_by_input_pair(self, x1, x2, n):
        if torch.rand([]) <= self.mixup_p:
            lam = torch.from_numpy(np.random.beta(self.mixup_alpha, self.mixup_alpha, n)).cuda()
            lam_ = lam.unsqueeze(0).unsqueeze(0).unsqueeze(0).view(-1, 1, 1, 1)
        else:
            lam = 0
            lam_ = 0
        lam = torch.tensor(lam, dtype=x1.dtype)
        lam_ = torch.tensor(lam_, dtype=x1.dtype)
        image = (1 - lam_) * x1 + lam_ * x2
        return image, lam
