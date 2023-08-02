import datetime

import numpy as np
import torch
from torch.optim import Adam
from experiment.dataset import get_data
from models.buffer import Buffer
from train import TrainLearner
from models.Resnet18 import resnet18
from utils.util import compute_performance


def multiple_run(args):
    test_all_acc = torch.zeros(args.run_nums)

    accuracy_list = []
    for run in range(args.run_nums):
        tmp_acc = []
        print('=' * 100)
        print(f"-----------------------------run {run} start--------------------------")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('=' * 100)
        data, class_num, class_per_task, task_loader, input_size = get_data(args.dataset, args.batch_size, args.n_workers)
        args.n_classes = class_num
        buffer = Buffer(args, input_size).cuda()

        model = resnet18(class_num).cuda()
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
        agent = TrainLearner(model, buffer, optimizer, class_num, class_per_task, input_size, args)

        for i in range(len(task_loader)):
            print(f"-----------------------------run {run} task id:{i} start training-----------------------------")

            agent.train(i, task_loader[i]['train'])
            acc_list = agent.test(i, task_loader)
            tmp_acc.append(acc_list)

        test_accuracy = acc_list.mean()
        test_all_acc[run] = test_accuracy
        accuracy_list.append(np.array(tmp_acc))

        print('=' * 100)
        print("{}th run's Test result: Accuracy: {:.2f}%".format(run, test_accuracy))
        print('=' * 100)

    accuracy_array = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
    print('=' * 100)
    print(f"total {args.run_nums}runs test acc results: {test_all_acc}")
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    print('=' * 100)
