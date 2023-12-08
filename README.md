# OnPro
Official implementation of ICCV 2023 paper "[Online Prototype Learning for Online Continual Learning](http://arxiv.org/abs/2308.00301)".



## Usage
### Requirements
* python==3.8
* pytorch==1.9.0
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Training
**CIFAR-10**
```
python main.py --buffer_size 200 --mixup_p 0.6 --mixup_base_rate 0.75 --gpu_id 0
```

**CIFAR-100**
```
python main.py --dataset cifar100 --buffer_size 500 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 0
```

**TinyImageNet**
```
python main.py --dataset tiny_imagenet --buffer_size 1000 --mixup_p 0.2 --mixup_base_rate 0.9 --gpu_id 0
```

## Citation
If you found this code or our work useful, please cite us:

```bibtex
@inproceedings{onpro,
  title={Online prototype learning for online continual learning},
  author={Wei, Yujie and Ye, Jiaxin and Huang, Zhizhong and Zhang, Junping and Shan, Hongming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18764--18774},
  year={2023}
}
```
