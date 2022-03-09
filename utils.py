import os 
import torch 

def mkdir(path):
  if not os.path.isdir(path):
    os.mkdir(path)

def check_dirs_SCG(opt):
  """
  parsing exp condition and generate work dirs
  Args:
    opt : argument parser

  Returns:
    work_dir, log_dir, checkpoint_dir path
  """
  exp_name = "[SCG]"
  exp_name += "_[data]"+opt.dataset
  exp_name += "_[arch]"+opt.arch
  exp_name += "_[bs]"+str(opt.batch_size)
  
  workdir = 'work_dir'
  mkdir(workdir)

  expdir = os.path.join(workdir, exp_name)
  mkdir(expdir)

  logdir = os.path.join(expdir, 'log')
  mkdir(logdir)

  ckptdir = os.path.join(expdir,'ckpt')
  mkdir(ckptdir)

  return expdir, logdir, ckptdir

def get_augmentor_SCG(opt):
  """
  parsing exp condition and return train/test augmentor 
  Args:
    opt : argument parser

    Returns:
        train_augmentor, test_augmentor
  """
  # === Data Augmentation ===

  # All input images are first resized to
  # 504  504, then cropped to 10 images of the size 448  448 (four
  # corners and the central crop plus the horizontal flipped version)
  # for data augmentation. 
  # pp. 15
  import torchvision.transforms as transforms 
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  train_augmentor = transforms.Compose([
    transforms.Resize(opt.reshape_size),
    transforms.TenCrop(opt.crop_size),
    transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
    transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]),
    transforms.Lambda(lambda crops: torch.stack(crops))
  ])
  test_augmentor = transforms.Compose([
    transforms.Resize(opt.reshape_size),
    transforms.TenCrop(opt.crop_size),
    transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
    transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]),
    transforms.Lambda(lambda crops: torch.stack(crops))
  ])
  return train_augmentor, test_augmentor

def get_loaders(
  train_list, 
  test_list, 
  dataset_root, 
  train_augmentor, 
  test_augmentor,
  opt
  ):
  """
  return data loaders  
  Args:
    train_list : [list] list of the lines from the train list txt file 
    test_list  : [list] list of the lines from the test list txt file 
    data_root  : [str]  dataset root path 
    train_augmentor : [torchvision.transforms] train_augmentor 
    test_augmentor  : [torchvision.transforms] test_augmentor  

    Returns:
        train_loader, teste_loader 
  """
  from dataset import ImageAR_Dataset
  from torch.utils.data import DataLoader
  # Get Dataset and Data loader
  train_dataset = ImageAR_Dataset(
    image_list = train_list,
    image_root = dataset_root,
    transform = train_augmentor
  )
  test_dataset = ImageAR_Dataset(
    image_list = test_list,
    image_root = dataset_root,
    transform = test_augmentor
  )
  train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = opt.batch_size,
    shuffle=True,
    num_workers=opt.workers,
    pin_memory=True
  )
  test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = opt.batch_size,
    shuffle=False,
    num_workers=opt.workers,
    pin_memory=True
  )
  return train_loader, test_loader

def _load_txt_from_path(path):
  """
  open txt list file from path 
  Args: 
    list_file_path  : [str] data list file path

  Returns:
    list of the line from list text file
  """
  with open(path, 'r') as f:
    lines = f.readlines()
  return lines

def get_list(path):
  return _load_txt_from_path(path)

def get_dataset_info(dataset_name):
  print('-- selected dataset : ', dataset_name)
  if dataset_name == 'stanford40':
    num_classes = 40 
    dataset_root = 'data/stanford40'
    train_list_path = os.path.join(
        dataset_root,
        dataset_name+'_train.txt'
    )
    test_list_path = os.path.join(
        dataset_root,
        dataset_name+'_test.txt'
    )
    image_root = os.path.join(
      dataset_root, 'JPEGImages'
    )
  else:
      NotImplementedError
  return num_classes, image_root, train_list_path, test_list_path
  
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res