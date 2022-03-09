import argparse
parser = argparse.ArgumentParser(description=\
"Pytorch implementation of 'Human action recognition by multiple spatial clues network'")

parser.add_argument('--dataset', type=str, default='stanford40', choices=['stanford40','voc','willow','ppmi'])
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="efficientnet-b3")
parser.add_argument('--num_classes', type=int, default=40)
parser.add_argument('--pretrained', type=bool, default=True)
# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=3*(10**-6), type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# ========================= Runtime Configs ==========================
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# ========================= Data-related Configs ==========================
parser.add_argument('--reshape_size', default=504, type=int, metavar='N',
                    help='size of the image in the resizing step (default: 504)')
parser.add_argument('--crop_size', default=448, type=int, metavar='N',
                    help='size of the image in the resizing step (default: 448)')