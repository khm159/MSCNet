import torch
import tensorboard_logger 
from opts import parser
import os 

from train_epoch import train_SCG, validate_SCG
from utils import get_dataset_info, \
get_loaders, get_augmentor_SCG, check_dirs_SCG

from models.SCG import SCG
from test import test

def main_worker():
    """
    un-official implementation of MSCNet from 
    "Human action recognition by multiple spatial clues network", 
    neurocomputing, 2022

    by H. M. Kim 
    github.com/khm159/MSCNet
    """
    # === Preparation ===
    opt = parser.parse_args()
    num_classes, dataset_root, train_list_path, test_list_path = \
    get_dataset_info(opt.dataset)

    train_augmentor, test_augmentor = get_augmentor_SCG(opt)
    train_loader, test_loader = get_loaders(
        train_list_path,
        test_list_path,
        dataset_root,
        train_augmentor,
        test_augmentor,
        opt
    )

    scg_model = SCG()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(      
        params = scg_model.parameters(), 
        lr = opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )   
    expdir, logdir, ckptdir = check_dirs_SCG(opt)
    tensorboard_logger.configure(logdir)

    # the rest parameters are trained using RMSProp optimizer with a
    # learning rate of 3*10^-6 pp.15
    # there are no explanation of the lr-schedulering, 
    # and number of epoch.

    # === Spatial Attention Network Training ===
    # from pp.12
    # After training the spatial-attention module, each input image
    # can obtain an action mask Maction, which is then normalized by
    # maxmin normalization.
    # --> firstly train the Attention Model. 

    num_epoch = 100
    best_score = 0.0
    for epoch in range(num_epoch):
        print(" train {} epoch".format(epoch))
        # train epoch 
        train_SCG(
            model = scg_model,
            train_loader = train_loader,
            criterion = criterion,
            optimizer = optimizer,
            epoch = epoch,
            num_epoch = num_epoch, 
        )
        val_top1 = validate_SCG(
            model = scg_model,
            test_loader = test_loader,
            criterion = criterion
        )
        # save checkpoint... 
        cur_path = 'model.ckpt.pth'
        cur_path = os.path.join(ckptdir, cur_path)
        state_dict = {
            'state_dict': scg_model.state_dict(),
            'val_top1': val_top1.avg,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'opt': opt
        }
        torch.save(scg_model.state_dict(), cur_path)
        if val_top1.avg > best_score:
            best_score = val_top1.avg
            torch.save(
                state_dict, 
                os.path.join(
                    ckptdir,
                    'best_model.ckpt.pth'
                )
            )

    # calculate mAP 
    print(" Training Finished!")
    print(" - Best Validation Score: {:.2f}%".format(best_score))
    test(scg_model,opt.dataset)

if __name__ == "__main__":
    main_worker()

    