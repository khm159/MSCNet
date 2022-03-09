import torch
import tensorboard_logger
from utils import AverageMeter, accuracy 

def train_SCG(model, train_loader, criterion, optimizer):

    model.train()
    semantic_top1  = AverageMeter()
    attention_top1 = AverageMeter()
    semantic_loss  = AverageMeter()
    attention_loss = AverageMeter()
    train_loss     = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        
        # ten-crop: reordering batch-wise
        batch,crop,chan,w,h = data.shape
        data = data.reshape([batch*crop,chan,w,h])
        print("reshaped data", data.shape)
        # device check
        if torch.cuda.is_available():
            data   = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()

        # calculate output 
        pred, pred_att = model(data)

        # ensemble 
        _, num_label = pred.shape
        pred     = pred.reshape([batch,crop,num_label])
        pred_att = pred_att.reshape([batch,crop,num_label])

        pred = torch.mean(pred, dim=1)
        pred_att = torch.mean(pred_att,dim=1)
        
        loss_pred = criterion(pred, target)
        loss_att  = criterion(pred_att, target)
        loss = loss_pred + loss_att
        sem_prec1, _ = accuracy(pred.data, target, topk=(1, 5))
        att_prec1, _ = accuracy(pred_att.data, target, topk=(1, 5))
        
        loss.backward()
        optimizer.step()

        semantic_loss.update(loss_pred.item(), batch*crop)
        attention_loss.update(loss_att.item(), batch*crop)
        semantic_top1.update(sem_prec1.item(), batch*crop)
        attention_top1.update(att_prec1.item(), batch*crop)
        train_loss.update(loss.item(),batch*crop)
    
    tensorboard_logger.log_value('loss_semantic_branch',semantic_loss.avg)
    tensorboard_logger.log_value('loss_attention_branch',attention_loss.avg)
    tensorboard_logger.log_value('train_top1_semantic',semantic_top1.avg)
    tensorboard_logger.log_value('train_top1_attention',attention_top1.avg)
    tensorboard_logger.log_value('total_train_loss',train_loss.avg)


def validate_SCG(model, test_loader, criterion):

    model.eval()
    semantic_top1  = AverageMeter()
    attention_top1 = AverageMeter()
    semantic_loss  = AverageMeter()
    attention_loss = AverageMeter()
    eval_loss     = AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            
            # ten-crop: reordering batch-wise
            batch,crop,chan,w,h = data.shape
            data = data.reshape([batch*crop,chan,w,h])
            
            # device check
            if torch.cuda.is_available():
                data   = data.cuda()
                target = target.cuda()
            
            # calculate output 
            pred, pred_att = model(data)

            # ensemble 
            _, num_label = pred.shape
            pred     = pred.reshape([batch,crop,num_label])
            pred_att = pred_att.reshape([batch,crop,num_label])
            pred = pred.mean(dim=1).squeeze(1)
            pred_att = pred.mean(dim=1).squeeze(1)

            loss_pred = criterion(pred, target)
            loss_att  = criterion(pred_att, target)
            loss = loss_pred + loss_att
            sem_prec1, _ = accuracy(pred.data, target, topk=(1, 5))
            att_prec1, _ = accuracy(pred_att.data, target, topk=(1, 5))

            semantic_loss.update(loss_pred.item(), batch*crop)
            attention_loss.update(loss_att.item(), batch*crop)
            semantic_top1.update(sem_prec1.item(), batch*crop)
            attention_top1.update(att_prec1.item(), batch*crop)
            eval_loss.update(loss.item(),batch*crop)
    
    tensorboard_logger.log_value('loss_semantic_branch',semantic_loss.avg)
    tensorboard_logger.log_value('loss_attention_branch',attention_loss.avg)
    tensorboard_logger.log_value('train_top1_semantic',semantic_top1.avg)
    tensorboard_logger.log_value('train_top1_attention',attention_top1.avg)
    tensorboard_logger.log_value('total_train_loss',eval_loss.avg)
        
