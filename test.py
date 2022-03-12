import torch 
from utils import get_test_loader
from tqdm import tqdm 
from sklearn import metrics
from sklearn.metrics import classification_report
from utils import AverageMeter, accuracy 


def test(model, dataset):
    """
    model test 
    """
    print(" Model Test in {}".format(dataset))

    # === Test ===
    test_loader = get_test_loader(dataset)
    model.eval()
    
    data_length = (len(test_loader) // 10)

    y_true=torch.zeros([data_length,1])
    y_pred=torch.zeros([data_length,1])
    y_pred_att = torch.zeros([data_length,1])
    i=0

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():

        for (data, target) in tqdm(test_loader):
            
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
            target = target.reshape ([batch, crop,-1])
            target = target.mean(dim=1) 
                
            pred_logits     = softmax(pred.reshape([batch,crop,-1]))
            pred_att_logits = softmax(pred_att.reshape([batch,crop,-1]))
            pred_logits     = pred_logits.mean(dim=1).squeeze(1)
            pred_att_logits = pred_att_logits.mean(dim=1).squeeze(1)
            
            y_true[i][0]     = target.item()
            y_pred[i][0]     = pred_logits.argmax(dim=1).item()
            y_pred_att[i][0] = pred_att_logits.argmax(dim=1).item()
            i+=1

            # ensemble and get accuracy 
            target = target.reshape ([batch, crop,-1])
            target = target.mean(dim=1)

        np_true=y_true.numpy()
        np_pred=y_pred.numpy()
        np_pred_att=y_pred_att.numpy()

        # scene branch 
        print(" === Scene-branch ===")
        precision = metrics.precision_score(np_true,np_pred,average='macro')
        recall = metrics.recall_score(np_true,np_pred,average='macro')
        accuracy = metrics.accuracy_score(np_true,np_pred)
        f1_score = metrics.f1_score(np_true,np_pred,average='macro')

        print('Confusion Matrix:\n',metrics.confusion_matrix(np_true, np_pred))
        line = 'Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision,recall,accuracy,f1_score)
        print(line)

        print(" === Attention-branch ===")
        precision = metrics.precision_score(np_true,np_pred_att,average='macro')
        recall = metrics.recall_score(np_true,np_pred_att,average='macro')
        accuracy = metrics.accuracy_score(np_true,np_pred_att)
        f1_score = metrics.f1_score(np_true,np_pred_att,average='macro')

        print('Confusion Matrix:\n',metrics.confusion_matrix(np_true, np_pred_att))
        line = 'Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision,recall,accuracy,f1_score)
        print(line)

