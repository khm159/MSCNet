"""
- Spatial Clues Generation - 

[from original paper pp.3-4]

1. Spatial attention Module 
   A spatialattention module is proposed to generate 
   the spatial attention regions with attention degrees.
2. Body-detection Module 
   A body-detection module generates positive and 
   negative human training samples from the original 
   action dataset, and then trains a body detector to
   obtain the human body locations.
3. Semantics-selection Module 
   a semantics-selection module is proposed to select 
   the action-specific parts from spatial attention regions.
"""
import torch 
import torch.nn as nn 
from models.efficientnet.efficientnet import EfficientNet

class SCG(nn.Module):
    def __init__(self, num_classes=40, drop_out_rate=0.2):
        super().__init__()
        # It is not explained whether the FC-layers of the attention branch 
        # are weight-shared or not.
        # I think they are not weight-shared.
        # So the FC-layers are implemented seperately. 

        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b3')
        self.feature_extractor._fc = nn.Identity()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._fc = nn.Linear(
            1536, num_classes
            )
        self._fc_att = nn.Linear(
            1536, num_classes
        )
        self.SAM = SAM()
        self._dropout = nn.Dropout(drop_out_rate) # from effiient-net global params
        

    def forward(self, x):
        """
        "All input images are first resized to
        504  504, then cropped to 10 images of the size 448  448 (four
        corners and the central crop plus the horizontal flipped version)
        for data augmentation. Thus the sizes of Fbackbone and Flast are
        14  14  384 and 14  14  1536, respectively. The size of
        Maction is 14  14, and each value in Maction corresponds to a spatial
        region of size 32  32 in the input image"

        pp.15 implememtation detail : from original paper
        """
        # print('input tensor', x.shape) # should be [b, 3, 448, 448]

        # extract feature 
        F_backbone, F_last = self.feature_extractor(x)

        # print('F_backbone', F_backbone.shape) # [b, 384, 14, 14]  checked.  w x h x c1
        # print('F_last', F_last.shape)         # [b, 1536, 14, 14] checked.  w x h x c2 

        # scene-branch 
        pred = self._fc(
            self._dropout(
                self._avg_pooling(F_last).flatten(start_dim=1)
            )
        )

        # attention-branch 
        f_attention = self.SAM(F_backbone, F_last)
        pred_att  = self._fc_att(
            self._dropout(
                self._avg_pooling(f_attention).flatten(start_dim=1)
        ))

        return pred, pred_att

class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Spatial Attention Module of SCG 
        """
        self.SAL = nn.Conv2d(in_channels = 384, out_channels = 1, kernel_size = 1)

    def forward(self, f_back, f_last):
        """
        In the attentionbranch, a Spatial Attention Layer (SAL) with 1  1 kernels is adopted
        on top of Fbackbone to generate an action mask shape: [w, h, 1]. 
        Then Maction is acted on Flast to select spatial attention features as follows:
        Fattention= matmul(Flast, Maction)

        pp.12 3.1.1 The spatial-attention module 
        """
        m_action = self.SAL(f_back)
        return f_last*m_action

if __name__ == "__main__":

    model = SCG()
    dummy = torch.zeros([1, 3, 448, 448])
    out = model(dummy)




