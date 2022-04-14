# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/DB_loss.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import nn

from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Layer):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 **kwargs):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # 声明不同的损失函数
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio)

    def forward(self, predicts, labels):
        predict_maps = predicts['maps']
        label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[
            1:]
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]
        # 1. 针对文本预测概率图，使用二值交叉熵损失函数
        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map,
                                         label_shrink_mask)
        # 2. 针对文本预测阈值图使用L1距离损失函数
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map,
                                           label_threshold_mask)
        # 3. 针对文本预测二值图，使用dice loss损失函数
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map,
                                          label_shrink_mask)
        # 4. 不同的损失函数乘上不同的权重
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        loss_all = loss_shrink_maps + loss_threshold_maps \
                   + loss_binary_maps
        losses = {'loss': loss_all, \
                  "loss_shrink_maps": loss_shrink_maps, \
                  "loss_threshold_maps": loss_threshold_maps, \
                  "loss_binary_maps": loss_binary_maps}
        return losses
