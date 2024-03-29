# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['DetMetric']

from .eval_det_iou import DetectionIoUEvaluator


class DetMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        '''
       batch: a list produced by dataloaders.
           image: np.ndarray  of shape (N, C, H, W).
           ratio_list: np.ndarray  of shape(N,2)
           polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
           ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
       preds: a list of dict produced by post process
            points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.

        自己调试补充的笔记：
        list batch:
            image: (1, 3, 736, 1280)，原始图数据，应该只有一张图N=1，检测不方便多图批量处理吧
            ratio_list: (1, 4)，与描述不符，这里应该是 (N, 4)
            polygons: (1, 4, 4, 2)，K是检测框数量，这里是gt标注框
            ignore_tags: (1, 4)，是否为难样本，一般都是False
        list[dict] preds: 一般长度只有1，直接取 preds[0]['points']
            points: (3, 4, 2)，检测出3个框
       '''
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred['points']]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)


    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        metircs = self.evaluator.combine_results(self.results)
        self.reset()
        return metircs

    def reset(self):
        self.results = []  # clear results
