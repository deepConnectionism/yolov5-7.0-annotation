# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """用在ComputeLoss类中
    标签平滑操作  [1, 0]  =>  [0.95, 0.05]
    :params eps: 平滑参数
    :return positive, negative label smoothing BCE targets  两个值分别代表正样本和负样本的标签取值
            原先的正样本=1 负样本=0 改为 正样本=1.0 - 0.5 * eps  负样本=0.5 * eps
    """
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """ BCE函数的一个替代，是yolov5作者的一个实验性的函数
    用在ComputeLoss类的__init__函数中
    https://github.com/ultralytics/yolov5/issues/1030
    The idea was to reduce the effects of false negatives (missing labels), which can occur often in COCO and other datasets.
    使用起来直接在ComputeLoss类的__init__函数中替代传统的BCE函数即可
    """
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        # dx = [-1, 1]  当pred=1 true=0时(网络预测说这里有个obj但是gt说这里没有), dx=1 => alpha_factor=0 => loss=0
        # 这种就是检测成正样本了但是检测错了（false positive）或者missing label的情况 这种情况不应该过多的惩罚->loss=0
        dx = pred - true  # reduce only missing label effects
        # 如果采样绝对值的话 会减轻pred和gt差异过大而造成的影响
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    """FocalLoss损失函数来自 Kaiming He在2017年发表的一篇论文：Focal Loss for Dense Object Detection. 
    这篇论文设计的主要思路: 希望那些hard examples对损失的贡献变大，使网络更倾向于从这些样本上学习。防止由于easy examples过多，主导整个损失函数。
    优点：
        解决了 one-stage object detection 中图片中正负样本（前景和背景）不均衡的问题； 降低简单样本的权重，使损失函数更关注困难样本；

    用在代替原本的BCEcls（分类损失）和BCEobj（置信度损失）
    论文: https://arxiv.org/abs/1708.02002
    https://blog.csdn.net/qq_38253797/article/details/116292496
    TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
    """
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss() 定义为多分类交叉熵损失函数
        self.gamma = gamma # 参数gamma  用于削弱简单样本对loss的贡献程度
        self.alpha = alpha # 参数alpha  用于平衡正负样本个数不均衡的问题
        self.reduction = loss_fcn.reduction # self.reduction: 控制FocalLoss损失输出模式 sum/mean/none  默认是Mean
        # focalloss中的BCE函数的reduction='None'  BCE不使用Sum或者Mean 
        # 需要将Focal loss应用于每一个样本之中
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        # 正常BCE的loss:   loss = -log(p_t)
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma # 这里代表Focal loss中的指数项
        loss *= alpha_factor * modulating_factor # 返回最终的loss=BCE * 两个参数  (看看公式就行了 和公式一模一样)
        # 最后选择focalloss返回的类型 默认是mean
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    """QFocalLoss损失函数来自20年的一篇文章： 
             Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection.
    博客：https://zhuanlan.zhihu.com/p/147691786
    用来代替FocalLoss
    QFocalLoss 来自General Focal Loss论文: https://arxiv.org/abs/2006.04388
    Args:
        nn (_type_): _description_
    """
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False # 后面筛选置信度损失正样本的时候是否先对iou排序

    # Compute losses
    def __init__(self, model, autobalance=False):
        # 获取模型所在的设备
        device = next(model.parameters()).device  # get model device
        # 获取模型的超参数
        h = model.hyp  # hyperparameters

        # Define criteria
        # 定义分类损失和置信度损失
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # 标签平滑  eps=0代表不做标签平滑-> cp=1 cn=0 /  eps!=0代表做标签平滑 
        # cp代表正样本的标签值 cn代表负样本的标签值
        # 请参考：Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss 代替原本的BCEcls和BCEobj
        g = h['fl_gamma']  # focal loss gamma; g=0 代表不用focal loss
        if g > 0:
            # 如果 g>0 将分类损失和置信度损失(BCE)都换成 FocalLoss 损失函数
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # m: 返回的是模型的3个检测头分别对应产生的3个输出特征图
        m = de_parallel(model).model[-1]  # Detect() module
        """self.balance  用来实现 obj,box,cls loss 之间权重的平衡
        {3: [4.0, 1.0, 0.4]} 表示有三个layer的输出，第一个layer的weight是4.0，第二个1.0，第三个以此类推。
        如果有5个layer的输出，那么权重分别是[4.0, 1.0, 0.25, 0.06, 0.02]
        """
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # 三个检测头的下采样率 m.stride: [8, 16, 32]  .index(16): 求出下采样率 stride=16 的索引
        # 这个参数会用来自动计算更新 3 个 feature map 的置信度损失系数 self.balance
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors 每个grid_cell的anchor数量 = 3
        self.nc = m.nc  # number of classes 数据集的总类别 = 80
        self.nl = m.nl  # number of layers 检测头的个数 = 3
        # anchors: 形状 [3, 3, 2]  代表 3 个 feature map 每个 feature map 上有 3 个 anchor(w,h)
        # 这里的 anchors 尺寸是相对 feature map 的
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    # build_targets 函数用于获得在训练时计算 loss 所需要的目标框，也即正样本。与yolov3/v4的不同，yolov5支持跨网格预测。
    # 对于任何一个 GT bbox，三个预测特征层上都可能有先验框匹配，所以该函数输出的正样本框比传入的 targets （GT框）数目多
    # 具体处理过程:
    # (1)首先通过 bbox 与当前层 anchor 做一遍过滤。对于任何一层计算当前 bbox 与当前层 anchor 的匹配程度，不采用IoU，而采用shape比例。如果anchor与bbox的宽高比差距大于4，则认为不匹配，此时忽略相应的bbox，即当做背景;
    # (2)根据留下的bbox，在上下左右四个网格四个方向扩增采样（即对 bbox 计算落在的网格所有 anchors 都计算 loss(并不是直接和 GT 框比较计算 loss) )
    # 注意此时落在网格不再是一个，而是附近的多个，这样就增加了正样本数。
    # yolov5 没有 conf 分支忽略阈值(ignore_thresh)的操作，而yoloy3/v4有。
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        """所有GT筛选相应的anchor正样本
        这里通过
        p       : list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
        targets : targets.shape[314, 6] 

        Args:
            p (_type_): p[i]的作用只是得到每个 feature map 的shape
                        预测框 由模型构建中的三个检测头 Detector 返回的三个 yolo 层的输出
                        tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                        如: list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
                        [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                        可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
            targets (_type_): 数据增强后的真实框 [63, 6] [num_target,  image_index+class+xywh] xywh为归一化后的框

        Returns:
            tcls: 表示这个target所属的class index
            tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            indices: b: 表示这个target属于的image index
                     a: 表示这个target使用的anchor index
                    gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                    gi: 表示这个网格的左上角x坐标
            anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        """
        # na = 3 ; nt = 314
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain.shape=[7]
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        # ai.shape = (na,nt) 生成anchor索引
        # anchor索引，后面有用，用于表示当前bbox和当前层的哪个anchor匹配
        # 需要在3个anchor上都进行训练 所以将标签赋值na=3个 
        #  ai代表3个anchor上在所有的target对应的anchor索引 就是用来标记下当前这个target属于哪个anchor
        # [1, 3] -> [3, 1] -> [3, 314]=[na, nt]   三行  第一行63个0  第二行63个1  第三行63个2
        # ai.shape  =[3, 314]
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # [314, 6] [3, 314] -> [3, 314, 6] [3, 314, 1] -> [3, 314, 7]  7: [image_index+class+xywh+anchor_index]
        # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
        # 先假设所有的target都由这层的三个anchor进行检测(复制三份)  再进行筛选  并将ai加进去标记当前是哪个anchor的target
        # targets.shape = [3, 314, 7]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
        # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
        # 设置网格中心偏移量
        g = 0.5  # bias
        # 附近的4个框
        # 以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets
        # 对每个检测层进行处理 
        # 遍历三个feature 筛选gt的anchor正样本
        for i in range(self.nl): #  self.nl: number of detection layers   Detect的个数 = 3
            # anchors: 当前feature map对应的三个anchor尺寸(相对feature map)  [3, 2]
            anchors, shape = self.anchors[i], p[i].shape
            # gain: 保存每个输出feature map的宽高 -> gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]] 
            # [1, 1, 1, 1, 1, 1, 1] -> [1, 1, 112, 112, 112,112, 1]=image_index+class+xywh+anchor_index
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # t.shape = [3, 314, 7]  将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度
            #    [3, 314, image_index+class+xywh+anchor_index]
            t = targets * gain  # shape(3,n,7)
            if nt: # 如果有目标就开始匹配
                # Matches
                # 所有的gt与当前层的三个anchor的宽高比(w/w  h/h)
                # r.shape = [3, 314, 2]
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
                # torch.max(r, 1. / r)=[3, 314, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
                # j.shape = [3, 314]  False: 当前anchor是当前gt的负样本  True: 当前anchor是当前gt的正样本
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # yolov3 v4的筛选方法: wh_iou  GT与anchor的wh_iou超过一定的阈值就是正样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                # 根据筛选条件j, 过滤负样本, 得到所有gt的anchor正样本(batch_size张图片)
                # 知道当前gt的坐标 属于哪张图片 正样本对应的idx 也就得到了当前gt的正样本anchor
                # t: [3, 314, 7] -> [555, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                t = t[j]  # filter

                # Offsets 筛选当前格子周围格子 找到 2 个离target中心最近的两个格子  
                # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 
                # 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个
                # 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                # grid xy 取target中心的坐标xy(相对feature map左上角的坐标)
                # gxy.shape = [555, 2]
                gxy = t[:, 2:4]  # grid xy
                # inverse  得到target中心点相对于右下角的坐标  gain[[2, 3]]为当前feature map的wh
                # gxi.shape = [555, 2]
                gxi = gain[[2, 3]] - gxy  # inverse
                # 筛选中心坐标距离当前grid_cell的左、上方偏移小于g=0.5 
                # 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [555] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [555] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # 筛选中心坐标距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [555] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [555] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # j.shape=[5, 555]
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*555 都不在边上等号成立
                # t: [555, 7] -> 复制 5 份target[5, 555, 7]  分别对应当前格子和左上右下格子5个格子
                # 使用 j 筛选后 t 的形状: [1659, 7] 
                t = t.repeat((5, 1, 1))[j]
                # flow.zeros_like(gxy)[None]: [1, 555, 2]   off[:, None]: [5, 1, 2]  => [5, 555, 2]
                # 得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界
                # （左右上下边框）的偏移量，然后通过 j 筛选最终 offsets 的形状是 [1659, 2]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # bc.shape = [1659, 2]
            # gxy.shape = [1659, 2]
            # gwh.shape  = [1659, 2]
            # a.shape = [1659, 1]
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            # a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            # a.shape = [1659]
            # (b, c).shape = [1659, 2]
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            # 预测真实框的网格所在的左上角坐标(有左上右下的网格)  
            # gij.shape = [1659, 2]
            
            # 这里的拆分我们可以用下面的示例代码来进行解释：

            # x = torch.randn(3, 2)
            # y, z = x.T
            # print(y.shape)
            # print(z.shape)

            # => torch.Size([3])
            # => torch.Size([3])

            # 因此：
            # gi.shape = [1659]
            # gj.shape = [1659]
            gi, gj = gij.T  # grid indices

            # Append

            # gi.shape = [1659]
            # gj.shape = [1659]
            # gi = gi.clamp(0, shape[3] - 1)
            # gj = gj.clamp(0, shape[2] - 1)
            # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors 对应的所有anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
    
# 参考：https://zhuanlan.zhihu.com/p/591833099
