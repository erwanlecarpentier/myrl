from torch import Tensor
from torch.nn.modules.loss import _Loss
# from torch.nn import functional as F


class OptimisticLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, alpha, y_max, optimistic_factor=1.0, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(OptimisticLoss, self).__init__(size_average, reduce, reduction)

        self.alpha = alpha
        self.one_m_alpha = 1.0 - self.alpha
        self.y_max = y_max
        self.optimistic_factor = optimistic_factor

    def forward(self, y_prediction: Tensor, y_target: Tensor, y_augmented: Tensor) -> Tensor:
        """
        Forward optimistic loss function
        :param y_prediction: prediction on data set
        :param y_target: target on data set
        :param y_augmented: prediction on additional data for optimism
        :return:
        """
        q_norm = 0.5 / float(y_prediction.size()[0])

        if self.alpha < 1.0:
            u_norm = 1.0 * self.optimistic_factor * self.one_m_alpha / float(y_augmented.size()[0])
            loss = q_norm * (self.alpha * (y_prediction - y_target).pow(2) -
                             self.one_m_alpha * (y_prediction - self.y_max).pow(2)).sum()\
                   + u_norm * (y_augmented - self.y_max).pow(2).sum()
        else:
            return q_norm * (y_prediction - y_target).pow(2).sum()

        return loss


class DualOptimisticLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, alpha, y_max, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(DualOptimisticLoss, self).__init__(size_average, reduce, reduction)

        self.alpha = alpha
        self.one_m_alpha = 1.0 - self.alpha
        self.y_max = y_max

    def forward(self, y_prediction: Tensor, y_target: Tensor, y_augmented: Tensor) -> Tensor:
        """
        Forward contour-optimistic loss function
        :param y_prediction: prediction on data set
        :param y_target: target on data set
        :param y_augmented: prediction on additional data for optimism
        :return:
        """
        q_norm = self.alpha * 0.5 / float(y_prediction.size()[0])
        # y_target = 0.5 * (y_target + self.y_max)  # Efficient include of symmetric

        if self.alpha < 1.0:
            c_norm = self.one_m_alpha * 0.5 / float(y_augmented.size()[0])
            return q_norm * (y_prediction - y_target).pow(2).sum() + c_norm * (y_augmented - self.y_max).pow(2).sum()
        else:
            return q_norm * (y_prediction - y_target).pow(2).sum()
