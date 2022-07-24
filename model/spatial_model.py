import torch.nn
import torch.nn as nn
from transform import *
from torchinfo import summary


class TSN_spatial(nn.Module):
    def __init__(self, num_class, num_segments, modality, base_model='resnet18',
                 new_length=None, dropout=0.8, crop_num=1, partial_bn=True):
        super(TSN_spatial, self).__init__()
        self.num_class = num_class
        self.num_segments = num_segments
        self.modality = modality
        self.dropout = dropout
        self.crop_num = crop_num
        self.partial_bn = partial_bn
        self.reshape = True

        if new_length is None:
            self.new_length = 1 if modality == 'RGB' else 5
        else:
            self.new_length = new_length

        self._prepare_model(base_model)
        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'

        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, self.num_class)
        nn.init.normal_(self.new_fc.weight, 0, 0.001)
        nn.init.constant_(self.new_fc.bias, 0)

    def train(self, mode=True):
        super(TSN_spatial, self).train(mode)
        count = 0
        if self._enable_pbn:
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def forward(self, input):
        sample_len = 3 * self.new_length
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[2:]))
        base_out = self.new_fc(base_out)

        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = base_out

        return output.squeeze(-1)


if __name__ == '__main__':
    model = TSN_spatial(15, 3, 'RGB').cuda()
    summary(model, (3, 9, 224, 224))
