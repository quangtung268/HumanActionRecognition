import torch.nn as nn
from transform import *
from torchinfo import summary


class TSN_temporal(nn.Module):
    def __init__(self, num_class, num_segments, modality, base_model='resnet18',
                 new_length=None, dropout=0.7, crop_num=1, partial_bn=True):
        super(TSN_temporal, self).__init__()
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

        self._prepare_base_model(base_model)
        self._enable_pbn = partial_bn

        if partial_bn:
            self.partialBN(True)

        if self.modality == 'flow':
            self.base_model = self._construct_flow_model(self.base_model)

    def _prepare_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'

        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, self.num_class)
        nn.init.normal_(self.new_fc.weight, 0, 0.001)
        nn.init.constant_(self.new_fc.bias, 0)

    def _construct_flow_model(self, base_model):
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(3 * self.new_length, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                             conv_layer.padding, bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.data.bias = params[1].data  # add bias is necessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first conv layer
        setattr(container, layer_name, new_conv)
        return base_model


    def train(self, mode=True):
        super(TSN_temporal, self).train(mode)
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
    model = TSN_temporal(15, 3, 'flow').cuda()
    summary(model, (8, 45, 224, 224))
