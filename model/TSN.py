import torch.nn
import torch.nn as nn
from utils import *
from torchsummary import summary
from model.temporal_model import TSN_temporal
from model.spatial_model import TSN_spatial


class TSN(nn.Module):
    def __init__(self, num_class, num_segments):
        super(TSN, self).__init__()
        self.spatial_model = TSN_spatial(num_class, num_segments, 'RGB', base_model='resnet18', new_length=1,
                                         crop_num=1, partial_bn=True)
        self.temporal_model = TSN_temporal(num_class, num_segments, 'flow', base_model='resnet18', new_length=5,
                                           crop_num=1, partial_bn=True)

    def consensus(self, output_rgb, output_flow):
        output = torch.concat((output_rgb, output_flow), dim=1)
        output = output.mean(dim=1)
        return output

    def forward(self, input_rgb, input_flow):
        output_rgb = self.spatial_model(input_rgb)
        output_flow = self.temporal_model(input_flow)

        output = self.consensus(output_rgb, output_flow)
        output = nn.Softmax(dim=1)(output)
        return output
    
    def get_transform_RGB(self):
        return self.spatial_model.get_augmentation()
    
    def get_transform_flow(self):
        return self.temporal_model.get_augmentation()
    


if __name__ == '__main__':
    model = TSN(num_class=15, num_segments=3).cuda()
    input_rgb = torch.randn(4, 9, 128, 128).cuda()
    input_flow = torch.randn(4, 45, 128, 128).cuda()
    print(summary(model, [(9, 128, 128,), (45, 128, 128,)]))
