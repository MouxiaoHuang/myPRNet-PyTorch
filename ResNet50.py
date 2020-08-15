import torchvision.models as models
import torch

ResNet50 = models.resnet50(pretrained=False)
ResNet50.fc = torch.nn.Linear(2048, 235, bias=True) # [shape_para, exp_para, pose_para] = [199, 29, 7] --> 235

device = torch.device("cuda")
ResNet50.to(device)

x = torch.randn(1, 3, 100, 100)
print(x.shape)
out = ResNet50(x.cuda())
print(out.shape)
print(ResNet50)