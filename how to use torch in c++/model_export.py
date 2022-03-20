import torch
import torchvision.models as models
from PIL import Image
import numpy as np

image = Image.open("dog.png")
image = image.resize((224, 224), Image.ANTIALIAS)
image = np.asarray(image)
image = image / 255
image = torch.Tensor(image).unsqueeze_(dim=0)
image = image.permute((0, 3, 1, 2)).float()

model = models.resnet50(pretrained=True)
model = model.eval()
resnet = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
# output=resnet(torch.ones(1,3,224,224))
output = resnet(image)
max_index = torch.max(output, 1)[1].item()
print(max_index)  # ImageNet1000类的类别序
resnet.save('resnet.pt')
