from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
img=Image.open("hymenoptera_data/train/ants/0013035.jpg")
writer=SummaryWriter("logs")
# for i in range(100):
#     writer.add_scalar("y=x",i,i)
img=np.array(img)
writer.add_image("img",img,1,dataformats="HWC")
writer.close()