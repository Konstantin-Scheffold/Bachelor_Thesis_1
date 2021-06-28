import numpy as np
from Managing_data.Read_dicom import Data_PD
from Managing_data.Def_functions import make_a_mesh
from models import *
from datasets import *
from torch.utils.data import DataLoader

load_model = False

if load_model:
    Tensor = torch.cuda.FloatTensor

    val_dataloader = DataLoader(
        ImageDataset("../Data/Dicom_Data_edited/val", transforms_=[transforms.ToTensor()]),
        batch_size=8,
        shuffle=True,
    )
    generator = GeneratorUNet()
    generator = generator.cuda()
    generator.load_state_dict(torch.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\saved_models\facades\generator_3.pth'))
    imgs = next(iter(val_dataloader))
    real_CT = imgs["CT"].type(Tensor)
    real_PD = imgs["PD"].type(Tensor)
    fake_PD = generator(real_CT)[0]
    real_PD = real_PD[0]

else:
    a = np.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\images\facades\1.npy', allow_pickle=True)
    real_PD = a[1][0]
    fake_PD = a[2][0]

Data_PD.data = np.array(list(real_PD.cpu().squeeze().detach().numpy()), dtype=float)
make_a_mesh(Data_PD, 'newsubsamples_bigdisc_MSE_biggen_real.ply', np.mean(Data_PD.data))

Data_PD.data = np.array(list(fake_PD.cpu().squeeze().detach().numpy()), dtype=float)
make_a_mesh(Data_PD, 'newsubsamples_bigdisc_MSE_biggen_fake.ply', np.mean(Data_PD.data))
