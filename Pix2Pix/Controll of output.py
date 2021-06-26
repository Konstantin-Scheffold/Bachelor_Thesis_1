import numpy as np
from Managing_data.Read_dicom import Data_PD
from Managing_data.Def_functions import make_a_mesh
a = np.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\images\facades\4160.npy', allow_pickle=True)
Data_PD.data = np.array(list(a[1][0].cpu().squeeze().detach().numpy()), dtype= float)
make_a_mesh(Data_PD, 'first_try_scheduler_real_2.ply', np.mean(Data_PD.data))

Data_PD.data = np.array(list(a[2][0].cpu().squeeze().detach().numpy()), dtype= float)
make_a_mesh(Data_PD, '~/Pictures/first_try_scheduler_fake_2.ply', np.mean(Data_PD.data))