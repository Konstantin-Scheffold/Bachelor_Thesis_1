import numpy as np
from Managing_data.Read_dicom import Data_PD
from Managing_data.Def_functions import make_a_mesh
a = np.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\archiv_images\images_run7_disc-2layer\facades\4168.npy', allow_pickle=True)
Data_PD.data = np.array(list(a[1][0].cpu().squeeze().detach().numpy()), dtype= float)
make_a_mesh(Data_PD, 'disc-2layers_real.ply', np.mean(Data_PD.data))

Data_PD.data = np.array(list(a[2][0].cpu().squeeze().detach().numpy()), dtype= float)
make_a_mesh(Data_PD, 'disc-2layers_fake.ply', np.mean(Data_PD.data))