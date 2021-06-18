import numpy as np
from Managing_data.Read_dicom import Data_PD
from Managing_data.Def_functions import make_a_mesh
a = np.load(r'C:\Users\Konra\PycharmProjects\Bachelor_Thesis\Pix2Pix\images\facades\177000.npy', allow_pickle=True)
Data_PD.data = np.array(list(a[0].cpu().squeeze().detach().numpy()), dtype= float)
make_a_mesh(Data_PD, 'generated_PD.ply', 8000)