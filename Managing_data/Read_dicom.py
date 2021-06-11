import numpy as np
import pydicom
import os


# Store volume data together with grid size and position in space


class VolumeData:
    """
    Storage class for a Volume inside a coordinate system. The data
    itself is stored as a np.ndarray. Additional to this the spacing
    and the position is stored.
    """

    def __init__(self,
                 data,
                 spacing,
                 position=np.zeros(3)):
        """
        Initializes the VolumeData class
        Args:
            data: Volume data
            spacing: Voxel size
            position: Position in an arbitrary coordinate system
        """
        # Store volume data
        assert len(data.shape) == 3
        self.data = np.asarray(data, dtype=float)
        # Store size of the cells
        assert len(spacing) == 3
        self.spacing = np.asarray(spacing, dtype=float)
        # Store position in 3D space
        assert len(position) == 3
        self.position = np.asarray(position, dtype=int)


def read_dicom_stack(filepath):

    """
    Reads a stack of DICOM images into an AnnotatedImage
    To read in the stack all filenames with the .dcm ending are stored
    in an array then sorted and read in one after another.
    To create AnnotatedImage.data all slices are appended to one lager
    three dimensional ndarray.
    The position is calculated from the first and last slice. It is
    defined as the center of both slices.
    Spacing is calculated from the first two slices. x- and y-direction
    can be taken directly but z-direction needs to be calculated using
    the difference between the position of those two slices.
    The data is stored as [z, y, x]
    Args:
        filepath: Path to the image folder.
    Returns: AnnotatedImage with data, (voxel) size and position
    """

    #  from: https://pyscience.wordpress.com/2014/09/08/dicom-in-python-
    #  importing-medical-image-data-into-numpy-with-pydicom-and-vtk/
    file_list = []
    #  Read in filenames from directory and sort them
    for root, dirs, files in os.walk(filepath):
        for filename in files:
            if ".dcm" in filename.lower():
                file_list.append(os.path.join(root, filename))
    file_list.sort()
    #  Read in first reference file to define shape of data
    ref_data = pydicom.read_file(file_list[0])
    data_shape = (len(file_list), int(ref_data.Rows), int(ref_data.Columns))
    data = np.zeros(data_shape, int)
    #  Read in image stack and store in single ndarray
    for filename in file_list:
        data_slice = pydicom.dcmread(filename)
        data[file_list.index(filename), :, :] = data_slice.pixel_array
    # Read and calculate voxel size
    # The voxel size within one slice in given. Between the slices
    # (z-direction) has to be calculated as the difference in the
    # position of two slices.
    ref_data_second = pydicom.read_file(file_list[1])
    spacing3 = (ref_data_second.ImagePositionPatient[2]
                - ref_data.ImagePositionPatient[2])
    spacing = np.asarray([float(spacing3),
                          float(ref_data.PixelSpacing[1]),
                          float(ref_data.PixelSpacing[0])])
    # Calculate the volume position
    # The stored position in the dicom file points to top center of the
    # lower left last corner of the slice. Center in x,y direction and
    # on end of the z direction. (0.5,0.5,1) for a unit cell
    position = np.asarray([float(ref_data.ImagePositionPatient[2]),
                           float(ref_data.ImagePositionPatient[0]),
                           float(ref_data.ImagePositionPatient[1])])
    position = position - (np.asarray([1, 0.5, 0.5]) * spacing)
    # To get the real lover left last corner the stated position has to
    # be decreased by half the spacing in x and y direction and a full
    # spacing in z direction

    #  Delete ref_data*
    del ref_data
    del ref_data_second
    return VolumeData(data, spacing, position)

# Daten einlesen

Data_PD = read_dicom_stack('/Users/konstantinscheffold/PycharmProjects/Bachelor_Thesis/Data/Data_BA_Thesis/NewExportsNov2020/PD')
Data_CT = read_dicom_stack('/Users/konstantinscheffold/PycharmProjects/Bachelor_Thesis/Data/Data_BA_Thesis/NewExportsNov2020/CT')
