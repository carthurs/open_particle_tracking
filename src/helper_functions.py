from __future__ import division
from __future__ import print_function
from past.utils import old_div
import numpy as np
import vtk
import fnmatch
import os


def divide_array_by_rank(array, rank, size):
    """Divide list/1D NumPy array into pieces for parallel computation

    The array is split into the number of pieces defined by size.
    The size of each piece is determined by dividing the length
    of the array by the variable size. On the last processor the
    piece extends to the last entry.

        Args:
            array : NumPy array
            rank : MPI rank
            size : MPI size

        Returns:
            array[start:stop] : Array piece for this rank

        Raises :
            None
        """

    # calculate piece size
    piece_size = old_div(len(array),size)

    # determine array start and stop by rank
    start = rank * piece_size
    stop = (rank + 1) * piece_size

    # return piece
    if rank == size - 1:
        return array[start:]
    else:
        return array[start:stop]


def read_vtu_file(filename):
    """Read .vtu file and return unstructured grid object

        Args :
            filename : filename of .vtu file

        Returns :
            Array piece

        Raises :
            None
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def get_particle_time_steps(particle_h5_file):
    """Get time steps stored in the HDF5 particle file

        Args :
            particle_h5_file : HDF5 file generated from particle tracking

        Returns :
            time_steps : list containing step numbers

        Raises :
            None
    """

    # read step numbers
    indices = particle_h5_file['Index']

    # add all keys to list, ignoring the 'Final' key
    time_steps = []
    for key in list(indices.keys()):
        if key != 'Final':
            time_steps.append(key)

    return time_steps


def find_files_matching_pattern(pattern, directory = None):
    """Find files matching string pattern

    Find files matching string pattern in specified directory.
    Default search directory is the current directory

        Args :
            pattern :  pattern to match, accepts wildcards, i.e. *.txt
            directory : folder to search (optional)

        Returns :
            list_of_files : list of matching file names

        Raises :
            None
    """

    if directory is None:
        directory = '.'

    list_of_files = []

    for file_name in os.listdir(directory):
        if fnmatch.fnmatch(file_name, pattern):
            list_of_files.append(file_name)

    return list_of_files


def find_files_in_folder(file_name_pattern = None):
    """Find file in folder which match optional file name pattern

    Find files in folder which match an optional file name pattern.
    If no pattern string is provided, the current contents of the
    current folder is given

        Args :
            file_name_pattern : optional string with file name pattern

        Returns :
            list_of_files : list of files

        Raises :
            None
    """


    list_of_files = []

    if file_name_pattern is not None:
        for file in os.listdir(os.getcwd()):
            if file_name_pattern in file:
                list_of_files.append(file)
    else:
        list_of_files = os.listdir(os.getcwd())

    return list_of_files


if __name__ == '__main__':

    # array = np.arange(0,1)
    # print array
    #
    # print divide_array_by_rank(array, 0, 3)
    # print divide_array_by_rank(array, 1, 3)
    # print divide_array_by_rank(array, 2, 3)

    print(find_files_in_folder())

    print(find_files_in_folder('vtu'))