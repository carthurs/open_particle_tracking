from __future__ import print_function
########################################################################################################################
#   generate_concentrations
#
#   Arguments:
#       h5_filename: particle HDF5 file
#       mesh_filename: .vtu that represents the mesh
########################################################################################################################

from builtins import range
from os.path import sep
from sys import argv
from mpi4py import MPI
import numpy as np
import vtk
from vtk.util import numpy_support as ns
import helper_functions as function


def generate_angio(file_name_pattern):
    """Generate virtual angiogram from particle concentrations

    Generate virtual angiogram from particle concentrations as calculated from generate_concentrations.py.


        Args:
            file_pattern : file name pattern of the .vtu concentration file, e.g. concentration_*
            directory : directory name, optional

        Returns:
            None

        Raises:
            Exception : if file_pattern cannot be found
    """

    # start MPI, get rank and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # get global file list
    file_list = function.find_files_in_folder(file_name_pattern)

    # check if list is empty, raise exception if pattern list is empty
    if not file_list:
        raise Exception('Cannot find specified file pattern!')

    # get list list for this rank
    file_list = function.divide_array_by_rank(file_list, rank, size)

    # read mesh from first file and get number of cells
    mesh_file_name = file_list[0]
    mesh = function.read_vtu_file(mesh_file_name)
    number_of_cells = mesh.GetNumberOfCells()

    # generate array for concentration sum
    concentration_sum = np.zeros(number_of_cells, dtype=np.int)

    # get current concentration and add it the to sum
    concentration = ns.vtk_to_numpy(mesh.GetCellData().GetArray("concentration"))
    concentration_sum += concentration

    print("Rank %d Processed %s" % (rank, mesh_file_name))

    # remove first entry
    del file_list[0]

    # loop through remaining entries in the list
    for file_name in file_list:

        # read new mesh
        mesh_file_name = file_name
        mesh = function.read_vtu_file(mesh_file_name )

        # get new concentration and add it the to sum
        concentration = ns.vtk_to_numpy(mesh.GetCellData().GetArray("concentration"))
        concentration_sum += concentration

        print("Rank %d Processed %s" % (rank, mesh_file_name))

    # mpi all reduce to get the concentration sum
    global_concentration_sum = np.zeros(number_of_cells, dtype=int)
    comm.Allreduce([concentration_sum , MPI.INT], [global_concentration_sum, MPI.INT], op=MPI.SUM)

    # write out files on root
    if rank == 0:

        # delete existing arrays
        numberOfCellArrays = mesh.GetCellData().GetNumberOfArrays()
        for i in range(numberOfCellArrays):
            mesh.GetCellData().RemoveArray(i)

        # set vtk array for concentration
        vtk_concentration = ns.numpy_to_vtk(global_concentration_sum)
        vtk_concentration.SetName("concentration sum")
        mesh.GetCellData().AddArray(vtk_concentration)

        # set writer and write out
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName('concentration_sum.vtu')
        if vtk.VTK_MAJOR_VERSION == 6:
            writer.SetInputData(mesh)
        else:
            writer.SetInput(mesh)
        writer.Write()

if __name__ == '__main__':

    # TODO remove test lines
    # generate_angio('concentration_*', r'C:\Users\kl12\Documents\CRIMSON\Fontan-II\Model-3521905-16\2-Post-Op-Fontan-To-Azygos\9-Adapted-Files\Concentrations')

    # if not argv[2]:
    #     generate_angio(argv[1])
    # else:

    generate_angio(argv[1])