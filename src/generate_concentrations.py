from __future__ import print_function
########################################################################################################################
#   generate_concentrations
#
#   Arguments:
#       h5_filename: particle HDF5 file
#       mesh_filename: .vtu that represents the mesh
########################################################################################################################

from builtins import str
from builtins import range
from os.path import sep, split
from sys import argv
from mpi4py import MPI
import h5py
import numpy as np
import vtk
from vtk.util import numpy_support as ns
import helper_functions as function

# start MPI, get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# input file
h5_filename = argv[1]
input_folder = split(h5_filename)[0]

# mesh file
mesh_filename =  argv[2]

# read mesh and number of cells
mesh = function.read_vtu_file(mesh_filename)
number_of_cells = mesh.GetNumberOfCells()

# build locator
cell_locator = vtk.vtkCellLocator()
cell_locator.SetDataSet(mesh)
cell_locator.BuildLocator()

# open h5 file
h5_file = h5py.File(h5_filename, 'r')

# get time steps
time_steps = function.get_particle_time_steps(h5_file)

# time step limit, read from command line, or set to limits
try: 
    time_step_start = argv[3]
    time_step_end   = argv[4]
except:
    time_step_start = time_steps[0]
    time_step_end   = time_steps[-1]

# apply limits
time_steps = [item for item in time_steps if int(item) >= int(time_step_start)]
time_steps = [item for item in time_steps if int(item) <= int(time_step_start)]

# loop through steps
for step in time_steps:

    # output step number to the terminal
    print("Rank: %s, Step: %s" % (rank, step))

    # get coordinates of all the particles
    coordinates = h5_file["Coordinates/" + step][:]

    # get plap of all the particles
    plap = h5_file["PLAP/" + step][:]

    # get array indices and divide amongst processors
    array_indices = np.arange(0,len(coordinates))
    local_array_indices = function.divide_array_by_rank(array_indices, rank, size)

    # zero local concentration for this step
    local_concentration = np.zeros(number_of_cells, dtype=int)

    # zero local plap for this step, set data type to match plap array
    local_plap = np.zeros(number_of_cells, dtype=plap.dtype)

    # loop through local array indices
    for local_id in local_array_indices:
        cell_id = cell_locator.FindCell(coordinates[local_id,:])
        if cell_id != -1:
            local_concentration[cell_id] += 1
            local_plap[cell_id] += plap[local_id]

    # mpi all reduce to get the global concentration sum
    global_concentration = np.zeros(number_of_cells, dtype=int)
    comm.Allreduce([local_concentration, MPI.INT], [global_concentration, MPI.INT], op=MPI.SUM)

    # mpi all reduce to get the global plap sum
    # plap is currently stored as float32, which corresponds to MPI.FLOAT
    global_plap = np.zeros(number_of_cells, dtype=plap.dtype)
    comm.Allreduce([local_plap, MPI.FLOAT], [global_plap, MPI.FLOAT], op=MPI.SUM)


    # write out files on root
    if rank == 0:

        # delete existing arrays
        numberOfCellArrays = mesh.GetCellData().GetNumberOfArrays()
        for i in range(numberOfCellArrays):
            mesh.GetCellData().RemoveArray(i)

        # set vtk array for concentration
        vtk_concentration = ns.numpy_to_vtk(global_concentration)
        vtk_concentration.SetName("concentration")
        mesh.GetCellData().AddArray(vtk_concentration)

        # set vtk array for concentration
        vtk_plap = ns.numpy_to_vtk(global_plap)
        vtk_plap.SetName("plap sum")
        mesh.GetCellData().AddArray(vtk_plap)

        # calculate average plap using concentrations only in non-zero locations
        global_plap_avg = np.zeros(global_plap.shape, dtype=global_plap.dtype)
        non_zero_concentration = np.where(global_concentration > 0) 
        global_plap_avg[non_zero_concentration] = np.divide(global_plap[non_zero_concentration], global_concentration[non_zero_concentration])

        # set vtk array
        vtk_plap_avg = ns.numpy_to_vtk(global_plap_avg)
        vtk_plap_avg.SetName("plap average")
        mesh.GetCellData().AddArray(vtk_plap_avg)

        # set writer and write out
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(input_folder+'concentration_'+str(step)+'.vtu')
        if vtk.VTK_MAJOR_VERSION == 6:
            writer.SetInputData(mesh)
        else:
            writer.SetInput(mesh)
        writer.Write()

    # mpi barrier
    comm.Barrier()

