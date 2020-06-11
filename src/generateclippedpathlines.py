from __future__ import print_function
#############################################
####### GENERATECLIPPEDPATHLINES ############
#############################################
#
# Generate clipped pathline from particle data
#
# Command line arguments:
# 1 - .h5 file containing the particle data
# 2 - starting step
# 3 - stop step
# 4 - interval
# 5 - .vtp containing the clipping surface, must be closed
#############################################################

from builtins import str
from builtins import range
import sys
import h5py
import vtk
from vtk.util import numpy_support as ns
import numpy as np
import numpy.ma as ma

h5filename  = sys.argv[1]
start = sys.argv[2]
stop = sys.argv[3]
interval = sys.argv[4]
clip_filename = sys.argv[5]

# TODO: remove these lines, only for testing
# h5filename = r"C:\Users\kl12\Documents\CRIMSON\Fontan-II\Model-3521905-16\1-Pre-Op\Mesh-2\Pre-Op-Particles\Bolus-I\pre-op-particles-2500.h5"
# start = str(2500)
# stop = str(2502)
# interval = str(2)
# clip_filename = r"C:\Users\kl12\Documents\CRIMSON\Fontan-II\Model-3521905-16\1-Pre-Op\Mesh-2\Pre-Op-Particles\Bolus-I\clip-test.vtp"

# open h5 file
h5file = h5py.File(h5filename, 'r')

# read step numbers
indices = h5file["Index"]

# load clip surface, must be closed
#reader = vtk.vtkXMLPolyDataReader()
reader = vtk.vtkPolyDataReader()
reader.SetFileName(clip_filename)
reader.Update()
surface_polydata = reader.GetOutput()

# determine maximum number of particles from first timestep
index = indices[start][:]
max_index = np.max(index)
number_of_particles = max_index + 1

# generate iterator and count steps
step_range = range(int(start), int(stop) + int(interval), int(interval))
number_of_timesteps = len(step_range)

# generate oversized 2d array for each particle and the total number of time steps
# array initialised with -1
particle_lines = np.zeros((number_of_particles, number_of_timesteps), dtype=int)
particle_lines[:] = -1

# numpy array for particle points
particle_points = np.zeros((number_of_particles*number_of_timesteps,3))

# numpy array for plap
plap_array = np.zeros(number_of_particles*len(step_range))

# numpy array for prt
prt_array = np.zeros(number_of_particles*len(step_range))

# counter
previous_number_of_particles = 0

# initialise filter with surface data external to iterator
# thus only the particle points are updated within the loop
select = vtk.vtkSelectEnclosedPoints()
if vtk.VTK_MAJOR_VERSION == 6:
	select.SetSurfaceData(surface_polydata)
else:
	select.SetSurface(surface_polydata)

# uncomment and modify to set lower (more accurate) tolerance for the bounding box
#select.SetTolerance(0.0002)

for i, step in enumerate(step_range):

    print("step number", step)

    # get global indices in this time step
    index = indices[str(step)][:]

    # read coordinates, plap and prt
    coordinates = h5file["Coordinates/" + str(step)][:]
    current_plap = h5file["PLAP/" + str(step)][:]
    current_prt = h5file["PRT/" + str(step)][:]

    # create vtkpoints and polydata
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(ns.numpy_to_vtk(coordinates))
    particle_polydata = vtk.vtkPolyData()
    particle_polydata.SetPoints(vtkpoints)

    # apply to filter and update
    if vtk.VTK_MAJOR_VERSION == 6:
    	select.SetInputData(particle_polydata)
    else: 
    	select.SetInput(particle_polydata)

    select.Update()

    # get mask, 0 outside, 1 inside
    vtk_mask = select.GetOutput().GetPointData().GetArray("SelectedPoints")
    mask = ns.vtk_to_numpy(vtk_mask)

    # get indices inside
    inside = np.where(mask == 1)[0]

    # count
    current_number_of_particles = len(inside)
    current_number_of_particles += previous_number_of_particles

    # store values of plap and prt
    plap_array[previous_number_of_particles:current_number_of_particles] = current_plap[inside]
    prt_array[previous_number_of_particles:current_number_of_particles] = current_prt[inside]

    # store coordinates and indices for the lines
    particle_points[previous_number_of_particles:current_number_of_particles] = coordinates[inside]
    particle_lines[index[inside], i] = np.arange(previous_number_of_particles, current_number_of_particles)

    # update count
    previous_number_of_particles = current_number_of_particles

# create array with final number of points
particle_points_final = np.copy(particle_points[:current_number_of_particles])
particle_points = vtk.vtkPoints()
particle_points.SetData(ns.numpy_to_vtk(particle_points_final))

# create vtk polydata
polydata = vtk.vtkPolyData()
polydata.SetPoints(particle_points)

# loop over each particle pathline and add to vtk cell
particle_cell = vtk.vtkCellArray()

for i in range(number_of_particles):

         # get the local indices of this line and remove any -1 values
         line_index = particle_lines[i]
         line_index = line_index[line_index != -1]
         number_of_line_points = len(line_index)        

         # if the number of points are > 1
         if number_of_line_points > 1:

                 # find the minimum prt value and subtract from line
                 minimum_prt = np.min(prt_array[line_index])
                 prt_array[line_index] -= minimum_prt

                 # generate line
                 line = vtk.vtkPolyLine()
                 line.GetPointIds().SetNumberOfIds(number_of_line_points)

                 for j in range(0,number_of_line_points):
                         line.GetPointIds().SetId(j, line_index[j]) # 2nd argument is the id of the vtkpoints data
		 
                 particle_cell.InsertNextCell(line)

# add lines to polydata
polydata.SetLines(particle_cell)

# add arrays
plap_array_vtk = ns.numpy_to_vtk(plap_array[:current_number_of_particles])
plap_array_vtk.SetName("PLAP")
polydata.GetPointData().AddArray(plap_array_vtk)

prt_array_vtk = ns.numpy_to_vtk(prt_array[:current_number_of_particles])
prt_array_vtk.SetName("PRT")
polydata.GetPointData().AddArray(prt_array_vtk)

# write out to file
writer = vtk.vtkXMLPolyDataWriter()

# TODO: remove line, only for testing
# writer.SetFileName(r"C:\Users\kl12\Documents\CRIMSON\Fontan-II\Model-3521905-16\1-Pre-Op\Mesh-2\Pre-Op-Particles\Bolus-I\clipped-pathlines.vtp")

#writer.SetFileName(r"clipped-pathlines.vtp")
#if vtk.VTK_MAJOR_VERSION == 6:
#    writer.SetInputData(polydata)
#else:
#    writer.SetInput(polydata)
#writer.Write()

# write out cleaned data
cleaner = vtk.vtkCleanPolyData()
if vtk.VTK_MAJOR_VERSION == 6:
    cleaner.SetInputData(polydata)
else:
    cleaner.SetInput(polydata)
cleaner.Update()

writer.SetFileName("clipped-pathlines-clean.vtp")
writer.SetInput(cleaner.GetOutput())
writer.Write()
