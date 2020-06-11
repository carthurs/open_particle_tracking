from __future__ import print_function
###################################################################################
####### ************************ GENERATEPATHLINES ************************ #######
###################################################################################
#                                                                                ##
# Particle pathline generation following particle advection.                     ##
# Requires HDF5, H5PY, NUMPY.                                                    ##
# Runs in serial.                                                                ##
#                                                                                ##  
# Command line arguments:                                                        ##
# 1. fname -- string containing the file name of the particle tracking HDF5 file ##
# 2. npathlines -- integer of desired number of equispaced pathlines             ##
#                  if npathlines==-1 : then all particles are considered to      ##
#                                      build the pathlines                       ##
###################################################################################
###################################################################################
###################################################################################

from builtins import range
import numpy as np
import sys
#sys.path.append('/opt/local/lib/python2.7/site-packages')

import h5py
import vtk
from vtk.util import numpy_support as ns

## open particle file 

fname  = sys.argv[1]
f = h5py.File(fname, 'r')

## load indices

indices = f["Index"]

## sort timesteps

timesteps = []

for key in list(indices.keys()):	
	if key != "Final":
		timesteps.append(key)
timesteps.sort(key=int)
number_of_timesteps = len(timesteps)

## determine maximum number of particles from first timestep

index = indices[timesteps[0]][:]
max_index = np.max(index)
number_of_particles = max_index + 1

#### ADDED BY PAOLO ##########
## Get representative 'seeds' from partitioning
npathlines = int(sys.argv[2])
if npathlines != -1:
    initial_pts_vtk = vtk.vtkPoints()
    initial_pts = f["Coordinates/"+timesteps[0]][:]
    initial_pts_vtk.SetData(ns.numpy_to_vtk(initial_pts))
    kdtree = vtk.vtkKdTree()
    kdtree.SetNumberOfRegionsOrMore(npathlines)
    kdtree.BuildLocatorFromPoints(initial_pts_vtk)
    decimated_index = np.zeros((kdtree.GetNumberOfRegions(),),dtype=np.int64)
    for i in range(kdtree.GetNumberOfRegions()):
        points_in_regions = ns.vtk_to_numpy(kdtree.GetPointsInRegion(i))
        decimated_index[i] = points_in_regions[0]
else:
        decimated_index = np.arange(0,number_of_particles,dtype=np.int64)
##### ENDED ADDED BY PAOLO ############


## generate empty (-1) array having each particle pathline in the rows
	
particle_lines_arr = np.zeros((number_of_particles,number_of_timesteps),dtype=np.int64)
particle_lines_arr[:] = -1
## vtk point array

particle_points = vtk.vtkPoints()
particle_points_arr = np.zeros((number_of_particles*number_of_timesteps,3))

## loop through timesteps and store point coordinates and array indices
field_array = np.zeros(number_of_particles*len(timesteps))
previous_nparticles = 0
for t in range(len(timesteps)):

    print("Loading timestep", timesteps[t])
    index = indices[timesteps[t]][:]
    coordinates = f["Coordinates/"+timesteps[t]][:]
    cur_field_array = f["PRT/"+timesteps[t]][:]

    is_dec_in_idx = np.in1d(index,decimated_index)
    index = index[is_dec_in_idx]
    coordinates = coordinates[is_dec_in_idx]
    cur_nparticles = index.shape[0]
    cur_nparticles += previous_nparticles
    field_array[previous_nparticles:cur_nparticles] = cur_field_array[is_dec_in_idx]


    particle_points_arr[previous_nparticles:cur_nparticles] = coordinates
    particle_lines_arr[index,t] = np.arange(previous_nparticles,cur_nparticles)

    previous_nparticles = cur_nparticles


particle_points_arr_final = np.copy(particle_points_arr[:cur_nparticles])
particle_points.SetData(ns.numpy_to_vtk(particle_points_arr_final))

## create vtk polydata 

polydata = vtk.vtkPolyData()

## add the points to the dataset

polydata.SetPoints(particle_points)

## loop over each particle's streamline and add it to the vtk cell 

particle_cell = vtk.vtkCellArray()

for i in range(number_of_particles):
        nlinepts = np.sum(particle_lines_arr[i]!=-1)
        if nlinepts > 0 :         
                line = vtk.vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(nlinepts)
                for j in range(0,nlinepts):
                        line.GetPointIds().SetId(j, particle_lines_arr[i][j])   # second point / global id
                particle_cell.InsertNextCell(line)

## add lines to polydata

polydata.SetLines(particle_cell)

## add field array
field_array_vtk = ns.numpy_to_vtk(field_array[:cur_nparticles])
field_array_vtk.SetName("PRT")
polydata.GetPointData().AddArray(field_array_vtk)

## write out to file

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("pathlines.vtp")
if vtk.VTK_MAJOR_VERSION == 6:
    writer.SetInputData(polydata)
else:
    writer.SetInput(polydata)
writer.Write()






