from __future__ import print_function
#######################################
####### EXTRACTSURFACEDATA ############
#######################################
#
# Extract surface data from hdf5 and calculates the force on each element
#
# Command line arguments:
# 1 - .ebc file of the surface
# 2 - .h5 file with the simulation data
# 3 - step number of the hdf5 file to extract
#############################################################

from builtins import range
import numpy as np
import vtk
from vtk.util import numpy_support
import sys 
import h5py

# open .ebc which contains the surface in question
surface_ebc_file = open(sys.argv[1], 'r')
surface_ebc_data = []
surface_nbc_list = []
for i,line in enumerate(surface_ebc_file):
    element_id, mat_id, n1, n2, n3 = line.split()[:]        
    surface_ebc_data.append([element_id, n1, n2, n3])

# open .h5 simulation file
simulation_file = h5py.File(sys.argv[2], 'r')

# load coordinate
coordinates = simulation_file["Mesh/coordinates"][()][()]

# set vtk points and polys
points = vtk.vtkPoints()
polys = vtk.vtkCellArray()

for i in range(len(surface_ebc_data)):

	polys.InsertNextCell(3)

	for j in range(3):

		node_id = int(surface_ebc_data[i][1+j]) - 1	
		x = coordinates[node_id][0]
		y = coordinates[node_id][1]
		z = coordinates[node_id][2]

		id = points.InsertNextPoint(x,y,z)    
		polys.InsertCellPoint(id)

# set grid
surface_polydata = vtk.vtkPolyData()
surface_polydata.SetPoints(points)
surface_polydata.SetPolys(polys)
surface_polydata.Update()

# load pressure data
pressure_groups = simulation_file["pressure"]
pressure_data = pressure_groups[sys.argv[3]]

# get pressure data
pressure_array = vtk.vtkFloatArray()
pressure_array.SetNumberOfTuples(surface_polydata.GetNumberOfPoints())

for i in range(len(surface_ebc_data)):

	cell_point_id_list = vtk.vtkIdList() # get vtk indicies	
	surface_polydata.GetCellPoints(i,cell_point_id_list)	

	for j in range(3):

		node_id = int(surface_ebc_data[i][1+j]) - 1	
		pressure_value = pressure_data[node_id]

		vtk_id = cell_point_id_list.GetId(j)
		pressure_array.SetTuple1(vtk_id, pressure_value)

# add pressure point array
surface_polydata.GetPointData().AddArray(pressure_array)
pressure_array.SetName("pressure")
surface_polydata.Update()

# convert to cell data
surface_polydata_cell = vtk.vtkPointDataToCellData()
if vtk.VTK_MAJOR_VERSION == 6:
	surface_polydata_cell.SetInputData(surface_polydata)
else:
	surface_polydata_cell.SetInput(surface_polydata)
surface_polydata_cell.PassPointDataOn()
surface_polydata_cell.Update()

# create polydata normal object and set cell normals on
surface_polydata_cell_normals = vtk.vtkPolyDataNormals()
# surface_polydata_cell_normals.SetFlipNormals(1)
surface_polydata_cell_normals.ComputeCellNormalsOn() 
surface_polydata_cell_normals.SetInputConnection(surface_polydata_cell.GetOutputPort())
surface_polydata_cell_normals.Update()

# get cell normals and pressures 
cell_normals_array = surface_polydata_cell_normals.GetOutput().GetCellData().GetArray("Normals")
cell_pressure_array = surface_polydata_cell_normals.GetOutput().GetCellData().GetArray("pressure")

cell_normals = numpy_support.vtk_to_numpy(cell_normals_array)
cell_pressures = numpy_support.vtk_to_numpy(cell_pressure_array)

# numpy array force vector
force_vector_array = np.empty([surface_polydata_cell_normals.GetOutput().GetNumberOfCells(), 3])

# calculate force
for i in range(surface_polydata_cell_normals.GetOutput().GetNumberOfCells()):

	# calculate area and force 
	cell_area = surface_polydata_cell_normals.GetOutput().GetCell(i).ComputeArea()	
	force_vector_array[i][:] = cell_pressures[i]*cell_area*cell_normals[i][:]		
	
# write out force sum
force_sum = np.sum(force_vector_array, axis=0)
print(force_sum)
with open('force_resultant.dat','a') as f: 
	f.write("%s " % sys.argv[3])
	for i in force_sum:
		print(i)
  		f.write("%s " % i)
	f.write("\n")


# convert to vtk
force_vector_vtk = numpy_support.numpy_to_vtk(force_vector_array)

# add to polydata
surface_polydata_cell_normals.GetOutput().GetCellData().AddArray(force_vector_vtk)
force_vector_vtk.SetName("force")
surface_polydata_cell_normals.Update()

# write to file
writer = vtk.vtkXMLPolyDataWriter();
writer.SetFileName("force_surface_"+sys.argv[3]+".vtp");
if vtk.VTK_MAJOR_VERSION == 6:
	writer.SetInputData(surface_polydata_cell_normals)			
else:
	writer.SetInput(surface_polydata_cell_normals.GetOutput())	
writer.Write()

# clean poly data
cleaner = vtk.vtkCleanPolyData()
cleaner.SetInput(surface_polydata_cell_normals.GetOutput())
cleaner.Update()

writer.SetFileName("force_surface_clean_"+sys.argv[3]+".vtp")
writer.SetInput(cleaner.GetOutput())
writer.Update()
