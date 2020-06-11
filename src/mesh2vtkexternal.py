#####################################
####### MESH2VTKEXTERNAL ############
#####################################
#
# External surface from CRIMSON mesh when provided with opportune .ebc file. 
# Outputs in VTK polydata format. Requires VTK
#
# Command line arguments:
# 1. name -- string indicating the representative mesh name
# 2. allwelements.ebc -- filename of .ebc file containing external faces of interest
#############################################################

import vtk
import sys
import particles

custom_config_file_name = 'particle_config.json'
config_manager = particles.Configuration(custom_config_file_name)

# Open the file with point coordinates and elements
fnodes = open("%s.coordinates" % sys.argv[1], 'r')
felems = open("%s.connectivity" % sys.argv[1], 'r')
fallw  = open(sys.argv[2], 'r')

MAX_N = 20000000 # 20e6, large number useful for initialization

points = vtk.vtkPoints()
points.SetNumberOfPoints(MAX_N)
for i,line in enumerate(fnodes):
    if config_manager.meshManuallyExtractedFromGeombc():
        x,y,z = line.split()
    else:
        x,y,z = line.split()[1:]
    x = float(x)
    y = float(y)
    z = float(z)
    points.SetPoint(i,x,y,z)
points.SetNumberOfPoints(i+1)

polys = vtk.vtkCellArray()
polys.EstimateSize(MAX_N,4)
for i,line in enumerate(fallw):    
    a,b,c = line.split()[2:]
    a = int(a)-1
    b = int(b)-1
    c = int(c)-1
    polys.InsertNextCell(3)
    polys.InsertCellPoint(a)
    polys.InsertCellPoint(b)
    polys.InsertCellPoint(c)

grid = vtk.vtkPolyData()
grid.SetPoints(points)
grid.SetPolys(polys)
if vtk.VTK_MAJOR_VERSION != 6:
    grid.Update()

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("%s.vtp" % sys.argv[2])
if vtk.VTK_MAJOR_VERSION == 6:
    writer.SetInputData(grid)
else:
    writer.SetInput(grid)
writer.Update()

cleaner = vtk.vtkCleanPolyData()
if vtk.VTK_MAJOR_VERSION == 6:
    cleaner.SetInputData(grid)
else:
    cleaner.SetInput(grid)
cleaner.Update()

writer.SetFileName("%s-clean.vtp" % sys.argv[2])
if vtk.VTK_MAJOR_VERSION == 6:
    writer.SetInputData(cleaner.GetOutput())
else:
    writer.SetInput(cleaner.GetOutput())
writer.Update()

###
# Write the data in .vtu format, too
###

# there's no filter to convert PolyData to Unstructured (vtu), but the append filter does this naturally:
appendFilter = vtk.vtkAppendFilter()
if vtk.VTK_MAJOR_VERSION == 6:
    appendFilter.AddInputData(cleaner.GetOutput())
else:
    appendFilter.AddInput(cleaner.GetOutput())
appendFilter.Update()

unstructuredGrid = vtk.vtkUnstructuredGrid()
unstructuredGrid.ShallowCopy(appendFilter.GetOutput())

unstructuredGridWriter = vtk.vtkXMLUnstructuredGridWriter()
unstructuredGridWriter.SetFileName("%s-clean.vtu" % sys.argv[2])
if vtk.VTK_MAJOR_VERSION == 6:
    unstructuredGridWriter.SetInputData(unstructuredGrid)
else:
    unstructuredGridWriter.SetInput(unstructuredGrid)
unstructuredGridWriter.Update()
    


