#############################
####### MESH2VTK ############
#############################
#
# Convert CRIMSON mesh (coordinates and connectivity) to VTK Unstructured GRID. 
# Requires VTK
#
# Command line arguments:
# 1. name -- string indicating the representative mesh name
#############################################################

import vtk
import sys
import particles

def mesh2vtk(input_filename, output_filename, config_manager, mesh_was_manually_extracted_from_geombc):
    # Open the file with point coordinates and elements
    fnodes = open("%s.coordinates" % input_filename)
    felems = open("%s.connectivity" % input_filename)

    MAX_N = 20000000 # 20e6, large number useful for initialization
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(MAX_N)
    for i,line in enumerate(fnodes):
        if mesh_was_manually_extracted_from_geombc:
            x,y,z = line.split()
        else:
            x,y,z = line.split()[1:]
        x = float(x)
        y = float(y)
        z = float(z)
        points.SetPoint(i,x,y,z)
    points.SetNumberOfPoints(i+1)

    polys = vtk.vtkCellArray()
    polys.EstimateSize(MAX_N, 4)
    for i,line in enumerate(felems):
        if mesh_was_manually_extracted_from_geombc:
            a,b,c,d = line.split()
        else:
            a,b,c,d = line.split()[1:]
        a = int(a)-1
        b = int(b)-1
        c = int(c)-1
        d = int(d)-1
        polys.InsertNextCell(4)
        polys.InsertCellPoint(a)
        polys.InsertCellPoint(b)
        polys.InsertCellPoint(c)
        polys.InsertCellPoint(d)

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(10, polys) # 10 is tetrahedra
    if vtk.VTK_MAJOR_VERSION != 6:
        grid.Update()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName("%s.vtu" % output_filename)
    if vtk.VTK_MAJOR_VERSION == 6:
        writer.SetInputData(grid)
    else:
        writer.SetInput(grid)
    writer.Update()

if __name__ == "__main__":
    custom_config_file_name = 'particle_config.json'
    config_manager = particles.Configuration(custom_config_file_name)
    mesh2vtk(sys.argv[1], sys.argv[1], config_manager, config_manager.meshManuallyExtractedFromGeombc())  # maintaining old functionality. may want to make the output filename different from the command line at some point

    


