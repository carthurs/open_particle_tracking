from __future__ import division
from __future__ import print_function
#####################################
####### TFP #########################
#####################################
#
# Extracts wall indices from particle tracking results. Requires HDF5, VTK, NUMPY
#
# Command line arguments:
# 1. fname -- string indicating the representative name of the simulation
# 2. percent -- integer indicating how much the domain close to the wall will be probed to determine "near wall" quantities (in percents of local radius)
#############################################################

from builtins import range
from past.utils import old_div
import vtk
import sys
import numpy as np
from vtk.util import numpy_support as ns
import h5py
import glob

# Command line arguments
fname      = sys.argv[1]
percent    = float(sys.argv[2]) # e.g. 5
percent    = old_div(percent,100.0)      # = 0.05

# Read accessory files
surf = vtk.vtkXMLPolyDataReader()
surf.SetFileName("%s-wss.vtp" % fname)
surf.Update()
surf_locator = vtk.vtkPointLocator()
surf_locator.SetDataSet(surf.GetOutput())
surf_locator.BuildLocator()
mesh = vtk.vtkXMLUnstructuredGridReader()
mesh.SetFileName("mesh.vtu")
mesh.Update()
mesh_locator = vtk.vtkPointLocator()
mesh_locator.SetDataSet(mesh.GetOutput())
mesh_locator.BuildLocator()

surf_npts = surf.GetOutput().GetNumberOfPoints()
mesh_npts = mesh.GetOutput().GetNumberOfPoints()

map_surf_to_mesh = np.zeros((surf_npts,),dtype=np.int64)
for i in range(surf_npts):
    tmp_pt = surf.GetOutput().GetPoint(i)
    map_surf_to_mesh[i] = mesh_locator.FindClosestPoint(tmp_pt)
map_surf_to_mesh_bool = np.zeros((mesh_npts,1),dtype=np.bool)
map_surf_to_mesh_bool[map_surf_to_mesh] = 1

normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(surf.GetOutputPort())
normals.Update()
normals_locator = vtk.vtkPointLocator()
normals_locator.SetDataSet(surf.GetOutput())
normals_locator.BuildLocator()
map_surf_to_normals = np.zeros((surf_npts,),dtype=np.int64)
for i in range(surf_npts):
    tmp_pt = surf.GetOutput().GetPoint(i)
    map_surf_to_normals[i] = normals_locator.FindClosestPoint(tmp_pt)

normals_arr = ns.vtk_to_numpy(normals.GetOutput().GetPointData().GetArray("Normals"))[map_surf_to_normals]
radius_arr  = np.reshape(ns.vtk_to_numpy(surf.GetOutput().GetPointData().GetArray("Radius")), (surf_npts,1))
surf_pts = np.copy(ns.vtk_to_numpy(surf.GetOutput().GetPoints().GetData()))


# Average results of particle tracking going through VTU files
plap_list = glob.glob("%s-particles-*.vtu" % fname)
plap_reader = vtk.vtkXMLUnstructuredGridReader()
plap_reader.SetFileName(plap_list[0])
plap_reader.Update()
plap_npts = plap_reader.GetOutput().GetNumberOfPoints()
plap_ncells = plap_reader.GetOutput().GetNumberOfCells()
plap_avg  = np.zeros((plap_npts,))
prt_avg   = np.zeros((plap_npts,))
ftle_avg  = np.zeros((plap_npts,))
for plap_file in plap_list:
    print(plap_file)
    plap_reader = vtk.vtkXMLUnstructuredGridReader()
    plap_reader.SetFileName(plap_file)
    plap_reader.Update()
    if plap_reader.GetOutput().GetPointData().GetArray("PLAP"):
        plap_avg += ns.vtk_to_numpy(plap_reader.GetOutput().GetPointData().GetArray("PLAP"))
    else:
        plap_avg += ns.vtk_to_numpy(plap_reader.GetOutput().GetPointData().GetArray("AP"))    
    prt_avg += ns.vtk_to_numpy(plap_reader.GetOutput().GetPointData().GetArray("PRT"))
    ftle_avg += ns.vtk_to_numpy(plap_reader.GetOutput().GetPointData().GetArray("FTLE"))

plap_avg /= len(plap_list)
plap_avg_vtk = ns.numpy_to_vtk(plap_avg)
plap_avg_vtk.SetName("PLAP avg")
plap_reader.GetOutput().GetPointData().AddArray(plap_avg_vtk)
prt_avg /= len(plap_list)
prt_avg_vtk = ns.numpy_to_vtk(prt_avg)
prt_avg_vtk.SetName("PRT avg")
plap_reader.GetOutput().GetPointData().AddArray(prt_avg_vtk)
ftle_avg /= len(plap_list)
ftle_avg_vtk = ns.numpy_to_vtk(ftle_avg)
ftle_avg_vtk.SetName("FTLE avg")
plap_reader.GetOutput().GetPointData().AddArray(ftle_avg_vtk)

plap_writer = vtk.vtkXMLUnstructuredGridWriter()
plap_writer.SetInputConnection(plap_reader.GetOutputPort())
plap_writer.SetFileName("%s_plap_avg.vtu" % fname)
plap_writer.Update()

### PROBE AT 20 LOCATIONS CLOSE TO THE WALL
probe_polydata = vtk.vtkPolyData()
probe_vtk_pts = vtk.vtkPoints()
probe_vtk_pts.SetData(ns.numpy_to_vtk(surf_pts))
probe_polydata.SetPoints(probe_vtk_pts)
probe_topology = np.zeros((surf_npts,2),dtype=np.int64)
probe_topology[:,0] = 1
probe_topology[:,1] = np.arange(surf_npts)
probe_topology = np.reshape(probe_topology,(surf_npts*2,))
probe_verts = vtk.vtkCellArray()
probe_verts.SetCells(surf_npts,ns.numpy_to_vtkIdTypeArray(probe_topology))
probe_polydata.SetVerts(probe_verts)

plap_wall = np.zeros((surf_npts,))
prt_wall = np.zeros((surf_npts,))
ftle_wall = np.zeros((surf_npts,))
for i in range(1,21,1):
    surf_pts -= (1.0/20.0*percent*radius_arr)*normals_arr
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInput(probe_polydata)
    probe_filter.SetSourceConnection(plap_reader.GetOutputPort())
    probe_filter.Update()
    probe_writer = vtk.vtkXMLPolyDataWriter()
    probe_writer.SetInputConnection(probe_filter.GetOutputPort())
    probe_writer.SetFileName("%s-probe-%d.vtp" % (fname, i))
    probe_writer.Update()
    plap_wall += ns.vtk_to_numpy(probe_filter.GetOutput().GetPointData().GetArray("PLAP avg"))
    prt_wall += ns.vtk_to_numpy(probe_filter.GetOutput().GetPointData().GetArray("PRT avg"))
    ftle_wall += ns.vtk_to_numpy(probe_filter.GetOutput().GetPointData().GetArray("FTLE avg"))

plap_wall /= 20.0
plap_wall_vtk = ns.numpy_to_vtk(plap_wall)
plap_wall_vtk.SetName("PLAP")
surf.GetOutput().GetPointData().AddArray(plap_wall_vtk)
prt_wall /= 20.0
prt_wall_vtk = ns.numpy_to_vtk(prt_wall)
prt_wall_vtk.SetName("PRT")
surf.GetOutput().GetPointData().AddArray(prt_wall_vtk)
ftle_wall /= 20.0
ftle_wall_vtk = ns.numpy_to_vtk(ftle_wall)
ftle_wall_vtk.SetName("FTLE")
surf.GetOutput().GetPointData().AddArray(ftle_wall_vtk)

######  WSS MEASURES and TFP
tawss_arr = ns.vtk_to_numpy(surf.GetOutput().GetPointData().GetArray("TAWSS"))
tawss_arr[tawss_arr<=1e-5] = 1e-5 # Prevents from dividing by zero
osi_arr = ns.vtk_to_numpy(surf.GetOutput().GetPointData().GetArray("OSI"))
ecap_arr = old_div(osi_arr,tawss_arr)
tfp_arr = osi_arr/tawss_arr*plap_wall

ecap_vtk = ns.numpy_to_vtk(ecap_arr)
ecap_vtk.SetName("ECAP")
surf.GetOutput().GetPointData().AddArray(ecap_vtk)

tfp_vtk = ns.numpy_to_vtk(tfp_arr)
tfp_vtk.SetName("TFP")
surf.GetOutput().GetPointData().AddArray(tfp_vtk)

# Output file
writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputConnection(surf.GetOutputPort())
writer.SetFileName("%s-tfp.vtp" % fname)
writer.Update()
   
