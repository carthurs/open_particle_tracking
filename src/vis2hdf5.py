from __future__ import division
#####################################
####### VIS2HDF5 ####################
#####################################
#
# Converts a series of CRIMSON .vis files into unique HDF5 file. 
# Performs accessory postprocessing operations such as shear rate computation 
# and lambdatwo criterion. Requires H5PY, VTK, MPI4PY, NUMPY. Runs in parallel
#
# Command line arguments:
# 1. name -- string indicating the representative of the simulation
# 2. start -- integer indicating initial timestep
# 3. stop -- integer indicating final timestep
# 4. step -- integer indicating interval between saved timesteps
# 5. mesh.vtu -- filename of mesh in VTK unstructured grid format
# 6. rp-sp.vtp -- filename of external surface in VTK polydata format
#############################################################

from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
from vtk.util import numpy_support
import vtk
import sys
from itertools import islice
import shutil
import os,errno
import subprocess
import mpi4py, mpi4py.MPI
import h5py
import pandas
import tqdm

import time

def copyFromVisToHdf5(fin_duplicate_for_pandas, npts, current_fin_line_index, group_name, timestep_index, ncols, fout):
    fin_duplicate_for_pandas.seek(0)  # pandas works from beginning of the file
    tmp_pandas_dataframe = pandas.read_csv(fin_duplicate_for_pandas,
                                           delim_whitespace=True,
                                           nrows = npts,
                                           header=None,
                                           skiprows=current_fin_line_index)
    read_datafield = np.reshape(tmp_pandas_dataframe,(npts, ncols))
    fout["/%s/%d" % (group_name, timestep_index)][:] = read_datafield
    return read_datafield

def vis2hdf5(fname,
             step_start,
             step_stop,
             step_step,
             mesh_fname,
             surface_fname,
             extract_velocity_only,
             compute_lambdatwo):

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    def printRankZero(message):
        if rank == 0:
            tqdm.tqdm.write(message)

    nprocs = comm.Get_size()

    nsteps = old_div((step_stop-step_start+step_step),step_step)
    nstepsperproc = old_div(nsteps,nprocs)
    rest   = nsteps % nprocs
    if rest:
        nstepsperproc += 1
    curstep_start = step_start

    mesh_reader = vtk.vtkXMLUnstructuredGridReader()
    mesh_reader.SetFileName(mesh_fname)
    mesh_reader.Update()

    if not extract_velocity_only:
        surface_data = vtk.vtkXMLPolyDataReader()
        surface_data.SetFileName(surface_fname)
        surface_data.Update()
        surface_npts = surface_data.GetOutput().GetNumberOfPoints()
        
    npts   = mesh_reader.GetOutput().GetNumberOfPoints()
    ncells = mesh_reader.GetOutput().GetNumberOfCells()

    # Open results file and in dictionary declare computed fields (name:n_of_components)
    printRankZero("Preparing hdf5 file...")
    fout = h5py.File("%s.h5" % fname,'w',driver='mpio',comm=comm)
    groups = {"pressure":1,
              "velocity":3,
              "wall shear stress":3,
              "traction":3,
              "displacement":3,
              "displacement_ref":3,
              "frobenius":1,
              "lambda two":1,
              }    
    # Create groups and datasets
    # Creating group and datasets are collective operations, all processes need to do it
    for group in groups:
        fout.create_group(group)
        for i in range(step_start,step_stop+step_step,step_step):
            fout[group].create_dataset("%d" % i, (npts,groups[group]),dtype='float64')

    # Create Mesh topology and coordinates
    printRankZero("Adding mesh data to hdf5 file...")
    fout.create_group("Mesh")
    fout["Mesh"].create_dataset("topology", (ncells,4), dtype='int64')
    fout["Mesh"].create_dataset("coordinates", (npts,3),dtype='float64')
    
    curnsteps=nstepsperproc
    if rank >= rest and rest:
        curnsteps -= 1
        curstep_start += rest*(curnsteps+1)*step_step+(rank-rest)*curnsteps*step_step
    else:
        curstep_start += rank*curnsteps*step_step        
    
   
    tmp_tawss = np.zeros(npts)    
    tmp_wssint = np.zeros((npts,3))
    tmp_lambdatwo = np.zeros((npts,1))
    tmp_frobenius = np.zeros((npts,1))
    #########################################################
    ####### PARSE VIS FILES #################################
    #########################################################
    arrayAttached = False
    for timestep_index in tqdm.trange(curstep_start,curstep_start+curnsteps*step_step,step_step, desc="Rank {}".format(rank)):
        currentVisFileName = "%s-%d.vis" % (fname,timestep_index)
        tqdm.tqdm.write("Rank {} Reading file {}.".format(rank, currentVisFileName))
        startTime = time.time()
        fin = open(currentVisFileName,'r')
        current_fin_line_index = 0
        fin_duplicate_for_pandas = open(currentVisFileName,'r')
        while True:
            # Get Group Name
            for line in fin:
                current_fin_line_index += 1
                line_split = line.split('"')
                if ("    analysis results " in line_split) : break
            
            if line == '\n':
                break # Reached end of file"

            group_name = line_split[-2]
            
            if group_name not in fout:
                tqdm.tqdm.write("ERROR: GROUP '%s' IS NOT ON YOUR LIST" % group_name)

            # Determine size of vector and initialize
            line_split = (next(fin)).split()
            current_fin_line_index += 1
            npts = int(line_split[-1])
            for line in fin:
                current_fin_line_index += 1
                line_split = line.split()
                if "length" in line_split: break
            ncols = int(line_split[-1])

            # Get to the data
            for line in fin:
                current_fin_line_index += 1
                line_split = line.split()
                if "data" in line_split: break
            #for j in xrange(npts):
                #tmp_vec[j] = [float(x) for x in fin.next().split()]   
            tqdm.tqdm.write(group_name)

            if extract_velocity_only:
                if group_name == "velocity":
                    read_datafield = copyFromVisToHdf5(fin_duplicate_for_pandas, npts,
                                      current_fin_line_index,
                                      group_name, timestep_index, ncols, fout)
                    # break
            else:
                read_datafield = copyFromVisToHdf5(fin_duplicate_for_pandas, npts,
                                  current_fin_line_index,
                                  group_name, timestep_index, ncols, fout)
            # print "shape", np.shape(read_datafield)

            if group_name == "wall shear stress" and not extract_velocity_only:
                tmp_tawss  += np.linalg.norm(read_datafield, axis=1)
                tmp_wssint += read_datafield


            if group_name == "velocity":
                # Compute gradient of velocity
                tmp_vec_vtk = None
                ctemp = np.ascontiguousarray(read_datafield)
                tmp_vec_vtk = numpy_support.numpy_to_vtk(ctemp)
                tmp_vec_vtk.SetName("tmp_velocity")
                
                mesh_reader = vtk.vtkXMLUnstructuredGridReader()
                mesh_reader.SetFileName(mesh_fname)
                mesh_reader.Update()

                mesh_reader.GetOutput().GetPointData().AddArray(tmp_vec_vtk)
                gradient_filter = vtk.vtkGradientFilter()
                gradient_filter.SetInputConnection(mesh_reader.GetOutputPort())
                gradient_filter.SetInputScalars(0,"tmp_velocity")
                gradient_filter.SetResultArrayName("tmp_velocity_gradient")
                gradient_filter.Update()

                tmp_vel_gradient = numpy_support.vtk_to_numpy(gradient_filter.GetOutput().GetPointData().GetArray("tmp_velocity_gradient"))
                tmp_vel_gradient   = np.reshape(tmp_vel_gradient, (npts,3,3))
                tmp_vel_gradient_t = np.transpose(tmp_vel_gradient, (0,2,1))
                tmp_S   = 0.5*(tmp_vel_gradient+tmp_vel_gradient_t)   
                tmp_frobenius[:] = np.reshape(np.linalg.norm(tmp_S, axis=(1,2)),(npts,1))
                fout["/frobenius/%d"  % (timestep_index,)][:] = tmp_frobenius

                if compute_lambdatwo:
                    tmp_W   = 0.5*(tmp_vel_gradient-tmp_vel_gradient_t)
                    tmp_S2  = np.einsum('abj,ajc->abc',tmp_S,tmp_S)
                    tmp_W2  = np.einsum('abj,ajc->abc',tmp_W,tmp_W)
                    eigvals = np.linalg.eigvals(tmp_S2+tmp_W2)
                    eigvals = np.sort(eigvals,axis=1)
                    tmp_lambdatwo = eigvals[:,1]
                    fout["/lambda two/%d" % (timestep_index,)][:] = np.reshape(tmp_lambdatwo,(npts,1))

                if extract_velocity_only:
                    break  #?? not sure whether necessary
        fin.close()
        tqdm.tqdm.write("Elapsed time Rank {}: {}".format(rank, time.time() - startTime))
    ####### END OF PARSING #################################

    printRankZero("Finished reading vis files.")
    # Average indices TAWSS and OSI can be conveniently computed through reduction operations
    if rank == 0:
        total_tawss  = np.zeros(npts)
        total_wssint = np.zeros((npts,3))
    else:
        total_tawss  = None
        total_wssint = None  

    comm.Reduce([tmp_tawss, mpi4py.MPI.DOUBLE],
                [total_tawss, mpi4py.MPI.DOUBLE],
                op = mpi4py.MPI.SUM,
                root = 0)

    comm.Reduce([tmp_wssint, mpi4py.MPI.DOUBLE],
                [total_wssint, mpi4py.MPI.DOUBLE],
                op = mpi4py.MPI.SUM,
                root = 0)    

    if extract_velocity_only:
        printRankZero("NOT Adding TAWSS and OSI data to hdf5 file...")
    else:
        printRankZero("Adding TAWSS and OSI data to hdf5 file...")
        fout.create_group("TAWSS")
        fout["TAWSS"].create_dataset("0", (npts,),dtype='float64')
        fout.create_group("WSSINT")
        fout["WSSINT"].create_dataset("0",(npts,3),dtype='float64')
        fout.create_group("OSI")
        fout["OSI"].create_dataset("0",(npts,),dtype='float64')
    
    if rank == 0:
        if not extract_velocity_only:
            # Also add TAWSS and OSI to HDF5 file
            total_tawss  /= nsteps
            total_wssint /= nsteps
            idx_ok = np.where(total_tawss>1e-6)
            total_wssint_norm = np.linalg.norm(total_wssint,axis=1)
            total_osi = np.zeros(npts)
            total_osi[idx_ok] = 0.5*(1.0-old_div(total_wssint_norm[idx_ok],total_tawss[idx_ok]))
            
            fout["/TAWSS/0"][:] = total_tawss
            fout["/OSI/0"][:] = total_osi
            fout["/WSSINT/0"][:] = total_wssint

        pts = numpy_support.vtk_to_numpy(mesh_reader.GetOutput().GetPoints().GetData())
        cells = numpy_support.vtk_to_numpy(mesh_reader.GetOutput().GetCells().GetData())
        cells = np.reshape(cells, (ncells,5))
        cells = cells[:,1:]
        fout["/Mesh/coordinates"][:] = pts
        fout["/Mesh/topology"][:] = cells        
        
        if not extract_velocity_only:
            mesh_locator = vtk.vtkPointLocator()
            mesh_locator.SetDataSet(mesh_reader.GetOutput())
            mesh_locator.BuildLocator()    
            map_surface_to_mesh = np.zeros(surface_npts, dtype=np.int64)
            for i in range(surface_npts):
                pt = surface_data.GetOutput().GetPoint(i)
                closept = mesh_locator.FindClosestPoint(pt)
                map_surface_to_mesh[i] = closept

            tawss_surface  = total_tawss[map_surface_to_mesh]
            osi_surface    = total_osi[map_surface_to_mesh]
            wssint_surface = total_wssint[map_surface_to_mesh]
            tawss_surface_vtk = numpy_support.numpy_to_vtk(tawss_surface)
            tawss_surface_vtk.SetName("TAWSS")
            osi_surface_vtk = numpy_support.numpy_to_vtk(osi_surface)
            osi_surface_vtk.SetName("OSI")
            wssint_surface_vtk = numpy_support.numpy_to_vtk(wssint_surface)
            wssint_surface_vtk.SetName("WSSINT")

            # Save the WSS based indices in VTK polydata format
            surface_data.GetOutput().GetPointData().AddArray(tawss_surface_vtk)
            surface_data.GetOutput().GetPointData().AddArray(osi_surface_vtk)
            surface_data.GetOutput().GetPointData().AddArray(wssint_surface_vtk)
            surface_writer = vtk.vtkXMLPolyDataWriter()
            surface_writer.SetInputConnection(surface_data.GetOutputPort())
            surface_writer.SetFileName("%s-wss.vtp" %fname )
            surface_writer.Update()

        # Write XDMF File for visualization in Paraview
        xdmf_out = open("%s.xdmf" % fname, 'w')
        # Header
        xdmf_out.write("""<?xml version="1.0"?>
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Time TimeType="List">\n""")
        # Line of timesteps
        timesteps_str = ' '.join(str(i) for i in range(step_start,step_stop+step_step,step_step))
        xdmf_out.write('<DataItem Format="XML" Dimensions="%d">%s</DataItem>\n</Time>' %(nsteps,timesteps_str) )
        # For each timestep point to grid topology and geometry, and attributes
        for i in range(step_start,step_stop+step_step,step_step):
            xdmf_out.write('<Grid Name="grid_%d" GridType="Uniform">\n' % i)
            xdmf_out.write('<Topology NumberOfElements="%d" TopologyType="Tetrahedron">\n' % ncells)
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 4">%s.h5:/Mesh/topology</DataItem>\n'
                           % (ncells,fname))
            xdmf_out.write('</Topology>\n<Geometry GeometryType="XYZ">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 3">%s.h5:/Mesh/coordinates</DataItem>\n'
                           % (npts, fname))
            xdmf_out.write('</Geometry>\n')

            for group in groups:
                if groups[group] == 1:
                    xdmf_out.write('<Attribute Name="%s" AttributeType="Scalar" Center="Node">\n'
                                   % group)
                if groups[group] == 3:
                    xdmf_out.write('<Attribute Name="%s" AttributeType="Vector" Center="Node">\n'
                                   % group)
                xdmf_out.write('<DataItem Format="HDF" Dimensions="%d %d">%s.h5:/%s/%d</DataItem>\n'
                                   % (npts,groups[group],fname,group,i))
                xdmf_out.write('</Attribute>\n')
            xdmf_out.write('</Grid>\n')
        xdmf_out.write('</Grid>\n</Domain>\n</Xdmf>')
        xdmf_out.close()    
        

    fout.close()

if __name__ == '__main__':
    # Command line arguments
    fname = sys.argv[1]
    step_start = int(sys.argv[2])
    step_stop  = int(sys.argv[3])
    step_step  = int(sys.argv[4])
    mesh_fname = sys.argv[5]
    surface_fname = sys.argv[6]
    extract_velocity_only = False
    compute_lambdatwo = True

    vis2hdf5(fname,
             step_start,
             step_stop,
             step_step,
             mesh_fname,
             surface_fname,
             extract_velocity_only,
             compute_lambdatwo)
