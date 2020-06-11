from __future__ import print_function
#####################################
####### PARTICLETRACKING ############
#####################################
#
# Lagrangian particle advection. Requires HDF5, MPI4PY, H5PY, NUMPY. 
# Runs in parallel. Computes PLAP, PRT, and FTLE
#
# Command line arguments:
# 1. fname -- string indicating the representative name of the simulation
# 2. cycle_start -- integer indicating initial timestep of the cardiac cycle
# 3. cycle_stop -- integer indicating final timestep of the cardiac cycle
# 4. cycle_step -- integer indicating interval between saved timesteps in the cardiac cycle
# 5. step_start -- integer indicating particle injection timestep
# 6. ncycles -- total number of cardiac cycles 
# 7. dt -- float indicating time interval between saved timesteps
# 8. disp -- flag integer indicating whether simulation is deformable (1) or rigid (0)
# 9. repartition -- integer indicating the frequency of repartitioning and saving
#############################################################

# WARNING: There's no guarantee that particles won't get advected out of the domain through the walls!

from builtins import zip
from builtins import str
from builtins import range
import numpy as np
import vtk
from vtk.util import numpy_support
import sys
import datetime
# Be sure that Python finds the parallel HDF5 e.g. editing the following line
#sys.path.insert(1,'/home/fas/humphrey/pd283/Work/FEniCS/lib/python2.7/site-packages')
import shutil
import os,errno
import subprocess
import mpi4py, mpi4py.MPI
import h5py

### READ COMMAND LINE ARGUMENTS
fname       = sys.argv[1]
cycle_start = int(sys.argv[2])
cycle_stop  = int(sys.argv[3])
cycle_step  = int(sys.argv[4])
step_start  = int(sys.argv[5])
ncycles     = int(sys.argv[6])
dt          = float(sys.argv[7])
disp        = int(sys.argv[8])
repartition = int(sys.argv[9])
repartition = repartition*cycle_step

final_step = step_start + (cycle_stop - cycle_start + cycle_step)*ncycles
### MPI INITIALIZTION
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


#***************** PARTICLES INJECTION PREPARATION ******************************************
domain_particles = vtk.vtkXMLUnstructuredGridReader()
domain_particles.SetFileName("particles.vtu")
domain_particles.Update()
particles_coordinates = np.copy(numpy_support.vtk_to_numpy(domain_particles.GetOutput().GetPoints().GetData()))
particles_vtk_pts = vtk.vtkPoints()
particles_vtk_pts.SetData(numpy_support.numpy_to_vtk(particles_coordinates))
nparticles = domain_particles.GetOutput().GetNumberOfPoints()
nliveparticles = nparticles
particles_indices = np.arange(nparticles,dtype=np.int64) # initially particles are 'tidy'
particles_offsets = np.zeros((nprocs+1,),dtype=np.int64)

#********************** I/O PREPARATION ***************************************************
# Prepare input and output files (HDF5)
results = h5py.File("%s.h5" % fname,'r',driver='mpio',comm=comm)
output  = h5py.File("%s-particles-%d.h5" % (fname, step_start),'w',driver='mpio',comm=comm)
output.create_group("Index")
output.create_group("Coordinates")
output.create_group("Topology")
output.create_group("PLAP")
output.create_group("PRT")
output.create_group("alive")
output.create_group("velocity")
output.create_group("partition")
output.create_group("TimeWritten")
# Also create group for final step. Here we can write results progressively when particles exit the domain
output["Index"].create_group("Final")
output["Coordinates"].create_group("Final")
output["Topology"].create_group("Final")
output["PLAP"].create_group("Final")
output["PRT"].create_group("Final")
output["alive"].create_group("Final")
output["velocity"].create_group("Final")
output["partition"].create_group("Final")
output["TimeWritten"].create_group("Final")
for i in range(nprocs):
    output["Index/Final"].create_dataset("%d" % i, (nparticles,), dtype=np.int64)    
    output["Coordinates/Final"].create_dataset("%d" % i, (nparticles,3), dtype="f")
    output["PLAP/Final"].create_dataset("%d" % i, (nparticles,), dtype="f")
    output["PRT/Final"].create_dataset("%d" % i, (nparticles,), dtype="f")
    output["alive/Final"].create_dataset("%d" % i, (nparticles,), dtype=np.int64)
    output["velocity/Final"].create_dataset("%d" % i, (nparticles,3), dtype="f")
    output["partition/Final"].create_dataset("%d" % i, (nparticles,), dtype=np.int64)
    output["TimeWritten/Final"].create_dataset("%d" % i, (nparticles,), dtype=np.int64)
output["Index/Final/%d" % rank][:] = -1
output["PLAP/Final/%d" % rank][:] = -1
output["PRT/Final/%d" % rank][:] = -1
output["Coordinates/Final"].create_dataset("%d" % (nprocs+1,), (nparticles,4),dtype="f")
output["PLAP/Final"].create_dataset("%d" % (nprocs+1,), (nparticles,2),dtype="f")
output["PRT/Final"].create_dataset("%d" % (nprocs+1,), (nparticles,2),dtype="f")
final_offset = 0 
#***********************************************************************************************

#****************************Fluid Domain****************************************************
# Build domain from results file
domain       = vtk.vtkUnstructuredGrid()
domain_pts   = results["/Mesh/coordinates"][:]
npts = domain_pts.shape[0]
domain_cells = results["/Mesh/topology"][:]
ncells = domain_cells.shape[0]
print("cells: %d" % ncells)
domain_cells_arr = np.zeros((ncells,5),dtype=np.int64)
domain_cells_arr[:,0]  = 4
domain_cells_arr[:,1:] = domain_cells
domain_cells_arr = np.reshape(domain_cells_arr,(ncells*5,))
domain_pts_vtk   = vtk.vtkPoints()
domain_pts_vtk.SetData(numpy_support.numpy_to_vtk(domain_pts))
domain.SetPoints(domain_pts_vtk)
domain_cells_vtk = vtk.vtkCellArray()
domain_cells_vtk.SetCells(ncells,numpy_support.numpy_to_vtkIdTypeArray(domain_cells_arr))
domain.SetCells(vtk.VTK_TETRA,domain_cells_vtk)
domain_locator = vtk.vtkPointLocator()
domain_locator.SetDataSet(domain)
domain_locator.BuildLocator()
velocity_arr  = np.zeros((npts,3))
velocity_vtk  = numpy_support.numpy_to_vtk(velocity_arr)
velocity_vtk.SetName("velocity")
frobenius_arr = np.zeros((npts,1))
frobenius_vtk = numpy_support.numpy_to_vtk(frobenius_arr)
frobenius_vtk.SetName("frobenius")
disp_arr = np.zeros((npts,3))
disp_vtk = numpy_support.numpy_to_vtk(disp_arr)
disp_vtk.SetName("displacement")
domain.GetPointData().AddArray(velocity_vtk)
domain.GetPointData().AddArray(frobenius_vtk)
# Get wall normals to compute inward velocity array
surface_reader = vtk.vtkXMLPolyDataReader()
surface_reader.SetFileName("rp-sp.vtp")
surface_reader.Update()
surface_normals = vtk.vtkPolyDataNormals()
surface_normals.SetInputConnection(surface_reader.GetOutputPort())
surface_normals.Update()
surface_normals_locator = vtk.vtkPointLocator()
surface_normals_locator.SetDataSet(surface_normals.GetOutput())
surface_normals_locator.BuildLocator()
allwnodes = open("allwnodes.nbc", 'r')
cnt_lines = 0
allwnodes_list = []
for line in allwnodes:
    allwnodes_list.append(int(line.split()[0]))
allwnodes_list = list(set(allwnodes_list)) # remove duplicates
allwnodes_list.sort()
allwnpts = len(allwnodes_list)
map_wall_to_domain = np.array(allwnodes_list,dtype=np.int64)
map_wall_to_normals = np.zeros((allwnpts,),dtype=np.int64)
for i in range(allwnpts):
    tmp_pt = domain_pts[map_wall_to_domain[i]]
    map_wall_to_normals[i] = surface_normals_locator.FindClosestPoint(tmp_pt)
wall_normals = numpy_support.vtk_to_numpy(surface_normals.GetOutput().GetPointData().GetArray("Normals"))[map_wall_to_normals]
wall_velocity = -10.0*wall_normals


# If simulation was deformable wall get also displacement and deform to current state
if disp == 1:
    reference_displacement = results["/displacement/%d" % cycle_start][:]
    disp_arr = results["/displacement/%d" % step_start][:]
    domain_pts[map_wall_to_domain] += disp_arr[map_wall_to_domain]-reference_displacement[map_wall_to_domain]
    reference_displacement[:] = disp_arr 
else:
    reference_displacement = None


# Split particles among processes for probefilter input
particles_partition = np.zeros((nliveparticles,),dtype=np.int64)
if rank ==0:
    kdtree = vtk.vtkKdTree()
    kdtree.SetNumberOfRegionsOrLess(nprocs)
    kdtree.BuildLocatorFromPoints(particles_vtk_pts)
    offset=0
    for i in range(nprocs):
        if i < kdtree.GetNumberOfRegions():
            points_in_regions = numpy_support.vtk_to_numpy(kdtree.GetPointsInRegion(i))
            offset += points_in_regions.shape[0]
            particles_partition[points_in_regions]=i        
        particles_offsets[i+1] = offset
print("Broadcasting...")
comm.Bcast(particles_offsets,root=0)
comm.Bcast(particles_partition,root=0)
map_local_to_global       = particles_indices[np.where(particles_partition==rank)]
# Create polydata containing local particles coordinates. For Runge Kutta integration we need 4
local_particles_coordinates = particles_coordinates[map_local_to_global]
nlocalparticles             = local_particles_coordinates.shape[0]
local_particles_topology    = np.zeros((nlocalparticles,2),dtype=np.int64)
local_particles_topology[:,0] = 1
local_particles_topology[:,1] = np.arange(nlocalparticles,dtype=np.int64)
local_particles_topology = np.reshape(local_particles_topology,(nlocalparticles*2,1))

local_particles = vtk.vtkPolyData()
local_particles_pts = vtk.vtkPoints()
local_particles_pts.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates))
local_particles_cells = vtk.vtkCellArray()
local_particles_cells.SetCells(nlocalparticles,numpy_support.numpy_to_vtkIdTypeArray(local_particles_topology))
local_particles.SetPoints(local_particles_pts)
local_particles.SetVerts(local_particles_cells)

local_particles_coordinates_k1 = np.copy(local_particles_coordinates)
local_particles_k1 = vtk.vtkPolyData()
local_particles_pts_k1 = vtk.vtkPoints()
local_particles_pts_k1.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k1))
local_particles_k1.SetPoints(local_particles_pts_k1)
local_particles_k1.SetVerts(local_particles_cells)

local_particles_coordinates_k2 = np.copy(local_particles_coordinates)
local_particles_k2 = vtk.vtkPolyData()
local_particles_pts_k2 = vtk.vtkPoints()
local_particles_pts_k2.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k2))
local_particles_k2.SetPoints(local_particles_pts_k2)
local_particles_k2.SetVerts(local_particles_cells)

local_particles_coordinates_k3 = np.copy(local_particles_coordinates)
local_particles_k3 = vtk.vtkPolyData()
local_particles_pts_k3 = vtk.vtkPoints()
local_particles_pts_k3.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k3))
local_particles_k3.SetPoints(local_particles_pts_k3)
local_particles_k3.SetVerts(local_particles_cells)

local_particles_coordinates_k4 = np.copy(local_particles_coordinates)
local_particles_k4 = vtk.vtkPolyData()
local_particles_pts_k4 = vtk.vtkPoints()
local_particles_pts_k4.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k4))
local_particles_k4.SetPoints(local_particles_pts_k4)
local_particles_k4.SetVerts(local_particles_cells)

plap          = np.zeros((nlocalparticles,))
prt           = np.zeros((nlocalparticles,))
t_step        = step_start

saved_steps = []
saved_nliveparticles = []

time_start = datetime.datetime.now()
print("entering cycles")
for ccc in range(ncycles):
    cur_step = step_start + cycle_step
    if (cycle_step < 0) and  (cur_step < cycle_stop) : cur_step = cycle_start
    if (cycle_step > 0) and  (cur_step > cycle_stop) : cur_step = cycle_start
    for ttt in range(cycle_start,cycle_stop+cycle_step,cycle_step):        
        # Update velocity and frobenius in domain unstructured grid
        velocity_arr[:]  = results["/velocity/%d" % cur_step][:]
        frobenius_arr[:] = results["/frobenius/%d" % cur_step][:]
        # Add inward velocity component
        velocity_arr[map_wall_to_domain] += wall_velocity
        # If deformable wall, deform wall domain
        if disp == 1:
            disp_arr = results["/displacement/%d" % cur_step][:]
            domain_pts[map_wall_to_domain] += disp_arr[map_wall_to_domain]-reference_displacement[map_wall_to_domain]
            reference_displacement[:] = disp_arr       

        # Probe at first location
        alive_k1 = np.array([0,]) # Empty initialization
        if nlocalparticles > 0 :
            probe_k1 = vtk.vtkProbeFilter()
            probe_k1.SetValidPointMaskArrayName("alive")
            probe_k1.SetInput(local_particles_k1)
            probe_k1.SetSource(domain)
            probe_k1.Update()
            velocity_k1  = numpy_support.vtk_to_numpy(probe_k1.GetOutput().GetPointData().GetArray("velocity"))
            frobenius_k1 = numpy_support.vtk_to_numpy(probe_k1.GetOutput().GetPointData().GetArray("frobenius"))
            alive_k1 = numpy_support.vtk_to_numpy(probe_k1.GetOutput().GetPointData().GetArray("alive"))
            plap += np.fabs(dt)*frobenius_k1
            prt  += np.fabs(dt)

        # Probe at second location
            local_particles_coordinates_k2 += 0.5*dt*velocity_k1
        
            probe_k2 = vtk.vtkProbeFilter()
            probe_k2.SetValidPointMaskArrayName("alive")
            probe_k2.SetInput(local_particles_k2)
            probe_k2.SetSource(domain)
            probe_k2.Update()
            velocity_k2 = numpy_support.vtk_to_numpy(probe_k2.GetOutput().GetPointData().GetArray("velocity"))

        # Probe at third location
            local_particles_coordinates_k3 += 0.5*dt*velocity_k2
        
            probe_k3 = vtk.vtkProbeFilter()
            probe_k3.SetValidPointMaskArrayName("alive")
            probe_k3.SetInput(local_particles_k3)
            probe_k3.SetSource(domain)
            probe_k3.Update()
            velocity_k3 = numpy_support.vtk_to_numpy(probe_k3.GetOutput().GetPointData().GetArray("velocity"))

        # Probe at fourth location
            local_particles_coordinates_k4 += dt*velocity_k3            
            probe_k4 = vtk.vtkProbeFilter()
            probe_k4.SetValidPointMaskArrayName("alive")
            probe_k4.SetInput(local_particles_k3)
            probe_k4.SetSource(domain)
            probe_k4.Update()
            velocity_k4 = numpy_support.vtk_to_numpy(probe_k3.GetOutput().GetPointData().GetArray("velocity"))          
        # Update position
            local_particles_coordinates_k1 += dt/6.0*(velocity_k1+2.0*velocity_k2+2.0*velocity_k3+velocity_k4)        
            local_particles_coordinates_k2[:] = local_particles_coordinates_k1
            local_particles_coordinates_k3[:] = local_particles_coordinates_k1
            local_particles_coordinates_k4[:] = local_particles_coordinates_k1

            local_particles_k1.Update()
            local_particles_k2.Update()
            local_particles_k3.Update()
            local_particles_k4.Update()

        # If necessary save data and repartition
        if (t_step-step_start) % repartition == 0:
        # if True
            time_stop = datetime.datetime.now()
            time_need = (time_stop-time_start).seconds
            time_start = time_stop
            saved_steps.append(t_step)
            saved_nliveparticles.append(nliveparticles)
            output["Index"].create_dataset("%d" % t_step, (nliveparticles,), dtype=np.int64)
            output["Topology"].create_dataset("%d" % t_step, (nliveparticles,), dtype=np.int64)
            output["Coordinates"].create_dataset("%d" % t_step, (nliveparticles,3))
            output["PLAP"].create_dataset("%d" % t_step, (nliveparticles,), dtype="f")
            output["PRT"].create_dataset("%d" % t_step, (nliveparticles,), dtype="f")
            output["alive"].create_dataset("%d" % t_step, (nliveparticles,), dtype=np.int64)
            output["velocity"].create_dataset("%d" % t_step, (nliveparticles,3), dtype="f")
            output["partition"].create_dataset("%d" % t_step, (nliveparticles,), dtype=np.int64)
            if nlocalparticles > 0 :
                output["/Index/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]] = map_local_to_global
                output["/Topology/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]] = np.arange(particles_offsets[rank],particles_offsets[rank+1],dtype=np.int64)
                output["/Coordinates/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]] = local_particles_coordinates_k1
                output["/PLAP/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]        = plap
                output["/PRT/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]         = prt
                output["/alive/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]       = alive_k1
                output["/velocity/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]    = velocity_k1
                output["/partition/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]   = rank
                indices_dead = map_local_to_global[np.where(alive_k1==0)]
                nlocaldead = indices_dead.shape[0]
                output["/Coordinates/Final/%d" % rank][final_offset:final_offset+nlocaldead] = local_particles_coordinates_k1[np.where(alive_k1==0)]
                output["/PLAP/Final/%d" % rank][final_offset:final_offset+nlocaldead] = plap[np.where(alive_k1==0)]
                output["/PRT/Final/%d" % rank][final_offset:final_offset+nlocaldead]  = prt[np.where(alive_k1==0)]
                output["/Index/Final/%d" % rank][final_offset:final_offset+nlocaldead] = indices_dead
                output["/TimeWritten/Final/%d" % rank][final_offset:final_offset+nlocaldead] = t_step
                final_offset += nlocaldead
            time_save_stop = datetime.datetime.now()
            time_save = (time_save_stop-time_start).seconds
            time_start = time_save_stop
            print("Cycle: %d Rank: %d Current timestep: %d, nparticles: %d, prt_active: %d, nliveparticles: %d, local_coords:%d, time_interp: %d, time_save:%d" % (ccc,rank,cur_step,nlocalparticles,np.sum(alive_k1),nliveparticles,local_particles_coordinates_k1.shape[0],time_need,time_save))  
            # Gather all updated particles and repartition            
            comm.Barrier() 
            alive_global   = np.array(output["/alive/%d" % t_step][:],dtype=np.bool)
            indices_global = output["/Index/%d" % t_step][:]
            plap_global    = output["/PLAP/%d" % t_step][:]
            plap_global    = plap_global[alive_global]
            prt_global     = output["/PRT/%d" % t_step][:]
            prt_global     = prt_global[alive_global]
            coordinates_global    = output["/Coordinates/%d" % t_step][:]     
            particles_indices     = indices_global[alive_global]
            particles_coordinates = np.copy(coordinates_global[alive_global])
            nliveparticles        = particles_indices.shape[0]           
            particles_partition   = np.zeros((nliveparticles,),dtype=np.int64)
            if rank == 0:                
                print("Repartitioning...")
                particles_vtk_pts = vtk.vtkPoints()
                particles_vtk_pts.SetData(numpy_support.numpy_to_vtk(particles_coordinates))                
                kdtree = vtk.vtkKdTree()
                kdtree.SetNumberOfRegionsOrLess(nprocs)
                kdtree.BuildLocatorFromPoints(particles_vtk_pts)
                offset = 0
                for i in range(nprocs):                    
                    if i < kdtree.GetNumberOfRegions():
                        points_in_regions = numpy_support.vtk_to_numpy(kdtree.GetPointsInRegion(i))
                        offset += points_in_regions.shape[0]
                        particles_partition[points_in_regions]=i                
                    particles_offsets[i+1] = offset
            # Broadcast new particles_partition and rebuild maps and local datasets
            comm.Bcast(particles_offsets,root=0)  
            comm.Bcast(particles_partition,root=0)            
            if rank ==0 : print("Broadcasted...")
            map_local_to_global       = particles_indices[np.where(particles_partition==rank)]
            nlocalparticles = map_local_to_global.shape[0]
            if nlocalparticles > 0:
            # Get PLAP and PRT of local particles
                plap = plap_global[np.where(particles_partition==rank)]
                prt  = prt_global[np.where(particles_partition==rank)]
                local_particles_coordinates = particles_coordinates[np.where(particles_partition==rank)]

############# ADDED BY KEVIN ##############################################################################
			# Get repartitioned alive and velocity arrays
                alive_repartitioned = alive_global[np.where(particles_partition==rank)]
                
                velocity_repartitioned = output["/velocity/%d" % t_step][:]
                velocity_repartitioned = velocity_repartitioned[alive_global]
                velocity_repartitioned = velocity_repartitioned[np.where(particles_partition==rank)]
############# END #########################################################################################

            # Update polydata containing local particles coordinates
                local_particles_topology      = np.zeros((nlocalparticles,2),dtype=np.int64)
                local_particles_topology[:,0] = 1
                local_particles_topology[:,1] = np.arange(nlocalparticles,dtype=np.int64)
                local_particles_topology = np.reshape(local_particles_topology,(nlocalparticles*2,1))

                local_particles = vtk.vtkPolyData()
                local_particles_pts = vtk.vtkPoints()
                local_particles_pts.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates))
                local_particles_cells = vtk.vtkCellArray()
                local_particles_cells.SetCells(nlocalparticles,numpy_support.numpy_to_vtkIdTypeArray(local_particles_topology))
                local_particles.SetPoints(local_particles_pts)
                local_particles.SetVerts(local_particles_cells)

                local_particles_coordinates_k1 = np.copy(local_particles_coordinates)
                local_particles_k1 = vtk.vtkPolyData()
                local_particles_pts_k1 = vtk.vtkPoints()
                local_particles_pts_k1.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k1))
                local_particles_k1.SetPoints(local_particles_pts_k1)
                local_particles_k1.SetVerts(local_particles_cells)

                local_particles_coordinates_k2 = np.copy(local_particles_coordinates)
                local_particles_k2 = vtk.vtkPolyData()
                local_particles_pts_k2 = vtk.vtkPoints()
                local_particles_pts_k2.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k2))
                local_particles_k2.SetPoints(local_particles_pts_k2)
                local_particles_k2.SetVerts(local_particles_cells)
            
                local_particles_coordinates_k3 = np.copy(local_particles_coordinates)
                local_particles_k3 = vtk.vtkPolyData()
                local_particles_pts_k3 = vtk.vtkPoints()
                local_particles_pts_k3.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k3))
                local_particles_k3.SetPoints(local_particles_pts_k3)
                local_particles_k3.SetVerts(local_particles_cells)

                local_particles_coordinates_k4 = np.copy(local_particles_coordinates)
                local_particles_k4 = vtk.vtkPolyData()
                local_particles_pts_k4 = vtk.vtkPoints()
                local_particles_pts_k4.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k4))
                local_particles_k4.SetPoints(local_particles_pts_k4)
                local_particles_k4.SetVerts(local_particles_cells)

        # Update cur_step and tstep
        cur_step += cycle_step
        if (cycle_step > 0) and cur_step > cycle_stop: cur_step = cycle_start
        if (cycle_step < 0) and cur_step < cycle_stop: cur_step = cycle_start

        t_step += cycle_step

# Save results at the end of cycle through cycles
print("Cycle: %d Rank: %d Current timestep: %d, nparticles: %d, particles_active: %d, local_coordinates:%d" % (ccc,rank,cur_step,nlocalparticles,np.sum(alive_k1),local_particles_coordinates_k1.shape[0]))
saved_steps.append(t_step)
saved_nliveparticles.append(nliveparticles)
output["Index"].create_dataset("%d" % t_step, (nliveparticles,), dtype=np.int64)
output["Topology"].create_dataset("%d" % t_step, (nliveparticles,), dtype=np.int64)
output["Coordinates"].create_dataset("%d" % t_step, (nliveparticles,3))
output["PLAP"].create_dataset("%d" % t_step, (nliveparticles,), dtype="f")
output["PRT"].create_dataset("%d" % t_step, (nliveparticles,), dtype="f")
output["alive"].create_dataset("%d" % t_step, (nliveparticles,), dtype=np.int64)
output["velocity"].create_dataset("%d" % t_step, (nliveparticles,3), dtype="f")
output["partition"].create_dataset("%d" % t_step, (nliveparticles,), dtype=np.int64)
if nlocalparticles > 0 :
    output["/Index/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]] = map_local_to_global
    output["/Topology/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]] = np.arange(particles_offsets[rank],particles_offsets[rank+1],dtype=np.int64)
    output["/Coordinates/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]] = local_particles_coordinates_k1
    output["/PLAP/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]        = plap
    output["/PRT/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]         = prt
##### ADDED BY KEVIN ###############################################################################################
    # Use repartitioned arrays 
    output["/alive/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]       = alive_repartitioned    
    output["/velocity/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]    = velocity_repartitioned
##### END ##########################################################################################################
    output["/partition/%d" % t_step][particles_offsets[rank]:particles_offsets[rank+1]]   = rank
    output["/Coordinates/Final/%d" % rank][final_offset:final_offset+nlocalparticles]     = local_particles_coordinates_k1
    output["/PLAP/Final/%d" % rank][final_offset:final_offset+nlocalparticles]  = plap
    output["/PRT/Final/%d" % rank][final_offset:final_offset+nlocalparticles]   = prt
    output["/Index/Final/%d" % rank][final_offset:final_offset+nlocalparticles] = map_local_to_global
    output["/TimeWritten/Final/%d" % rank][final_offset:final_offset+nlocalparticles] = t_step
        
# Write VTK unstructured Grid with results at the end of the advection
comm.Barrier()
if rank ==0:
    print("Final Step: %d, Time Step: %d" % (final_step, t_step))
    final_indices = np.zeros(nparticles,dtype=np.int64)
    final_coordinates = np.zeros((nparticles,3),dtype=np.float)
    final_plap    = np.zeros(nparticles,dtype=np.float)
    final_prt     = np.zeros(nparticles,dtype=np.float)
    final_offset  = 0
    print("Reducing Final Step Results...")
    for i in range(nprocs):
        indices_proc = output["/Index/Final/%d" % i][:]
        indices_proc = indices_proc[np.where(indices_proc != -1)]
        coordinates_proc = output["/Coordinates/Final/%d" % i][:]
        coordinates_proc = coordinates_proc[np.where(indices_proc != -1)]
        plap_proc = output["/PLAP/Final/%d" % i][:]
        plap_proc = plap_proc[np.where(indices_proc != -1)]
        prt_proc = output["/PRT/Final/%d" % i][:]
        prt_proc = prt_proc[np.where(indices_proc != -1)]
        nindices = indices_proc.shape[0]
        final_indices[final_offset:final_offset+nindices] = indices_proc
        final_coordinates[final_offset:final_offset+nindices] = coordinates_proc
        final_plap[final_offset:final_offset+nindices] = plap_proc
        final_prt[final_offset:final_offset+nindices]  = prt_proc
        final_offset += nindices
    sort_indices = np.argsort(final_indices)
    output["Coordinates/Final/%d" % (nprocs+1,)][:,0] = final_indices[sort_indices]
    sorted_coordinates = final_coordinates[sort_indices]
    output["Coordinates/Final/%d" % (nprocs+1,)][:,1:] = sorted_coordinates
    output["PLAP/Final/%d" % (nprocs+1,)][:,0] = final_indices[sort_indices]
    sorted_plap = final_plap[sort_indices]
    output["PLAP/Final/%d" % (nprocs+1,)][:,1] = sorted_plap
    output["PRT/Final/%d" % (nprocs+1,)][:,0] = final_indices[sort_indices]
    sorted_prt = final_prt[sort_indices]
    output["PRT/Final/%d" % (nprocs+1,)][:,1] = sorted_prt
    coordinates_vtu_vtk = numpy_support.numpy_to_vtk(sorted_coordinates)
    coordinates_vtu_vtk.SetName("Coordinates")
    plap_vtu_vtk = numpy_support.numpy_to_vtk(sorted_plap)
    plap_vtu_vtk.SetName("PLAP")
    prt_vtu_vtk = numpy_support.numpy_to_vtk(sorted_prt)
    prt_vtu_vtk.SetName("PRT")
    domain_particles.GetOutput().GetPointData().AddArray(coordinates_vtu_vtk)
    domain_particles.GetOutput().GetPointData().AddArray(plap_vtu_vtk)
    domain_particles.GetOutput().GetPointData().AddArray(prt_vtu_vtk)
    # Compute Right Cauchy Green tensor for FTLE
    deformation_gradient = vtk.vtkGradientFilter()
    deformation_gradient.SetInputConnection(domain_particles.GetOutputPort())
    deformation_gradient.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,"Coordinates")
    deformation_gradient.Update()
    deformation_gradient_tensor = numpy_support.vtk_to_numpy(deformation_gradient.GetOutput().GetPointData().GetArray("Gradients"))
    deformation_gradient_tensor = np.reshape(deformation_gradient_tensor,(-1,3,3))
    right_cauchy_green_tensor = np.einsum('abj,acj->abc',deformation_gradient_tensor,deformation_gradient_tensor)
    eigvals = np.linalg.eigvals(right_cauchy_green_tensor)
    eigvals = np.sort(eigvals,axis=1)
##### ADDED BY KEVIN ###############################################################################################
    # Compute real part of eigenvalues, as VTK cannot accept complex numbers 
    # Occasionally the imaginary part is present, but equal to 0
    real_eigvals = np.real(eigvals[:,2])
    ftle = np.ascontiguousarray(np.log(real_eigvals))    
##### END ##########################################################################################################
    ftle_vtu_vtk = numpy_support.numpy_to_vtk(ftle)
    ftle_vtu_vtk.SetName("FTLE")
    domain_particles.GetOutput().GetPointData().AddArray(ftle_vtu_vtk)    
    particles_writer = vtk.vtkXMLUnstructuredGridWriter()
    particles_writer.SetInputConnection(domain_particles.GetOutputPort())
    particles_writer.SetFileName("%s-particles-%d.vtu" % (fname, step_start))
    particles_writer.Update()
    print("...Done!")

# Write XDMF File for Visualization in Paraview
if rank ==0:
    xdmf_out = open("%s-particles-%d.xdmf" % (fname,step_start), 'w')
    xdmf_out.write("""<?xml version="1.0"?>
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Time TimeType="List">\n""")
    timesteps_str = ' '.join(str(i) for i in saved_steps)
    nsteps = len(saved_steps)
    xdmf_out.write('<DataItem Format="XML" Dimensions="%d">%s</DataItem>\n</Time>' %(nsteps,timesteps_str) )
    # For each timestep point to grid topology and geometry, and attributes
    for i,nliveparticles in zip(saved_steps,saved_nliveparticles):
        xdmf_out.write('<Grid Name="grid_%d" GridType="Uniform">\n' % i)
        xdmf_out.write('<Topology NumberOfElements="%d" TopologyType="Polyvertex">\n' % nliveparticles)
        xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s-particles-%d.h5:/Topology/%d</DataItem>\n'
                       % (nliveparticles,fname,step_start,i))
        xdmf_out.write('</Topology>\n<Geometry GeometryType="XYZ">\n')
        xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 3">%s-particles-%d.h5:/Coordinates/%d</DataItem>\n'
                   % (nliveparticles,fname,step_start,i))
        xdmf_out.write('</Geometry>\n')
    
        xdmf_out.write('<Attribute Name="PLAP" AttributeType="Scalar" Center="Cell">\n')
        xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s-particles-%d.h5:/PLAP/%d</DataItem>\n'
                                   % (nliveparticles,fname,step_start,i))
        xdmf_out.write('</Attribute>\n')
        xdmf_out.write('<Attribute Name="PRT" AttributeType="Scalar" Center="Cell">\n')
        xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s-particles-%d.h5:/PRT/%d</DataItem>\n'
                                   % (nliveparticles,fname,step_start,i))
        xdmf_out.write('</Attribute>\n')
        xdmf_out.write('<Attribute Name="alive" AttributeType="Scalar" Center="Cell">\n')
        xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s-particles-%d.h5:/alive/%d</DataItem>\n'
                                   % (nliveparticles,fname,step_start,i))
        xdmf_out.write('</Attribute>\n')
        xdmf_out.write('<Attribute Name="partition" AttributeType="Scalar" Center="Cell">\n')
        xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s-particles-%d.h5:/partition/%d</DataItem>\n'
                                   % (nliveparticles,fname,step_start,i))
        xdmf_out.write('</Attribute>\n')
        xdmf_out.write('<Attribute Name="velocity" AttributeType="Vector" Center="Cell">\n')
        xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 3">%s-particles-%d.h5:/velocity/%d</DataItem>\n'
                                   % (nliveparticles,fname,step_start,i))
        xdmf_out.write('</Attribute>\n')
        xdmf_out.write('</Grid>\n')
    xdmf_out.write('</Grid>\n</Domain>\n</Xdmf>')
    xdmf_out.close()
                        
                       
results.close()
output.close()
        

