'''
PARTICLE TRACKING

Lagrangian particle advection. Requires HDF5, MPI4PY, H5PY, NUMPY.
Runs in parallel. Computes PLAP, PRT, and FTLE

Note that you must be careful with your start and end steps, because this code
loops a single cycle lots of times to repeat the advection over several
identical cardiac cycles.

This means that you should include restarts in the range
[cycle start, cycle end) (note the half-open interval) so that when the code
repeats the cycles, you don't get the same point in the cardiac cycle repeated
twice at the interface between two cycle repeats (as you would get with
repeating [cycle start, cycle end] ).

Command line arguments:
DEPRECATED - config should now be placed in particle_config.json.
See example_config.json in the mercurial repository.

1. fname -- string indicating the representative name of the simulation
2. cycle_start -- integer indicating initial timestep of the cardiac cycle
3. cycle_stop -- integer indicating final timestep of the cardiac cycle
4. cycle_step -- integer indicating interval between saved timesteps in the
cardiac cycle
5. step_start -- integer indicating particle injection timestep
6. ncycles -- total number of cardiac cycles
7. dt -- float indicating time interval between saved timesteps
8. disp -- flag integer indicating whether simulation is deformable (1) or
rigid (0)
# 9. repartition -- integer indicating the frequency of repartitioning and
saving

WARNING: There's no guarantee that particles won't get advected out of the
domain through the walls!
'''
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
import h5py
import numpy as np
import vtk
from vtk.util import numpy_support
import datetime
import mpi4py
import mpi4py.MPI
import argparse
import warnings
import os
import tqdm

import copy

import particles

import math

import time

import sys
# def trace(frame, event, arg):
#     comm = mpi4py.MPI.COMM_WORLD
#     if comm.Get_rank() == 0:
#         print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
#     return trace
# sys.settrace(trace)

def parseCommandLineArguments():
    command_line_argument_parser = argparse.ArgumentParser(description='''
        Performs Lagrangian particle tracking,
        including computation of vorticity fields
        for each time-step, and tracking helical
        flow index accumulation for each particle.
        Designed for modification: additional time
        bins can be added to the source code in
        order to examine particle-based indices
        at different points in the cardiac cycle.
        ''')
    # command_line_argument_parser.add_argument(
    #     'fname',
    #     help='''
    #         Name of input file containing flow simuluation
    #         results. Must be in hdf5 format. Give base
    #         name only without the file extension.
    #         ''',
    #         type=str
    #     )
    # command_line_argument_parser.add_argument(
    #     'cycle_start',
    #     help='''
    #         Time-step index in the input data of the
    #         start of the flow data. The specified
    #         cycle will be repeated ncycles times.
    #         ''',
    #         type=int
    #     )
    # command_line_argument_parser.add_argument(
    #     'cycle_stop',
    #     help='''
    #     Time-step index in the input data of the
    #     end of the flow data. The specified
    #     cycle will be repeated ncycles times.
    #     ''',
    #     type=int
    #     )
    # command_line_argument_parser.add_argument(
    #     'cycle_step',
    #     help='''
    #     Number of time-steps to advance by at once
    #     during the cycle.
    #     ''',
    #     type=int
    #     )
    # command_line_argument_parser.add_argument(
    #     'step_start',
    #     help='''
    #     The time-step to start from at the beginning
    #     of the particle tracking simulation.
    #     ''',
    #     type=int
    #     )
    # command_line_argument_parser.add_argument(
    #     'ncycles',
    #     help='''
    #     Number of times to repeat the specified
    #     flow data cycle.
    #     ''',
    #     type=int
    #     )
    # command_line_argument_parser.add_argument(
    #     'dt',
    #     help='''
    #     Amount of real time represented by cycle_step.
    #     Equal to cycle_step * Navier-Stokes simulation
    #     time-step.
    #     ''',
    #     type=float
    #     )
    # command_line_argument_parser.add_argument(
    #     'disp',
    #     help='''
    #     Indicates whether the simulation is deformable.
    #     Input 0 for rigid; 1 for deformable.
    #     ''',
    #     type=int
    #     )
    # # WARNING this affects the output frequency too; it only
    # # writes out on repartition! todo fix this
    # command_line_argument_parser.add_argument(
    #     'repartition_frequency_in_simulation_steps',
    #     help='''
    #     Repartition and data-output frequency, given in
    #     particle tracking steps (i.e. number of cycle_step
    #     steps).
    #     ''',
    #     type=float
    #     )

    # # Helicity systole/diastole tracking parameters:
    # command_line_argument_parser.add_argument(
    #     'simulationStartTime',
    #     help='''
    #     Real-time in seconds within the input flow
    #     data that the simulation begins.
    #     ''',
    #     type=float
    #     )
    # command_line_argument_parser.add_argument(
    #     'firstSystoleStartTime',
    #     help='''
    #     Start-time in seconds of the start of the first
    #     systole in the input flow data.
    #     ''',
    #     type=float
    #     )
    # command_line_argument_parser.add_argument(
    #     'firstSystoleEndTime',
    #     help='''
    #     End-time in seconds of the start of the first
    #     systole in the input flow data.
    #     ''',
    #     type=float
    #     )
    # command_line_argument_parser.add_argument(
    #     'cardiacCycleLength',
    #     help='''
    #     Duration in seconds of one cardiac cycle in the
    #     input flow data.
    #     ''',
    #     type=float
    #     )
    command_line_argument_parser.add_argument(
        '-p',
        dest='particle_vtu_file',
        default='./particles.vtu',
        help='''
        File containing the particles, defined by the
        nodes of a .vtu mesh.
        ''',
        type=str
        )
    command_line_argument_parser.add_argument(
        '-w',
        dest='allwnodes_nbc_file',
        default='./allwnodes.nbc',
        help='''
        File containing all the nodes which comprise the
        vessel wall. Should include true wall nodes only,
        not those on the flow surfaces.
        ''',
        type=str
        )
    command_line_argument_parser.add_argument(
        '-s',
        dest='surface_vtp_file',
        default='rp-sp.vtp',
        help='''
        .vtp file giving the surface of the geometric model
        of the vasculature.
        ''',
        type=str
        )
    command_line_argument_parser.add_argument(
        '-c',
        dest='custom_config_file_name',
        default='particle_config.json',
        help='''
        .json file specifying custom options for the
        particle tracking. Overrides the defaults found
        in the particle source code folder in the file
        default_config.json.
        ''',
        type=str
        )

    parsed_aruments = command_line_argument_parser.parse_args()
    parsed_arguments_dict = vars(parsed_aruments)
    print("\n---------------------------------------------")
    print("Running with the following configuration:")
    print("---------------------------------------------")
    for argument in parsed_arguments_dict:
        print(argument, "=", parsed_arguments_dict[argument])
    print("---------------------------------------------\n")

    return parsed_aruments

def createParticleSpacetimeBins(config_manager, firstSystoleStartTime, firstSystoleEndTime, simulationStartTime,cardiacCycleLength):
    # Setup time "bins" within which we require additional seperate reports
    # of particle helicity
    particle_spacetime_bins = particles.ParticleDataTimeBinsSpecifier()    
    
    for bin_name in config_manager.particleBinNames():
        for bin_time_interval in config_manager.particleBinTimeIntervals(bin_name):
            particle_spacetime_bins.addTimeBinInterval(bin_time_interval[0],
                                                  bin_time_interval[1],
                                                  bin_name)

        for bin_mesh_file_name in config_manager.particleBinMeshFileNames(bin_name):
            bin_reader = vtk.vtkXMLUnstructuredGridReader()
            bin_reader.SetFileName(bin_mesh_file_name)
            bin_reader.Update()
            particle_spacetime_bins.addSpatialRegionToBin(bin_reader.GetOutput(), bin_name)



    # particle_spacetime_bins = particles.ParticleDataTimeBinsSpecifier()    
    # particle_spacetime_bins.addTimeBin(firstSystoleStartTime, firstSystoleEndTime,
    #                               "absolute_helical_flow_index_systole")
    # particle_spacetime_bins.addTimeBin(simulationStartTime, firstSystoleEndTime,
    #                               "absolute_helical_flow_index_diastole")

    # particle_spacetime_bins.addIntervalToExistingTimeBin(firstSystoleEndTime,
    #                                                 cardiacCycleLength,
    #                                                 "absolute_helical_flow_index_diastole")


    # bin_reader = vtk.vtkXMLUnstructuredGridReader()
    # bin_reader.SetFileName('test/data/mesh.vtu')
    # bin_reader.SetFileName('/media/sf_vmShared/diederikParticleTrackingBins/Preoperative_case/Bins/BCT_bin_Lofted_Meshed.sms.vtu')
    # bin_reader.Update()
    # particle_spacetime_bins.addSpatialRegionToBin(bin_reader.GetOutput(), "absolute_helical_flow_index_diastole")

    # bin_reader = vtk.vtkXMLUnstructuredGridReader()
    # bin_reader.SetFileName('test/data/mesh.vtu')
    # bin_reader.SetFileName('/media/sf_vmShared/diederikParticleTrackingBins/Preoperative_case/Bins/BCT_bin_Lofted_Meshed.sms.vtu')
    # bin_reader.Update()
    # particle_spacetime_bins.addSpatialRegionToBin(bin_reader.GetOutput(), "absolute_helical_flow_index_diastole")

    return particle_spacetime_bins


def addScalarParticleDataXdmfAttribute(xdmf_file, attribute_name,
                                       number_of_particles, base_file_name,
                                       starting_timestep, step_index):
    xdmf_file.write('<Attribute Name="{name}" AttributeType="Scalar" Center="Cell">\n'.format(name=attribute_name))
    xdmf_file.write('<DataItem Format="HDF" Dimensions="{num_particles} 1">{base_file_name}-particles-{starting_timestep}.h5:/{name}/{step_index}</DataItem>\n'.format(
                               name=attribute_name,
                               num_particles=number_of_particles,
                               base_file_name=base_file_name,
                               starting_timestep=starting_timestep,
                               step_index=step_index
                               ))
    xdmf_file.write('</Attribute>\n')


def runTracking(fname,
               cycle_start,
               cycle_stop,
               cycle_step,
               step_start,
               ncycles,
               dt,
               disp,
               repartition_frequency_in_simulation_steps,
               simulationStartTime,
               firstSystoleStartTime,
               firstSystoleEndTime,
               cardiacCycleLength,
               particle_vtu_file,
               allwnodes_nbc_file,
               surface_vtp_file,
               custom_config_file_name):

    def setupAvailableParticleDataVectorsMap():
        availableParticleDataVectors = dict()
        availableParticleDataVectors["absolute helicity"] = lambda : local_normalised_helicity_k1 * np.fabs(dt)
        availableParticleDataVectors["plap"] = lambda : np.fabs(dt) * frobenius_k1  #plap

        return availableParticleDataVectors

    config_manager = particles.Configuration(custom_config_file_name)
    
    includeVorticityComputations = config_manager.includeVorticityComputations()

    particle_spacetime_bins = createParticleSpacetimeBins(config_manager, firstSystoleStartTime,
                                               firstSystoleEndTime, simulationStartTime,
                                               cardiacCycleLength)

    availableParticleDataVectors = setupAvailableParticleDataVectorsMap()

    # particle_reinjection_frequency_in_steps = float(sys.argv[14])
                       #repartition_frequency_in_simulation_steps))  # int to round down

    # This is just an example spatial bin, specified as a vtu mesh. It's used to
    # select only particles which lie within the bin on a time-step for data gathering
    # or processing.
    #
    # This example bin is just the whole domain mesh, so it selects all particles.
    # Use smaller meshes and the addSpatialRegionToBin method to use this in real
    # applications.
    # bin_reader = vtk.vtkXMLUnstructuredGridReader()
    # bin_reader.SetFileName('test/data/mesh.vtu')
    # bin_reader.SetFileName('/media/sf_vmShared/diederikParticleTrackingBins/Preoperative_case/Bins/BCT_bin_Lofted_Meshed.sms.vtu')
    # bin_reader.Update()
    # particle_spacetime_bins.addSpatialRegionToBin(bin_reader.GetOutput(), "absolute_helical_flow_index_diastole")

    # centerOfMassFilter = vtk.vtkCenterOfMass()
    # centerOfMassFilter.SetInput(bin_reader.GetOutput())
    # centerOfMassFilter.SetUseScalarsAsWeights(False)
    # centerOfMassFilter.Update()
    # center_of_mass = centerOfMassFilter.GetCenter()
    # print "center of mass:", center_of_mass

    # domain_bin_pts = np.zeros((bin_reader.GetOutput().GetNumberOfPoints(),3))

    # for point_index in range(0,bin_reader.GetOutput().GetNumberOfPoints()):
    #     point = bin_reader.GetOutput().GetPoint(point_index)
    #     for dimension in range(0,3):
    #         domain_bin_pts[point_index, dimension] = center_of_mass[dimension] + 0.5 * (point[dimension] - center_of_mass[dimension])

    # domain_points_vtk = vtk.vtkPoints()
    # domain_points_vtk.SetData(numpy_support.numpy_to_vtk(domain_bin_pts))

    # bin_domain = bin_reader.GetOutput()
    # bin_domain.SetPoints(domain_points_vtk)

    # particle_spacetime_bins.addSpatialRegionToBin(bin_reader.GetOutput(), "absolute_helical_flow_index_diastole")

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    repartition_frequency_in_original_timesteps = repartition_frequency_in_simulation_steps * cycle_step

    # New stepper class. Recommend gradually moving all time-stepping
    # functionality into this for encapsulation and clarity.
    cardiac_cycle_timekeeper = particles.CardiacCycleTimekeeper(cycle_start, cycle_stop, cycle_step,
                                                      ncycles, simulationStartTime, dt,
                                                      repartition_frequency_in_original_timesteps,
                                                      firstSystoleStartTime,
                                                      firstSystoleEndTime,
                                                      cardiacCycleLength,
                                                      particle_spacetime_bins,
                                                      rank,
                                                      config_manager,
                                                      step_start)

    print("total_simulation_steps", cardiac_cycle_timekeeper.getTotalSimulationSteps())

    final_step = step_start + (cycle_stop - cycle_start + cycle_step) * ncycles
    particle_manager = particles.ParticleManager(comm, particle_vtu_file)
    nparticles = particle_manager.getNumberOfParticles()
    # Prepare input and output files (HDF5)
    data_manager = particles.DataManager(comm, fname, step_start, nparticles,
                                config_manager,
                                particleDataTimeBinsSpecifiers=particle_spacetime_bins)
    final_offset = 0
    # Build domain from results file
    print("Setting up MPI...")
    if config_manager.mpiSupportForH5pyAvailable():
        print("MPI h5py found.")
        results = h5py.File("%s.h5" % fname, 'r', driver='mpio', comm=comm)
    else:
        print("MPI h5py not available. Falling back on sequential I/O")
        results = h5py.File("%s.h5" % fname, 'r')
    domain = vtk.vtkUnstructuredGrid()
    domain_pts = results["/Mesh/coordinates"][:]
    npts = domain_pts.shape[0]
    domain_cells = results["/Mesh/topology"][:]
    ncells = domain_cells.shape[0]
    print("cells: %d" % ncells)
    domain_cells_arr = np.zeros((ncells, 5), dtype=np.int64)
    domain_cells_arr[:, 0] = 4
    domain_cells_arr[:, 1:] = domain_cells
    domain_cells_arr = np.reshape(domain_cells_arr, (ncells * 5,))
    domain_pts_vtk = vtk.vtkPoints()
    domain_pts_vtk.SetData(numpy_support.numpy_to_vtk(domain_pts))
    domain.SetPoints(domain_pts_vtk)
    domain_cells_vtk = vtk.vtkCellArray()
    domain_cells_vtk.SetCells(ncells, numpy_support.numpy_to_vtkIdTypeArray(domain_cells_arr))
    domain.SetCells(vtk.VTK_TETRA, domain_cells_vtk)
    domain_locator = vtk.vtkPointLocator()
    domain_locator.SetDataSet(domain)
    domain_locator.BuildLocator()
    velocity_arr = np.zeros((npts, 3))
    velocity_vtk = numpy_support.numpy_to_vtk(velocity_arr)
    velocity_vtk.SetName("velocity")

    if includeVorticityComputations:
        vorticity_array = np.zeros((npts, 3))
        vorticity_vtk = numpy_support.numpy_to_vtk(vorticity_array)
        vorticity_vtk.SetName("Vorticity")

    # helical_flow_index_array = np.zeros((npts,3))
    # helicity_vtk = numpy_support.numpy_to_vtk(helical_flow_index_array)
    # helicity_vtk.SetName("helical_flow_index")

    frobenius_arr = np.zeros((npts, 1))
    frobenius_vtk = numpy_support.numpy_to_vtk(frobenius_arr)
    frobenius_vtk.SetName("frobenius")
    disp_arr = np.zeros((npts, 3))
    disp_vtk = numpy_support.numpy_to_vtk(disp_arr)
    disp_vtk.SetName("displacement")
    domain.GetPointData().AddArray(velocity_vtk)
    domain.GetPointData().AddArray(frobenius_vtk)
    if includeVorticityComputations:
        domain.GetPointData().AddArray(vorticity_vtk)
    # domain.GetPointData().AddArray(helicity_vtk)

    withoutInwardWallVelocityTerm = True

    if not withoutInwardWallVelocityTerm:
        # Get wall normals to compute inward velocity array
        surface_reader = vtk.vtkXMLPolyDataReader()
        surface_reader.SetFileName(surface_vtp_file)
        surface_reader.Update()
        surface_normals = vtk.vtkPolyDataNormals()
        surface_normals.SetInputConnection(surface_reader.GetOutputPort())
        surface_normals.Update()
        surface_normals_locator = vtk.vtkPointLocator()
        surface_normals_locator.SetDataSet(surface_normals.GetOutput())
        surface_normals_locator.BuildLocator()

        allwnodes = open(allwnodes_nbc_file, 'r')
        allwnodes_list = []
        for line in allwnodes:
            allwnodes_list.append(int(line.split()[0]))
        allwnodes_list = list(set(allwnodes_list))  # remove duplicates
        allwnodes_list.sort()
        allwnpts = len(allwnodes_list)
        # map_wall_to_domain is the array containing all the wall node indices.
        map_wall_to_domain = np.array(allwnodes_list, dtype=np.int64)
        map_wall_to_normals = np.zeros((allwnpts,), dtype=np.int64)
        for i in range(allwnpts):
            tmp_pt = domain_pts[map_wall_to_domain[i]]
            # get the array containing in index i the index of the matching
            # (i.e. closest; but distance to the point on the surface)
            # the normal belongs to should be zero) surface normal.
            map_wall_to_normals[i] = surface_normals_locator.FindClosestPoint(tmp_pt)

        wall_normals = numpy_support.vtk_to_numpy(surface_normals.GetOutput().
                                                  GetPointData().
                                                  GetArray("Normals"))[map_wall_to_normals]

        # Generate a small inward velocity at the walls to stop particles becoming stuck there:
        wall_perturbation_velocity_magnitude = -5.0
        wall_velocity = wall_perturbation_velocity_magnitude * wall_normals

    # If simulation was deformable wall get also displacement and deform to current state
    if disp == 1:
        reference_displacement = results["/displacement/%d" % cycle_start][:]
        disp_arr = results["/displacement/%d" % step_start][:]
        domain_pts[map_wall_to_domain] += (disp_arr[map_wall_to_domain] -
                                           reference_displacement[map_wall_to_domain])
        reference_displacement[:] = disp_arr
    else:
        reference_displacement = None

    # Split particles among processes for probefilter input
    particle_manager.repartition()

    map_local_to_global = particle_manager.getParticleLocalToGlobalMap()
    # Create polydata containing local particles coordinates. For Runge Kutta integration we need 4
    local_particles_coordinates = particle_manager.getParticleCoordinatesSlice(map_local_to_global)
    # local_particles_coordinates = particles_coordinates[map_local_to_global]
    nlocalparticles = local_particles_coordinates.shape[0]
    # local_particles_topology = np.zeros((nlocalparticles, 2),
    #                                     dtype=np.int64)
    # local_particles_topology[:, 0] = 1
    # local_particles_topology[:, 1] = np.arange(nlocalparticles,
    #                                            dtype=np.int64)
    # local_particles_topology = np.reshape(local_particles_topology,
    #                                       (nlocalparticles * 2, 1))

    # local_particles = vtk.vtkPolyData()
    # local_particles_pts = vtk.vtkPoints()
    # local_particles_pts.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates))
    local_particles_cells = particle_manager.getLocalParticleCells(nlocalparticles)

    local_particles_coordinates_k1 = np.copy(local_particles_coordinates)
    local_particles_k1 = particle_manager.computeAndGetLocalParticlesVtk(local_particles_coordinates_k1,
                                                                         nlocalparticles)

    local_particles_coordinates_k2 = np.copy(local_particles_coordinates)
    local_particles_k2 = particle_manager.computeAndGetLocalParticlesVtk(local_particles_coordinates_k2,
                                                                         nlocalparticles)

    local_particles_coordinates_k3 = np.copy(local_particles_coordinates)
    local_particles_k3 = particle_manager.computeAndGetLocalParticlesVtk(local_particles_coordinates_k3,
                                                                         nlocalparticles)

    local_particles_coordinates_k4 = np.copy(local_particles_coordinates)
    local_particles_k4 = particle_manager.computeAndGetLocalParticlesVtk(local_particles_coordinates_k4,
                                                                         nlocalparticles)

    # local_particles_coordinates_k1 = np.copy(local_particles_coordinates)
    # local_particles_k1 = vtk.vtkPolyData()
    # local_particles_pts_k1 = vtk.vtkPoints()
    # local_particles_pts_k1.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k1))
    # local_particles_k1.SetPoints(local_particles_pts_k1)
    # local_particles_k1.SetVerts(local_particles_cells)
    #
    # local_particles_coordinates_k2 = np.copy(local_particles_coordinates)
    # local_particles_k2 = vtk.vtkPolyData()
    # local_particles_pts_k2 = vtk.vtkPoints()
    # local_particles_pts_k2.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k2))
    # local_particles_k2.SetPoints(local_particles_pts_k2)
    # local_particles_k2.SetVerts(local_particles_cells)
    #
    # local_particles_coordinates_k3 = np.copy(local_particles_coordinates)
    # local_particles_k3 = vtk.vtkPolyData()
    # local_particles_pts_k3 = vtk.vtkPoints()
    # local_particles_pts_k3.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k3))
    # local_particles_k3.SetPoints(local_particles_pts_k3)
    # local_particles_k3.SetVerts(local_particles_cells)
    #
    # local_particles_coordinates_k4 = np.copy(local_particles_coordinates)
    # local_particles_k4 = vtk.vtkPolyData()
    # local_particles_pts_k4 = vtk.vtkPoints()
    # local_particles_pts_k4.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates_k4))
    # local_particles_k4.SetPoints(local_particles_pts_k4)
    # local_particles_k4.SetVerts(local_particles_cells)

    # See Morbiducci et al. 2009 "In Vivo Quantification of Helical
    # Blood Flow in Human Aorta by Time-Resolved Three-Dimensional
    # Cine Phase Contrast Magnetic Resonance Imaging"
    if includeVorticityComputations:
        helical_flow_index = np.zeros((nlocalparticles,))
        absolute_helical_flow_index = np.zeros((nlocalparticles,))
    data_manager.createTimeBinArrays(nlocalparticles)
    # absolute_helical_flow_index_systole = np.zeros((nlocalparticles,))  # todo remove train
    # absolute_helical_flow_index_diastole = np.zeros((nlocalparticles,))  # todo remove train
    plap = np.zeros((nlocalparticles,))
    prt = np.zeros((nlocalparticles,))
    # t_step = step_start

    saved_steps = []
    saved_nliveparticles = []

    print("Particle setup done.")

    if includeVorticityComputations:
        try:
            if config_manager.mpiSupportForH5pyAvailable():
                vorticityDatafileIn = h5py.File("%s-global_vorticity-%d.h5" % (fname, step_start),'r',driver='mpio',comm=comm)
            else:
                vorticityDatafileIn = h5py.File("%s-global_vorticity-%d.h5" % (fname, step_start),'r')
        except IOError:
            if rank==0:
                print("Computing vorticity for all time frames. This will be saved to disc (*-global_vorticity*.h5), so you can disable it to save time after it has been done once.")
                print("Partitioning the input data...")

            steps_needing_processing = list(range(cycle_start, cycle_stop+cycle_step, cycle_step))
            chunk_size = int(math.ceil(old_div(float(len(steps_needing_processing)), nprocs)))
            this_processors_chunk = steps_needing_processing[rank*chunk_size:(rank+1)*chunk_size]
            if rank ==0:
                print("A total of", len(steps_needing_processing), "time-slices need computing:", steps_needing_processing)
            print("Processor", rank, "will compute the voriticity field for", len(this_processors_chunk), "time slices:", this_processors_chunk)

            # Precompute the vorticity for every time-step and save it to a file.
            # We will re-load each timestep as we need it.
            if config_manager.mpiSupportForH5pyAvailable():
                vorticityDatafile = h5py.File("%s-global_vorticity-%d.h5" % (fname, step_start),'w',driver='mpio',comm=comm)
            else:
                vorticityDatafile = h5py.File("%s-global_vorticity-%d.h5" % (fname, step_start),'w')
            vorticityDatafile.create_group("global_vorticity")
            
            # Data structure modification operations are collective, so we must have all procs do it before proceeding:
            for t_step_precreate_datasets in range(cycle_start, cycle_start + cycle_step * len(steps_needing_processing), cycle_step):
                print("Creating space for vorticity field for step", t_step_precreate_datasets)
                vorticityDatafile["global_vorticity"].create_dataset("%d" % t_step_precreate_datasets, (npts,3), dtype="f")

            for cur_step in tqdm.tqdm(this_processors_chunk, desc="Vorticity field computation for process " + str(rank)):
                startTime = time.time()
                # Get velocity field for current step in cycle:
                velocity_arr[:] = results["/velocity/%d" % cur_step][:]
                gradient_filter = vtk.vtkGradientFilter()
                if vtk.VTK_MAJOR_VERSION <= 5:
                    gradient_filter.SetInput(domain)
                else:
                    gradient_filter.SetInputData(domain)
                gradient_filter.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,"velocity")
                # gradient_filter.SetComputeVorticity(1)
                gradient_filter.ComputeVorticityOn()

                gradient_filter.SetResultArrayName("Vorticity")
                gradient_filter.Update()

                vorticity = gradient_filter.GetOutput().GetPointData().GetArray("Vorticity")
                vorticity_numpy_array = numpy_support.vtk_to_numpy(vorticity)


                vorticityDatafile["/global_vorticity/%d" % cur_step][:] = vorticity_numpy_array

                if rank == 0:
                    tqdm.tqdm.write("Step {} time taken: {}".format(cur_step, time.time() - startTime))

            comm.Barrier()
            vorticityDatafile.close()
            comm.Barrier()

            # Reload the file we just created:
            if config_manager.mpiSupportForH5pyAvailable():
                vorticityDatafileIn = h5py.File("%s-global_vorticity-%d.h5" % (fname, step_start),'r',driver='mpio',comm=comm)
            else:
                vorticityDatafileIn = h5py.File("%s-global_vorticity-%d.h5" % (fname, step_start),'r')



    # t_step        = step_start

    time_start = datetime.datetime.now()
    print("entering cycles")
    consecutiveIntegerStepIndex = 0

    all_cycles_iterator = tqdm.trange(ncycles, desc="All cardiac cycles progress for process " + str(rank))
    for ccc in all_cycles_iterator:
        cur_step = step_start + cycle_step
        if (cycle_step < 0) and  (cur_step < cycle_stop) : cur_step = cycle_start
        if (cycle_step > 0) and  (cur_step > cycle_stop) : cur_step = cycle_start
        for _ in tqdm.trange(cycle_start, cycle_stop+cycle_step, cycle_step, desc="Current cardiac cycle progress for process " + str(rank)):
            consecutiveIntegerStepIndex += 1
            this_partition_start_time = time.time()

            frobenius_path = "/frobenius/%d" % cur_step
            frobenius_arr[:] = results[frobenius_path][:]

            # Get velocity field for current step in cycle:
            velocity_arr[:] = results["/velocity/%d" % cur_step][:]
            if not withoutInwardWallVelocityTerm:
                velocity_arr[map_wall_to_domain] += wall_velocity

            if includeVorticityComputations:
                vorticity_array[:] = vorticityDatafileIn["/global_vorticity/%d" % cur_step][:]

            # vorticity = gradient_filter.GetOutput().GetPointData().GetArray("Vorticity")
            # vorticity_array = numpy_support.vtk_to_numpy(vorticity)

            # If deformable wall, deform wall domain
            if disp == 1:
                disp_arr = results["/displacement/%d" % cur_step][:]
                domain_pts[map_wall_to_domain] += disp_arr[map_wall_to_domain] - reference_displacement[map_wall_to_domain]
                reference_displacement[:] = disp_arr

            # Probe at first location
            alive_k1 = np.array([0, ])  # Empty initialization
            if nlocalparticles > 0:
                probe_k1 = particle_manager.getParticleProbe(domain, local_particles_k1)

                velocity_k1 = particle_manager.getFromProbe("velocity", probe_k1)
                frobenius_k1 = particle_manager.getFromProbe("frobenius", probe_k1)
                alive_k1 = particle_manager.getFromProbe("alive", probe_k1)


                all_cycles_iterator.write("frobenius_k1 {}".format(np.amax(np.absolute(frobenius_k1))))

                plap += np.fabs(dt) * frobenius_k1
                prt += np.fabs(dt)
                
                if includeVorticityComputations:
                    vorticity_k1 = particle_manager.getFromProbe("Vorticity", probe_k1)

                    # These arays are of dimension [number of particles, 3]
                    # print np.shape(vorticity_k1), np.shape(velocity_k1)

                    # See Morbiducci et al. 2009 "In Vivo Quantification of Helical Blood Flow in Human Aorta by Time-Resolved 
                    #                             Three-Dimensional Cine Phase Contrast Magnetic Resonance Imaging"
                    vorticity_dot_velocity_at_current_particle_locations = np.zeros([nlocalparticles,])
                    for index in range(len(vorticity_dot_velocity_at_current_particle_locations)):
                        vorticity_dot_velocity_at_current_particle_locations[index] = np.dot(vorticity_k1[index, :], velocity_k1[index, :])
                    
                    vorticity_norms_at_current_particle_locations = np.linalg.norm(vorticity_k1, axis=1)
                    velocity_norms_at_current_particle_locations = np.linalg.norm(velocity_k1, axis=1)
                    
                    velocity_norm_times_vorticity_norm = vorticity_norms_at_current_particle_locations * velocity_norms_at_current_particle_locations
                    
                    # Contrive the local_normalised_helicity_k1 to be zero if either the velocity or the vorticity is zero at a point.
                    # This makes sense as if there's no velocity or vorticity, then there is no helicity either.
                    boolean_mask_giving_zero_locations = [True if arrayEntry==0.0 else False for arrayEntry in velocity_norm_times_vorticity_norm]
                    velocity_norm_times_vorticity_norm = np.where(boolean_mask_giving_zero_locations, 1.0, velocity_norm_times_vorticity_norm)
                    vorticity_dot_velocity_at_current_particle_locations = np.where(boolean_mask_giving_zero_locations, 0.0, vorticity_dot_velocity_at_current_particle_locations)

                    # Actually compute the local normalised helicity:
                    local_normalised_helicity_k1 = old_div(vorticity_dot_velocity_at_current_particle_locations, velocity_norm_times_vorticity_norm)

                    # Gather the integral of the helical flow index. This will be divided by the total time
                    # to get a time-average just before we write its value to file.
                    helical_flow_index += local_normalised_helicity_k1 * np.fabs(dt)
                    absolute_helical_flow_index += np.abs(local_normalised_helicity_k1) * np.fabs(dt)

                for spacetime_bin_name in config_manager.particleBinNames():
                    if cardiac_cycle_timekeeper.currentTimeLiesInBin(spacetime_bin_name):
                        if rank == 0:
                            all_cycles_iterator.write("====> Current time lies in time dimension of bin {}.".format(spacetime_bin_name))
                        binDatatype = config_manager.getDataNameForBin(spacetime_bin_name)
                        vectorToAdd = availableParticleDataVectors[binDatatype]()
                        data_manager.vectorAddToTimeBinWithinSpatialLimits(spacetime_bin_name, vectorToAdd,
                                                                           dt,
                                                                           local_particle_coordinates=local_particles_coordinates_k1)


                # # Also gather independent info on systole and diastole:
                # if cardiac_cycle_timekeeper.currentTimeLiesInBin("absolute_helical_flow_index_systole"):
                #     if rank == 0:
                #         print "====> SYSTOLE"
                #     data_manager.vectorAddToTimeBinWithinSpatialLimits("absolute_helical_flow_index_systole",
                #                                     (np.abs(local_normalised_helicity_k1) *
                #                                      np.fabs(dt)))
                #     systoleStepIndex += 1
                # elif cardiac_cycle_timekeeper.currentTimeLiesInBin("absolute_helical_flow_index_diastole"):
                #     if rank == 0:
                #         print "====> DIASTOLE"
                #     data_manager.vectorAddToTimeBinWithinSpatialLimits("absolute_helical_flow_index_diastole",
                #                                     (np.abs(local_normalised_helicity_k1) *
                #                                      np.fabs(dt)), local_particles_coordinates)
                #     diastoleStepIndex += 1
                # else:
                #     if rank == 0:
                #         warnings.warn("Current time point falls in none of the time bins. " + str(cardiac_cycle_timekeeper.getTime()))

            # Probe at second location
                local_particles_coordinates_k2 += 0.5 * dt * velocity_k1

                # probe_k2 = vtk.vtkProbeFilter()
                # probe_k2.SetValidPointMaskArrayName("alive")
                # if vtk.VTK_MAJOR_VERSION <= 5:
                #     probe_k2.SetInput(local_particles_k2)
                #     probe_k2.SetSource(domain)
                # else:
                #     probe_k2.SetInputData(local_particles_k2)
                #     probe_k2.SetSourceData(domain)
                # probe_k2.Update()

                probe_k2 = particle_manager.getParticleProbe(domain, local_particles_k2)
                velocity_k2 = particle_manager.getFromProbe("velocity", probe_k2)

            # Probe at third location
                local_particles_coordinates_k3 += 0.5*dt*velocity_k2
            
                probe_k3 = particle_manager.getParticleProbe(domain, local_particles_k3)
                velocity_k3 = particle_manager.getFromProbe("velocity", probe_k3)

            # Probe at fourth location
                local_particles_coordinates_k4 += dt*velocity_k3            
                probe_k4 = particle_manager.getParticleProbe(domain, local_particles_k4)
                velocity_k4 = particle_manager.getFromProbe("velocity", probe_k4)

            # Update position
                local_particles_coordinates_k1 += dt/6.0*(velocity_k1+2.0*velocity_k2+2.0*velocity_k3+velocity_k4)        
                local_particles_coordinates_k2[:] = local_particles_coordinates_k1
                local_particles_coordinates_k3[:] = local_particles_coordinates_k1
                local_particles_coordinates_k4[:] = local_particles_coordinates_k1

                # Probably this can be removed altogether...
                if vtk.VTK_MAJOR_VERSION <= 6:
                    local_particles_k1.Update()
                    local_particles_k2.Update()
                    local_particles_k3.Update()
                    local_particles_k4.Update()

            # If necessary save data and repartition
            if cardiac_cycle_timekeeper.repartitionThisStep():
                time_stop = datetime.datetime.now()
                time_need = (time_stop-time_start).seconds
                time_start = time_stop
                saved_steps.append(cardiac_cycle_timekeeper.getTStep())

                nliveparticles = particle_manager.getNumberOfGlobalLiveParticles()
                saved_nliveparticles.append(nliveparticles)
                data_manager.createSpaceForThisStepsParticles(cardiac_cycle_timekeeper.getTStep(), nliveparticles)
                
                if nlocalparticles > 0 :
                    local_particle_data_slice = particle_manager.getParticleOffsetSlice(rank)

                    data_manager.writeLocalParticleTimestepData("/Index/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, map_local_to_global)
                    data_manager.writeLocalParticleTimestepData("/Topology/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, np.arange(particle_manager.getParticleOffsetStart(rank), particle_manager.getParticleOffsetEnd(rank), dtype=np.int64))
                    data_manager.writeLocalParticleTimestepData("/Coordinates/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, local_particles_coordinates_k1)
                    data_manager.writeLocalParticleTimestepData("/PLAP/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, plap)
                    data_manager.writeLocalParticleTimestepData("/PRT/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, prt)
                    data_manager.writeLocalParticleTimestepData("/alive/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, alive_k1)
                    data_manager.writeLocalParticleTimestepData("/velocity/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, velocity_k1)
                    if includeVorticityComputations:
                        data_manager.writeLocalParticleTimestepData("/vorticity/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, vorticity_k1)
                        data_manager.writeLocalParticleTimestepData("/helical_flow_index/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, helical_flow_index)
                        data_manager.writeLocalParticleTimestepData("/absolute_helical_flow_index/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, absolute_helical_flow_index)
                    

                    data_manager.writeInternallyManagedLocalParticleTimestepData(
                        cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice
                    )

                    data_manager.writeLocalParticleTimestepData("/partition/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, rank)

                    dead_particle_locations = np.where(alive_k1 == 0)
                    indices_dead = map_local_to_global[dead_particle_locations]
                    nlocaldead = indices_dead.shape[0]
                    just_exited_particles_data_placement_slice = slice(final_offset, final_offset + nlocaldead)

                    data_manager.writeLocalJustExitedParticleData("/Coordinates/Final/", just_exited_particles_data_placement_slice, local_particles_coordinates_k1[dead_particle_locations])
                    data_manager.writeLocalJustExitedParticleData("/PLAP/Final/", just_exited_particles_data_placement_slice, plap[dead_particle_locations])
                    data_manager.writeLocalJustExitedParticleData("/PRT/Final/", just_exited_particles_data_placement_slice, prt[dead_particle_locations])
                    data_manager.writeLocalJustExitedParticleData("/Index/Final/", just_exited_particles_data_placement_slice, indices_dead)
                    data_manager.writeLocalJustExitedParticleData("/TimeWritten/Final/", just_exited_particles_data_placement_slice, cardiac_cycle_timekeeper.getTStep())

                    # Modify the variables before saving that need to be normalised by residence time:
                    if includeVorticityComputations:
                        normalised_helical_flow_index = old_div(helical_flow_index[dead_particle_locations], prt[dead_particle_locations])
                        data_manager.writeLocalJustExitedParticleData("helical_flow_index/Final/", just_exited_particles_data_placement_slice, normalised_helical_flow_index)

                        normalised_absolute_helical_flow_index = old_div(absolute_helical_flow_index[dead_particle_locations], prt[dead_particle_locations])
                        data_manager.writeLocalJustExitedParticleData("absolute_helical_flow_index/Final/", just_exited_particles_data_placement_slice, normalised_absolute_helical_flow_index)

                    data_manager.writeInternallyManagedLocalJustExitedParticleData(just_exited_particles_data_placement_slice,
                                                                                   dead_particle_locations)

                    final_offset += nlocaldead

                time_save_stop = datetime.datetime.now()
                time_save = (time_save_stop-time_start).seconds
                time_start = time_save_stop
                elapsed_time = time.time() - this_partition_start_time
                all_cycles_iterator.write("Cycle: %d Rank: %d Current timestep: %d, nparticles: %d, prt_active: %d, nliveparticles: %d, local_coords:%d, time_interp: %d, time_save:%d, elapsed_time:%d" % (ccc,rank,cur_step,nlocalparticles,np.sum(alive_k1),particle_manager.getNumberOfGlobalLiveParticles(),local_particles_coordinates_k1.shape[0],time_need,time_save, elapsed_time))
                # Gather all updated particles and repartition            
                comm.Barrier()

                if includeVorticityComputations:
                    helical_flow_index_global = data_manager.getOutputFileArrayAliveParticlesOnly("/helical_flow_index/", cardiac_cycle_timekeeper.getTStep())
                    absolute_helical_flow_index_global = data_manager.getOutputFileArrayAliveParticlesOnly("/absolute_helical_flow_index/", cardiac_cycle_timekeeper.getTStep())

                plap_global    = data_manager.getOutputFileArrayAliveParticlesOnly("/PLAP/", cardiac_cycle_timekeeper.getTStep())
                prt_global     = data_manager.getOutputFileArrayAliveParticlesOnly("/PRT/", cardiac_cycle_timekeeper.getTStep())
                indices_global = data_manager.getOutputFileArrayAliveParticlesOnly("/Index/", cardiac_cycle_timekeeper.getTStep())

                particle_manager.setParticleIndices(indices_global)

                coordinates_global = data_manager.getOutputFileArrayAliveParticlesOnly("/Coordinates/", cardiac_cycle_timekeeper.getTStep())

                particle_manager.setParticleCoordinates(np.copy(coordinates_global))

                particle_manager.repartition()
                # Broadcast new particles_partition and rebuild maps and local dataset

                map_local_to_global = particle_manager.\
                    getParticleLocalToGlobalMap()
                nlocalparticles = map_local_to_global.shape[0]
                if nlocalparticles > 0:
                    # Get PLAP and PRT of local particles
                    plap = particle_manager.getLocalArrayValues(plap_global)
                    prt = particle_manager.getLocalArrayValues(prt_global)
                    if includeVorticityComputations:
                        helical_flow_index = particle_manager.\
                            getLocalArrayValues(helical_flow_index_global)
                        absolute_helical_flow_index = particle_manager.\
                            getLocalArrayValues(absolute_helical_flow_index_global)

                    local_particle_indices = particle_manager.getLocalParticleIndices()
                    data_manager.relocaliseDataBinArrays(local_particle_indices, cardiac_cycle_timekeeper.getTStep())

                    #fixme (the above with the global_arrays dict patches this for now)
                    ## absolute_helical_flow_index_systole = particle_manager.\
                    ##     getLocalArrayValues(
                    ##         absolute_helical_flow_index_systole_global
                    ##     )
                    ## absolute_helical_flow_index_diastole = particle_manager.\
                    ##     getLocalArrayValues(
                    ##         absolute_helical_flow_index_diastole_global
                    ##     )

                    local_particles_coordinates = particle_manager.\
                        getLocalParticleCoordinates()

                    # Get repartitioned alive and velocity arrays
                    alive_global_repartitioned = \
                        np.array(
                            data_manager.getOutputFileArray("/alive/", cardiac_cycle_timekeeper.getTStep()),
                            dtype=np.bool)
                    particle_manager.setLocalAliveParticleBooleanMask(
                        alive_global_repartitioned)
                    # particle_manager.updateGlobalAliveParticleBooleanMask(domain)
                    
                    # alive_repartitioned = particle_manager.getLocalAliveParticles()

                    velocity_unmasked = data_manager.getOutputFileArray("/velocity/", cardiac_cycle_timekeeper.getTStep())
                    velocity_repartitioned = particle_manager.maskArrayByParticleLocalAlivenessAndGetLocalValues(velocity_unmasked)

                    if includeVorticityComputations:
                        vorticity_unmasked = data_manager.getOutputFileArray("/vorticity/", cardiac_cycle_timekeeper.getTStep())
                        vorticity_repartitioned = particle_manager.maskArrayByParticleLocalAlivenessAndGetLocalValues(vorticity_unmasked)

                        helical_flow_index_unmasked = data_manager.getOutputFileArray("/helical_flow_index/", cardiac_cycle_timekeeper.getTStep())
                        helical_flow_index_repartitioned = particle_manager.maskArrayByParticleLocalAlivenessAndGetLocalValues(helical_flow_index_unmasked)

                        absolute_helical_flow_index_unmasked = data_manager.getOutputFileArray("/absolute_helical_flow_index/", cardiac_cycle_timekeeper.getTStep())
                        absolute_helical_flow_index_repartitioned = particle_manager.maskArrayByParticleLocalAlivenessAndGetLocalValues(absolute_helical_flow_index_unmasked)

                    # Doing the repartition here is probably what causes a crash on the
                    # final time-step sometimes, as the partition may change before the
                    # post-cycles write-out. \todo FIX THIS!
                    data_manager.maskAndRepartitionDataArrays(cardiac_cycle_timekeeper.getTStep())

                if cardiac_cycle_timekeeper.reinjectThisStep():
                    all_cycles_iterator.write("Reinjecting particles...")
                    particle_manager.reinjectParticles()
                    map_local_to_global = particle_manager.getParticleLocalToGlobalMap()

                    data_manager.expandFinalTimestepStorageArrays(particle_manager.getNumberOfParticles())

                    # previous_number_of_local_particles = len(local_particles_coordinates)
                    # local_particles_coordinates = particle_manager.getLocalParticleCoordinates()

                    # number_of_new_local_particles = len(local_particles_coordinates) - previous_number_of_local_particles
                    # new_particles_array_extension = np.zeros((number_of_new_local_particles,))

                    local_particles_coordinates = particle_manager.getLocalParticleCoordinates()
                    
                    
                    new_particles_global_data_array_extension = np.zeros((particle_manager.getNumberOfNewlyInjectedParticles(),))
                    
                    # todo move this into extendAndRelocaliseDataBinArrays()
                    plap_global = np.append(plap_global, new_particles_global_data_array_extension)
                    prt_global = np.append(prt_global, new_particles_global_data_array_extension)

                    # alive_global_repartitioned_extension = \
                    #     np.ones(
                    #         (particle_manager.getNumberOfNewlyInjectedParticles(),),
                    #         dtype=np.bool)
                    #
                    # alive_global_repartitioned = np.append(alive_global_repartitioned, alive_global_repartitioned_extension)
                    # particle_manager.setLocalAliveParticleBooleanMask(
                    #     alive_global_repartitioned)

                    if includeVorticityComputations:
                        helical_flow_index_global = np.append(helical_flow_index_global, new_particles_global_data_array_extension)
                        absolute_helical_flow_index_global = np.append(absolute_helical_flow_index_global, new_particles_global_data_array_extension)

                    # Get PLAP and PRT of local particles
                    plap = particle_manager.getLocalArrayValues(plap_global)
                    prt  = particle_manager.getLocalArrayValues(prt_global)

                    if includeVorticityComputations:
                        helical_flow_index = particle_manager.getLocalArrayValues(helical_flow_index_global)
                        absolute_helical_flow_index = particle_manager.getLocalArrayValues(absolute_helical_flow_index_global)
                    
                    local_particle_indices = particle_manager.getLocalParticleIndices()
                    numberOfNewParticles = particle_manager.getNumberOfNewlyInjectedParticles()
                    data_manager.extendAndRelocaliseDataBinArrays(local_particle_indices,
                                                                  cardiac_cycle_timekeeper.getTStep(),
                                                                  numberOfNewParticles)

                    #fixme (the above with the to_append_to_global_arrays dict patches this for now)
                    ## absolute_helical_flow_index_systole = particle_manager.getLocalArrayValues(absolute_helical_flow_index_systole_global)
                    ## absolute_helical_flow_index_diastole = particle_manager.getLocalArrayValues(absolute_helical_flow_index_diastole_global)


                map_local_to_global = particle_manager.getParticleLocalToGlobalMap()
                nlocalparticles = map_local_to_global.shape[0]
                if nlocalparticles > 0:

                    local_particles_coordinates_k1 = np.copy(local_particles_coordinates)
                    local_particles_k1 = particle_manager.computeAndGetLocalParticlesVtk(local_particles_coordinates_k1,
                                                                                         nlocalparticles)

                    probe_k1 = particle_manager.getParticleProbe(domain, local_particles_k1)
                    velocity_repartitioned = particle_manager.getFromProbe("velocity", probe_k1)
                    frobenius_k1 = particle_manager.getFromProbe("frobenius", probe_k1)
                    alive_repartitioned = particle_manager.getFromProbe("alive", probe_k1)
                    # if you want the other arrays here for the helicity computations - you probably will need
                    # to sort out the alive particle boolean mask in particle_manager, as they don't come from a probe
                    # (vorticity, helical_flow_index, absolute_helical_flow_index)

                    local_particles_coordinates_k2 = np.copy(local_particles_coordinates)
                    local_particles_k2 = particle_manager.computeAndGetLocalParticlesVtk(local_particles_coordinates_k2,
                                                                                         nlocalparticles)
                
                    local_particles_coordinates_k3 = np.copy(local_particles_coordinates)
                    local_particles_k3 = particle_manager.computeAndGetLocalParticlesVtk(local_particles_coordinates_k3,
                                                                                         nlocalparticles)

                    local_particles_coordinates_k4 = np.copy(local_particles_coordinates)
                    local_particles_k4 = particle_manager.computeAndGetLocalParticlesVtk(local_particles_coordinates_k4,
                                                                                         nlocalparticles)


            # Update cur_step and tstep
            cur_step += cycle_step
            if (cycle_step > 0) and cur_step > cycle_stop:
                cur_step = cycle_start
            if (cycle_step < 0) and cur_step < cycle_stop:
                cur_step = cycle_start

            # t_step += cycle_step
            # New stepper class. Recommend gradually moving all time-stepping functionality into this for encapsulation and clarity.
            cardiac_cycle_timekeeper.step()
            all_cycles_iterator.refresh()

    # Save results at the end of cycle through cycles
    print("Cycle: %d Rank: %d Current timestep: %d, nparticles: %d, particles_active: %d, local_coordinates:%d" % (ccc,rank,cur_step,nlocalparticles,np.sum(alive_k1),local_particles_coordinates_k1.shape[0]))
    saved_steps.append(cardiac_cycle_timekeeper.getTStep())
    nliveparticles = particle_manager.getNumberOfGlobalLiveParticles()
    saved_nliveparticles.append(nliveparticles)

    data_manager.createSpaceForThisStepsParticles(cardiac_cycle_timekeeper.getTStep(), nliveparticles)

    if nlocalparticles > 0 :
        local_particle_data_slice = particle_manager.getParticleOffsetSlice(rank)
        data_manager.writeLocalParticleTimestepData("/Index/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, map_local_to_global)
        data_manager.writeLocalParticleTimestepData("/Topology/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, np.arange(particle_manager.getParticleOffsetStart(rank), particle_manager.getParticleOffsetEnd(rank), dtype=np.int64))
        data_manager.writeLocalParticleTimestepData("/Coordinates/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, local_particles_coordinates_k1)
        data_manager.writeLocalParticleTimestepData("/PLAP/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, plap)
        data_manager.writeLocalParticleTimestepData("/PRT/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, prt)

        # Use repartitioned arrays
        # alive_repartitioned = particle_manager.getLocalAliveParticles()
        data_manager.writeLocalParticleTimestepData("/alive/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, alive_repartitioned)
        data_manager.writeLocalParticleTimestepData("/velocity/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, velocity_repartitioned)

        if includeVorticityComputations:
            data_manager.writeLocalParticleTimestepData("/vorticity/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, vorticity_repartitioned)
            data_manager.writeLocalParticleTimestepData("/helical_flow_index/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, helical_flow_index_repartitioned)
            data_manager.writeLocalParticleTimestepData("/absolute_helical_flow_index/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, absolute_helical_flow_index_repartitioned)

        data_manager.writeInternallyManagedLocalParticleTimestepData(
                        cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice
                    )
        data_manager.writeLocalParticleTimestepData("/partition/", cardiac_cycle_timekeeper.getTStep(), local_particle_data_slice, rank)
        
        simulation_end_particles_data_placement_slice = slice(final_offset, final_offset + nlocalparticles)

        data_manager.writeLocalJustExitedParticleData("/Coordinates/Final/", simulation_end_particles_data_placement_slice, local_particles_coordinates_k1)
        data_manager.writeLocalJustExitedParticleData("/PLAP/Final/", simulation_end_particles_data_placement_slice, plap)
        data_manager.writeLocalJustExitedParticleData("/PRT/Final/", simulation_end_particles_data_placement_slice, prt)
        data_manager.writeLocalJustExitedParticleData("/Index/Final/", simulation_end_particles_data_placement_slice, map_local_to_global)
        data_manager.writeLocalJustExitedParticleData("/TimeWritten/Final/", simulation_end_particles_data_placement_slice, cardiac_cycle_timekeeper.getTStep())

        dead_particle_locations = np.where(alive_repartitioned==0)
        indices_dead = map_local_to_global[dead_particle_locations]

        if includeVorticityComputations:
            # Modify the variables before saving that need to be normalised by residence time:
            normalised_helical_flow_index = old_div(helical_flow_index_repartitioned, prt)
            data_manager.writeLocalJustExitedParticleData("helical_flow_index/Final/", simulation_end_particles_data_placement_slice, normalised_helical_flow_index)

            normalised_absolute_helical_flow_index = old_div(absolute_helical_flow_index_repartitioned, prt)
            data_manager.writeLocalJustExitedParticleData("absolute_helical_flow_index/Final/", simulation_end_particles_data_placement_slice, normalised_absolute_helical_flow_index)

        data_manager.writeInternallyManagedFinalTimestepParticleData(
                                simulation_end_particles_data_placement_slice
                            )
            
    # Write VTK unstructured Grid with results at the end of the advection
    comm.Barrier()
    if rank == 0:
        nparticles = particle_manager.getNumberOfParticles()
        print("Final Step: %d, Time Step: %d" % (final_step, cardiac_cycle_timekeeper.getTStep()))
        print("Reducing Final Step Results...")
        data_manager.reduceIndexArrayAndSaveInH5File()

        sorted_coordinates = data_manager.reduceArrayAndSaveInH5File("/Coordinates/Final/", True)
        sorted_plap = data_manager.reduceArrayAndSaveInH5File("/PLAP/Final/", False)
        sorted_prt = data_manager.reduceArrayAndSaveInH5File("/PRT/Final/", False)

        if includeVorticityComputations:
            sorted_helical_flow_index = data_manager.reduceArrayAndSaveInH5File("/helical_flow_index/Final/", False)
            sorted_absolute_helical_flow_index = data_manager.reduceArrayAndSaveInH5File("/absolute_helical_flow_index/Final/", False)

        sorted_internal_arrays_generator = data_manager.reduceInternallyManagedArraysAndSaveInH5File()

        coordinates_vtu_vtk = numpy_support.numpy_to_vtk(sorted_coordinates)
        coordinates_vtu_vtk.SetName("Coordinates")
        plap_vtu_vtk = numpy_support.numpy_to_vtk(sorted_plap)
        plap_vtu_vtk.SetName("PLAP")
        prt_vtu_vtk = numpy_support.numpy_to_vtk(sorted_prt)
        prt_vtu_vtk.SetName("PRT")

        if includeVorticityComputations:
            helical_flow_index_vtu_vtk = numpy_support.numpy_to_vtk(sorted_helical_flow_index)
            helical_flow_index_vtu_vtk.SetName("Helical Flow Index")

            absolute_helical_flow_index_vtu_vtk = numpy_support.numpy_to_vtk(sorted_absolute_helical_flow_index)
            absolute_helical_flow_index_vtu_vtk.SetName("Absolute Helical Flow Index")

        particle_manager.addParticlePointDataArray(coordinates_vtu_vtk)
        particle_manager.addParticlePointDataArray(plap_vtu_vtk)
        particle_manager.addParticlePointDataArray(prt_vtu_vtk)

        if includeVorticityComputations:
            particle_manager.addParticlePointDataArray(helical_flow_index_vtu_vtk)
            particle_manager.addParticlePointDataArray(absolute_helical_flow_index_vtu_vtk)
        
        # IIRC this thing1 thing2 nonsense is to avoid triggering garbage
        # collection on these arrays - vtk python bindings dont seem to handle the reference
        # counters correctly at the time of writing
        thing1Refs = []
        thing2Refs = []
        for internally_managed_particle_data_array, thing1, thing2 in sorted_internal_arrays_generator:
            thing1Refs.append(thing1)
            thing2Refs.append(thing2)
            particle_manager.addParticlePointDataArray(internally_managed_particle_data_array)

        # Compute Right Cauchy Green tensor for FTLE
        deformation_gradient = vtk.vtkGradientFilter()
        deformation_gradient.SetInputConnection(particle_manager.getParticleDataOutputPort())
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
        particle_manager.addParticlePointDataArray(ftle_vtu_vtk)    
        particles_writer = vtk.vtkXMLUnstructuredGridWriter()
        particles_writer.SetInputConnection(particle_manager.getParticleDataOutputPort())
        particles_writer.SetFileName("%s-particles-%d.vtu" % (fname, step_start))
        particles_writer.Update()
        print("...Done!")

    # Write XDMF File for Visualization in Paraview
    if rank == 0:
        xdmf_out = open("%s-particles-%d.xdmf" % (fname, step_start), 'w')
        xdmf_out.write("""<?xml version="1.0"?>
    <Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
      <Domain>
        <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
          <Time TimeType="List">\n""")
        timesteps_str = ' '.join(str(i) for i in saved_steps)
        nsteps = len(saved_steps)
        xdmf_out.write('<DataItem Format="XML" Dimensions="%d">%s</DataItem>\n</Time>' %(nsteps,timesteps_str) )
        # For each timestep point to grid topology and geometry, and attributes
        nliveparticles = particle_manager.getNumberOfGlobalLiveParticles() # todo confirm that this line is redundant & remove it
        for i, nliveparticles in zip(saved_steps, saved_nliveparticles):
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

            if includeVorticityComputations:
                xdmf_out.write('<Attribute Name="vorticity" AttributeType="Vector" Center="Cell">\n')
                xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 3">%s-particles-%d.h5:/vorticity/%d</DataItem>\n'
                                           % (nliveparticles,fname,step_start,i))
                xdmf_out.write('</Attribute>\n')
                xdmf_out.write('<Attribute Name="helical_flow_index" AttributeType="Scalar" Center="Cell">\n')
                xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s-particles-%d.h5:/helical_flow_index/%d</DataItem>\n'
                                           % (nliveparticles,fname,step_start,i))
                xdmf_out.write('</Attribute>\n')
                xdmf_out.write('<Attribute Name="absolute_helical_flow_index" AttributeType="Scalar" Center="Cell">\n')
                xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s-particles-%d.h5:/absolute_helical_flow_index/%d</DataItem>\n'
                                           % (nliveparticles,fname,step_start,i))
                xdmf_out.write('</Attribute>\n')

            for bin_name in config_manager.particleBinNames():
                addScalarParticleDataXdmfAttribute(xdmf_out, bin_name, nliveparticles, fname, step_start, i)
                addScalarParticleDataXdmfAttribute(xdmf_out, bin_name + particles.data_manager.PARTICLE_RESIDENCE_TIME_NAMETAG, nliveparticles, fname, step_start, i)

            xdmf_out.write('<Attribute Name="Invariant Particle Index" AttributeType="Integer" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s-particles-%d.h5:/Index/%d</DataItem>\n'
                                       % (nliveparticles,fname,step_start,i))
            xdmf_out.write('</Attribute>\n')

            xdmf_out.write('</Grid>\n')
        xdmf_out.write('</Grid>\n</Domain>\n</Xdmf>')
        xdmf_out.close()

    results.close()
    data_manager.close()


def runTrackingWithConfiguration(config_manager):
    command_line_input = parseCommandLineArguments()
    runTracking(config_manager.fname(),
               config_manager.cycle_start(),
               config_manager.cycle_stop(),
               config_manager.cycle_step(),
               config_manager.step_start(),
               config_manager.ncycles(),
               config_manager.dt(),
               config_manager.disp(),
               config_manager.repartition_frequency_in_simulation_steps(),
               config_manager.simulationStartTime(),
               config_manager.firstSystoleStartTime(),
               config_manager.firstSystoleEndTime(),
               config_manager.cardiacCycleLength(),
               command_line_input.particle_vtu_file,
               command_line_input.allwnodes_nbc_file,
               command_line_input.surface_vtp_file,
               command_line_input.custom_config_file_name)


if __name__ == "__main__":
    command_line_input = parseCommandLineArguments()
    print("Attempting to use local config file ", command_line_input.custom_config_file_name)
    config_manager = particles.Configuration(str(command_line_input.custom_config_file_name))
    runTrackingWithConfiguration(config_manager)

    
