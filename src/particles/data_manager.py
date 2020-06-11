from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
import numpy as np
import h5py
from .particle_data_time_bin_specifier import *
import vtk
from vtk.util import numpy_support
import particles.configuration
import tqdm

UNWRITTEN_VALUE = -1
PARTICLE_RESIDENCE_TIME_NAMETAG = "_particle_residence_time"

class DataManager(object):
    ARRAY_ONE_DIMENSIONAL = "arrayOneDimensional"
    ARRAY_TWO_DIMENSIONAL = "arrayTwoDimensional"
    ARRAY_THREE_DIMENSIONAL = "arrayThreeDimensional"
    ARRAY_FOUR_DIMENSIONAL = "arrayFourDimensional"

    def __init__(self, comm_in, fname, step_start, numberOfParticles,
                 config_manager,
                 particleDataTimeBinsSpecifiers=None):
        print("Creating data manager...")
        self.config_manager = config_manager
        # use finalStepArrayHandles for ease of access when we want to
        # resize the arrays
        self.finalStepArrayHandles = []

        self.comm = comm_in
        self.numberOfProcessors = self.comm.Get_size()
        self.numberOfParticles = numberOfParticles
        self.setupHelicityTimeBins(particleDataTimeBinsSpecifiers)
        self.rank = self.comm.Get_rank()
        # todo wrap this file in the DataManager, too.
        # self.results = h5py.File("%s.h5" % fname,'r',driver='mpio',
        # comm=self.comm)

        if config_manager.mpiSupportForH5pyAvailable():
          self.output = h5py.File("%s-particles-%d.h5" % (fname, step_start),
                                  'w', driver='mpio', comm=self.comm)
        else:
          self.output = h5py.File("%s-particles-%d.h5" % (fname, step_start),
                                  'w')
        self.createBasicDataGroups()
        self.createOutputFieldsForParticleExitTimeData()
        print("Data manager init complete")

    def createTimeBinArrays(self, numberOfLocalParticles):
        self.spacetime_bin_data_arrays = dict()
        for array_name in self.particleDataTimeBinsSpecifiers.names():
            self.spacetime_bin_data_arrays[array_name] = \
                np.zeros((numberOfLocalParticles,))
            
            # Create bin residence time counter for each metric, too:
            residence_time_array_name = array_name + PARTICLE_RESIDENCE_TIME_NAMETAG
            self.spacetime_bin_data_arrays[residence_time_array_name] = \
                np.zeros((numberOfLocalParticles,))

    # Adds data to an exsiting space-time bin, summing the components
    # of the existing vector with the vector_to_add. They
    # must have the same length.
    #
    # Points which do not lie in the spatial extent of the bin will
    # be masked out by this function and not added to the output data.
    def vectorAddToTimeBinWithinSpatialLimits(self,
                           bin_name,
                           vector_to_add,
                           time_step,
                           local_particle_coordinates=None):
      if not self.particleDataTimeBinsSpecifiers.binHasSpatialLimits(bin_name):
        self.spacetime_bin_data_arrays[bin_name] += vector_to_add
        
        # Increase the residence time for particles currently in this bin:
        residence_time_array_name = bin_name + PARTICLE_RESIDENCE_TIME_NAMETAG
        self.spacetime_bin_data_arrays[residence_time_array_name] += time_step
      else:
        if local_particle_coordinates is None:
          raise RuntimeError("vectorAddToTimeBinWithinSpatialLimits was not passed the local_particle_coordinates.")

        particles_in_spatial_region = self.__getAliveParticlesInSpatialBin(bin_name, local_particle_coordinates)

        if len(particles_in_spatial_region) > 0:
          self.spacetime_bin_data_arrays[bin_name][particles_in_spatial_region] += \
              vector_to_add[particles_in_spatial_region]

          # Increase the residence time for particles currently in this bin:
          residence_time_array_name = bin_name + PARTICLE_RESIDENCE_TIME_NAMETAG
          tqdm.tqdm.write("pre time: {}".format(self.spacetime_bin_data_arrays[residence_time_array_name][particles_in_spatial_region][0]))
          self.spacetime_bin_data_arrays[residence_time_array_name][particles_in_spatial_region] += time_step
          tqdm.tqdm.write("post time: {}".format(self.spacetime_bin_data_arrays[residence_time_array_name][particles_in_spatial_region][0]))
        else:
          if self.rank is 0:
            tqdm.tqdm.write("No particles found in spatial bin {} (only checked Rank 0).".format(bin_name))

    def __getAliveParticlesInSpatialBin(self, bin_name, local_particle_coordinates):
      probe = vtk.vtkProbeFilter()
      probe.SetValidPointMaskArrayName("lies_within_spatial_bin")

      # todo cache this if there are performance issues
      # (or evaluate alternative methods of working with the vtkProbeFilter)
      nlocalparticles = local_particle_coordinates.shape[0]
      local_particles_topology = np.zeros((nlocalparticles, 2),
                                          dtype=np.int64)
      local_particles_topology[:, 0] = 1
      local_particles_topology[:, 1] = np.arange(nlocalparticles,
                                                 dtype=np.int64)
      local_particles_topology = np.reshape(local_particles_topology,
                                            (nlocalparticles * 2, 1))

      local_particles = vtk.vtkPolyData()
      local_particles_pts = vtk.vtkPoints()
      local_particles_pts.SetData(numpy_support.numpy_to_vtk(local_particle_coordinates))
      local_particles_cells = vtk.vtkCellArray()
      local_particles_cells.SetCells(nlocalparticles,
                                     numpy_support.numpy_to_vtkIdTypeArray(local_particles_topology))
      local_particles.SetPoints(local_particles_pts)
      local_particles.SetVerts(local_particles_cells)

      if vtk.VTK_MAJOR_VERSION >= 6:
        probe.SetInputData(local_particles)
      else:
        probe.SetInput(local_particles)

      # Gather together the multiple spatial regions which comprise the spatial extent of
      # this bin.
      #
      # todo cache this if it is found to impact performance.
      append_filter = vtk.vtkAppendFilter()
      for spatialSubregionBinIndex in range(
        self.particleDataTimeBinsSpecifiers.getNumberOfSpatialRegionsComprisingBin(bin_name)
        ):

        if vtk.VTK_MAJOR_VERSION >= 6:
          append_filter.AddInputData(
          self.particleDataTimeBinsSpecifiers.getBinSpatialLimits(bin_name)[spatialSubregionBinIndex]
          )
        else:
          append_filter.AddInput(
          self.particleDataTimeBinsSpecifiers.getBinSpatialLimits(bin_name)[spatialSubregionBinIndex]
          )

      append_filter.Update()

      if vtk.VTK_MAJOR_VERSION >= 6:
        probe.SetSourceData(append_filter.GetOutput())
      else:
        probe.SetSource(append_filter.GetOutput())
      probe.Update()

      numpy_data = numpy_support.vtk_to_numpy(probe.GetValidPoints())
  
      return numpy_data


    def setupHelicityTimeBins(self, particleDataTimeBinsSpecifiers):
        if (isinstance(particleDataTimeBinsSpecifiers, ParticleDataTimeBinsSpecifier)):
            self.particleDataTimeBinsSpecifiers = particleDataTimeBinsSpecifiers

    def close(self):
        self.output.close()

    def createBasicDataGroups(self):
        self.basicDataGroupNames = ["Index", "Coordinates", "Topology", "PLAP",
                                    "PRT", "alive", "velocity", "vorticity",
                                    "helical_flow_index",
                                    "absolute_helical_flow_index", "partition",
                                    "TimeWritten"]

        if self.particleDataTimeBinsSpecifiers is not None:
            for timeBin in self.particleDataTimeBinsSpecifiers.names():
                self.basicDataGroupNames.append(timeBin)
                self.basicDataGroupNames.append(timeBin + PARTICLE_RESIDENCE_TIME_NAMETAG)

        for groupName in self.basicDataGroupNames:
            self.output.create_group(groupName)

    def expandFinalTimestepStorageArrays(self, new_number_of_particles):
        old_data_size = len(self.finalStepArrayHandles[0])

        for h5_array in self.finalStepArrayHandles:
            tqdm.tqdm.write("resizing to {}".format(new_number_of_particles))
            # self.comm.Barrier()
            h5_array.resize(new_number_of_particles, axis=0)
        # self.comm.Barrier() 

        self.output["Index/Final/%d" % self.rank][old_data_size:new_number_of_particles] = \
            UNWRITTEN_VALUE
        self.numberOfParticles = new_number_of_particles

    def __createHdf5DataArray(self, array_name,
                              array_length, array_width, data_type):
        # Create the HDF5 array for each processor.
        # Stores them in finalStepArrayHandles for later use.
        self.comm.Barrier()
        for processor_index in range(self.numberOfProcessors):
            self.__createHdf5DataArraySingleSlice(array_name,
                                                  array_length,
                                                  array_width,
                                                  data_type,
                                                  processor_index)

    def __createHdf5DataArraySingleSlice(self, array_name,
                                         array_length,
                                         array_width,
                                         data_type,
                                         slice_index):

        # Determine whether the array to create is 1D or 3D
        if array_width is self.ARRAY_ONE_DIMENSIONAL:
            shape_tuple = (array_length,)
            max_shape_tuple = (None,)
        elif array_width is self.ARRAY_TWO_DIMENSIONAL:
            shape_tuple = (array_length, 2)
            max_shape_tuple = (None, 2)
        elif array_width is self.ARRAY_THREE_DIMENSIONAL:
            shape_tuple = (array_length, 3)
            max_shape_tuple = (None, 3)
        elif array_width is self.ARRAY_FOUR_DIMENSIONAL:
            shape_tuple = (array_length, 4)
            max_shape_tuple = (None, 4)
        else:
            raise RuntimeError("Unknown array_width parameter provided.")

        # print "creating exit data step array", array_name, slice_index
        self.finalStepArrayHandles.append(
            self.output[array_name].create_dataset(
                str(slice_index), shape_tuple,
                maxshape=max_shape_tuple, dtype=data_type
            )
        )

    def createOutputFieldsForParticleExitTimeData(self):
        # Also create group for final step. Here we can write results
        # progressively when particles exit the domain
        for groupName in self.basicDataGroupNames:
            self.output[groupName].create_group("Final")
        self.__createHdf5DataArray("Index/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_ONE_DIMENSIONAL, np.int64)
        self.__createHdf5DataArray("Coordinates/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_THREE_DIMENSIONAL, np.float64)
        self.__createHdf5DataArray("PLAP/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_ONE_DIMENSIONAL, np.float64)
        self.__createHdf5DataArray("PRT/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_ONE_DIMENSIONAL, np.float64)
        self.__createHdf5DataArray("alive/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_ONE_DIMENSIONAL, np.int64)
        self.__createHdf5DataArray("velocity/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_THREE_DIMENSIONAL, np.float64)
        self.__createHdf5DataArray("vorticity/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_THREE_DIMENSIONAL, np.float64)
        self.__createHdf5DataArray("helical_flow_index/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_ONE_DIMENSIONAL, np.float64)
        self.__createHdf5DataArray("absolute_helical_flow_index/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_ONE_DIMENSIONAL, np.float64)
        self.__createHdf5DataArray("partition/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_ONE_DIMENSIONAL, np.int64)

        self.__createHdf5DataArray("TimeWritten/Final",
                                   self.numberOfParticles,
                                   self.ARRAY_ONE_DIMENSIONAL, np.int64)
        for binName in self.particleDataTimeBinsSpecifiers.names():
            self.__createHdf5DataArray(binName + "/Final",
                                       self.numberOfParticles,
                                       self.ARRAY_ONE_DIMENSIONAL, np.float64)
            self.output[binName + "/Final/%d" % self.rank][:] = UNWRITTEN_VALUE

            self.__createHdf5DataArray(binName + PARTICLE_RESIDENCE_TIME_NAMETAG + "/Final",
                                       self.numberOfParticles,
                                       self.ARRAY_ONE_DIMENSIONAL, np.float64)

            self.output[binName + PARTICLE_RESIDENCE_TIME_NAMETAG + "/Final/%d" % self.rank][:] = UNWRITTEN_VALUE
        self.output["Index/Final/%d" % self.rank][:] = UNWRITTEN_VALUE
        self.output["PLAP/Final/%d" % self.rank][:] = UNWRITTEN_VALUE
        self.output["PRT/Final/%d" % self.rank][:] = UNWRITTEN_VALUE
        self.output["helical_flow_index/Final/%d" % self.rank][:] = \
            UNWRITTEN_VALUE
        self.output["absolute_helical_flow_index/Final/%d" % self.rank][:] =\
            UNWRITTEN_VALUE

        # Make another space after the highest-rank processor's final data to
        # store the sorted reduced data.
        # These are populated after the main time-loop finishes.
        #
        # They also contain sorted coordinate indices, so each array is one
        # bigger in one dimension (c.f. e.g. Coordinates/Final above in the
        # for-loop for the procs: (self.numberOfParticles,3) vs
        # (self.numberOfParticles,4)).
        #
        # todo replace this processors-plus-one data index design with
        # something tidier and more intuitive.
        self.__createHdf5DataArraySingleSlice("Coordinates/Final",
                                              self.numberOfParticles,
                                              self.ARRAY_FOUR_DIMENSIONAL, np.float64,
                                              self.numberOfProcessors + 1)
        self.__createHdf5DataArraySingleSlice("PLAP/Final",
                                              self.numberOfParticles,
                                              self.ARRAY_TWO_DIMENSIONAL, np.float64,
                                              self.numberOfProcessors + 1)
        self.__createHdf5DataArraySingleSlice("PRT/Final",
                                              self.numberOfParticles,
                                              self.ARRAY_TWO_DIMENSIONAL, np.float64,
                                              self.numberOfProcessors + 1)
        self.__createHdf5DataArraySingleSlice("helical_flow_index/Final",
                                              self.numberOfParticles,
                                              self.ARRAY_TWO_DIMENSIONAL, np.float64,
                                              self.numberOfProcessors + 1)
        self.__createHdf5DataArraySingleSlice(
            "absolute_helical_flow_index/Final",
            self.numberOfParticles,
            self.ARRAY_TWO_DIMENSIONAL, np.float64,
            self.numberOfProcessors + 1
        )
        for binName in self.particleDataTimeBinsSpecifiers.names():
            self.__createHdf5DataArraySingleSlice(
                binName + "/Final",
                self.numberOfParticles,
                self.ARRAY_TWO_DIMENSIONAL, np.float64,
                self.numberOfProcessors + 1
            )

            self.__createHdf5DataArraySingleSlice(
                binName + PARTICLE_RESIDENCE_TIME_NAMETAG + "/Final",
                self.numberOfParticles,
                self.ARRAY_TWO_DIMENSIONAL, np.float64,
                self.numberOfProcessors + 1
            )

    def createSpaceForThisStepsParticles(self, time_step_index,
                                         number_of_live_particles):
        self.output["Index"].create_dataset("%d" % time_step_index, (number_of_live_particles,), dtype=np.int64)
        self.output["Topology"].create_dataset("%d" % time_step_index, (number_of_live_particles,), dtype=np.int64)
        self.output["Coordinates"].create_dataset("%d" % time_step_index, (number_of_live_particles,3))
        self.output["PLAP"].create_dataset("%d" % time_step_index, (number_of_live_particles,), dtype=np.float64)
        self.output["PRT"].create_dataset("%d" % time_step_index, (number_of_live_particles,), dtype=np.float)
        self.output["alive"].create_dataset("%d" % time_step_index, (number_of_live_particles,), dtype=np.int64)
        self.output["velocity"].create_dataset("%d" % time_step_index, (number_of_live_particles,3), dtype=np.float)
        self.output["vorticity"].create_dataset("%d" % time_step_index, (number_of_live_particles,3), dtype=np.float)
        self.output["helical_flow_index"].create_dataset("%d" % time_step_index, (number_of_live_particles,), dtype=np.float)
        self.output["absolute_helical_flow_index"].create_dataset("%d" % time_step_index, (number_of_live_particles,), dtype=np.float)
        self.output["partition"].create_dataset("%d" % time_step_index, (number_of_live_particles,), dtype=np.int64)

        for binName in self.particleDataTimeBinsSpecifiers.names():
            self.output[binName].create_dataset("%d" % time_step_index,
                                                (number_of_live_particles,), dtype=np.float64)
            self.output[binName + PARTICLE_RESIDENCE_TIME_NAMETAG].create_dataset("%d" % time_step_index,
                                                (number_of_live_particles,), dtype=np.float)

    # Eventually, all the particle data arrays should be transitioned
    # to being "internally managed", so that we don't have to make
    # separate calls from outside to writeLocalParticleTimestepData().
    #
    # For now, only some of the arrays are truly "internally managed".
    #
    # todo implement this fully.
    def writeInternallyManagedLocalParticleTimestepData(self, time_step,
                                                        local_particle_data_slice):
        for bin_name in self.spacetime_bin_data_arrays:
            bin_path = '/' + bin_name + '/'
            # print "Writing into bin", bin_path
            self.__writeData(bin_path, time_step,
                             local_particle_data_slice,
                             self.spacetime_bin_data_arrays[bin_name])

    def writeLocalParticleTimestepData(self, field_name, time_step,
                                       data_placement_slice, data_array):
        self.__writeData(field_name, time_step,
                         data_placement_slice, data_array)


    def writeInternallyManagedFinalTimestepParticleData(self, simulation_end_particles_data_placement_slice):
        for bin_name in self.spacetime_bin_data_arrays:
            particle_exit_bin_path = bin_name + "/Final/"
            particle_exit_data = self.spacetime_bin_data_arrays[bin_name][:] # / dead_particle_bin_residence_times[bin_name]
            self.__writeData(particle_exit_bin_path, self.rank,
                             simulation_end_particles_data_placement_slice, particle_exit_data)

    # Eventually, all the particle data arrays should be transitioned
    # to being "internally managed", so that we don't have to make
    # separate calls from outside to writeLocalParticleTimestepData().
    #
    # For now, only some of the arrays are truly "internally managed".
    #
    # todo implement this fully.
    def writeInternallyManagedLocalJustExitedParticleData(self, just_exited_particles_data_placement_slice,
                                                          dead_particle_locations):
        # dead_particle_bin_residence_times = dict()

        # # Gather the amount of time that each particle spent
        # # in each of the user-defined particle residence 
        # # time bins.
        # #
        # # First, prepare storage space:
        # for bin_name in self.spacetime_bin_data_arrays:
        #     dead_particle_bin_residence_times[bin_name] = np.zeros((len(dead_particle_locations[0]),))

        # # Now gather the actual temporal residency times:
        # for particle_index, dead_particle_location in enumerate(dead_particle_locations[0]):
        #     # bin_residence_times = cardiac_cycle_timekeeper.getBinResidenceTimesForExitingParticle(prt[dead_particle_location])
        #     for bin_residence_time_name in self.spacetime_bin_residence_time_array_names:
        #         dead_particle_bin_residence_times[bin_residence_time_name][particle_index] = bin_residence_times[bin_residence_time_name]
        
        # dead_particle_systolic_residence_times = np.zeros((len(dead_particle_locations[0]),))
        # dead_particle_diastolic_residence_times = np.zeros((len(dead_particle_locations[0]),))
        # for index, dead_particle_location in enumerate(dead_particle_locations[0]):
        #     systolic_and_diastolic_residence_times = cardiac_cycle_timekeeper.getSystolicAndDiastolicResidenceTimesForExitingParticle(prt[dead_particle_location])
        #     dead_particle_systolic_residence_times[index] = systolic_and_diastolic_residence_times[CardiacCycleTimekeeper.TAG_SYSTOLE]
        #     dead_particle_diastolic_residence_times[index] = systolic_and_diastolic_residence_times[CardiacCycleTimekeeper.TAG_DIASTOLE]


        # todo : bring back this sanity check somehow
        #
        # entryIsZero_systole = [True if value==0 else False for value in dead_particle_systolic_residence_times]
        # dead_particle_systolic_residence_times[np.where(entryIsZero_systole)] = 1.0
        # if len(absolute_helical_flow_index_systole[dead_particle_locations][np.where(entryIsZero_systole)]) > 0:
        #     if np.max(absolute_helical_flow_index_systole[dead_particle_locations][np.where(entryIsZero_systole)] != 0):
        #         raise RuntimeError("Error","Systolic helical flow index nonzero for a particle which was not resident during systloe!" + str(absolute_helical_flow_index_systole[np.where(entryIsZero_systole)]))
        
        for bin_name in self.spacetime_bin_data_arrays:
            particle_exit_bin_path = bin_name + "/Final/"
            postprocessed_particle_exit_data = self.spacetime_bin_data_arrays[bin_name][dead_particle_locations] # / dead_particle_bin_residence_times[bin_name]
            
            self.writeLocalJustExitedParticleData(particle_exit_bin_path,
                                                  just_exited_particles_data_placement_slice,
                                                  postprocessed_particle_exit_data)


        # entryIsZero_systole = [True if value==0 else False for value in dead_particle_systolic_residence_times]
        # dead_particle_systolic_residence_times[np.where(entryIsZero_systole)] = 1.0
        # if len(absolute_helical_flow_index_systole[dead_particle_locations][np.where(entryIsZero_systole)]) > 0:
        #     if np.max(absolute_helical_flow_index_systole[dead_particle_locations][np.where(entryIsZero_systole)] != 0):
        #         raise RuntimeError("Error","Systolic helical flow index nonzero for a particle which was not resident during systloe!" + str(absolute_helical_flow_index_systole[np.where(entryIsZero_systole)]))
        
        # normalised_absolute_helical_flow_index_systole = absolute_helical_flow_index_systole[dead_particle_locations] / dead_particle_systolic_residence_times
        # data_manager.writeLocalJustExitedParticleData("absolute_helical_flow_index_systole/Final/", just_exited_particles_data_placement_slice, normalised_absolute_helical_flow_index_systole)

    def writeLocalJustExitedParticleData(self, field_name, data_placement_slice, data_array):
        self.__writeData(field_name, self.rank, data_placement_slice, data_array)

    def __writeData(self, field_name, finalFieldTag,
                    data_placement_slice, data_array):
        internal_h5_path = field_name + str(finalFieldTag)
        self.output[internal_h5_path][data_placement_slice] = data_array

    # Sometimes, the h5 file is used to pass data between threads.
    # todo: replace with a better method.
    def getOutputFileArray(self, internal_h5_path, time_step):
        full_internal_path = internal_h5_path + "/" + str(time_step)
        return self.output[full_internal_path][:]


    def extendAndRelocaliseDataBinArrays(self, local_particle_indices,
                                         time_step, numberOfNewParticles):
        for bin_name in self.spacetime_bin_data_arrays:

            current_bin_data = \
                self.getOutputFileArrayAliveParticlesOnly("/" + bin_name + "/", time_step)
            
            current_bin_data_with_space_for_new_data = \
                np.append(current_bin_data, np.zeros((numberOfNewParticles,)))

            self.spacetime_bin_data_arrays[bin_name] = current_bin_data_with_space_for_new_data[local_particle_indices]


    def relocaliseDataBinArrays(self, local_particle_indices, time_step):
        #todo critical make all these bin names be auto-generated
        # absolute_helical_flow_index_systole_global = self.getOutputFileArrayAliveParticlesOnly("/absolute_helical_flow_index_systole/", time_step)
        # absolute_helical_flow_index_diastole_global = self.getOutputFileArrayAliveParticlesOnly("/absolute_helical_flow_index_diastole/", time_step)

        # absolute_helical_flow_index_systole_residence_time_global = self.getOutputFileArrayAliveParticlesOnly("/absolute_helical_flow_index_systole"+PARTICLE_RESIDENCE_TIME_NAMETAG+"/", time_step)
        # absolute_helical_flow_index_diastole_residence_time_global = self.getOutputFileArrayAliveParticlesOnly("/absolute_helical_flow_index_diastole"+PARTICLE_RESIDENCE_TIME_NAMETAG+"/", time_step)

        # spacetime_bin_data_arrays_global = dict()
        # spacetime_bin_data_arrays_global['absolute_helical_flow_index_systole'] = absolute_helical_flow_index_systole_global
        # spacetime_bin_data_arrays_global['absolute_helical_flow_index_diastole'] = absolute_helical_flow_index_diastole_global
        # spacetime_bin_data_arrays_global['absolute_helical_flow_index_systole' + PARTICLE_RESIDENCE_TIME_NAMETAG] = absolute_helical_flow_index_systole_residence_time_global
        # spacetime_bin_data_arrays_global['absolute_helical_flow_index_diastole' + PARTICLE_RESIDENCE_TIME_NAMETAG] = absolute_helical_flow_index_diastole_residence_time_global

        for bin_name in self.spacetime_bin_data_arrays:
            # self.spacetime_bin_data_arrays[bin_name] = spacetime_bin_data_arrays_global[bin_name][local_particle_indices]
            self.spacetime_bin_data_arrays[bin_name] = self.getOutputFileArrayAliveParticlesOnly(bin_name, time_step)[local_particle_indices]


    # ugly design warning: this method must be the first call when reducing
    # arrays, because the index array reduction produces data which is
    # required to reduce any of the other arrays.
    def reduceIndexArrayAndSaveInH5File(self):
        final_indices = np.zeros(self.numberOfParticles, dtype=np.int64)

        final_offset = 0
        self.used_value_mask = {}
        for processor_id in range(self.numberOfProcessors):
            indices_proc_unmasked = self.output["/Index/Final/" + str(processor_id)][:]
            # populate this dictionary so that later reduce calls can access it
            self.used_value_mask[processor_id] = np.where(indices_proc_unmasked != UNWRITTEN_VALUE)

            indices_proc = indices_proc_unmasked[self.used_value_mask[processor_id]]
            nindices = indices_proc.shape[0]

            print("final_offset", final_offset, "nindices", nindices, "indices_proc", np.shape(indices_proc), "numberOfParticles",self.numberOfParticles)

            final_indices[final_offset:final_offset + nindices] = indices_proc
            final_offset += nindices

        self.sort_indices = np.argsort(final_indices)
        self.sorted_final_indices = final_indices[self.sort_indices]


    def reduceArrayAndSaveInH5File(self, h5_field_name,
                                   reduce_a_three_vector):

        if reduce_a_three_vector:
            reduced_field_builder = np.zeros((self.numberOfParticles, 3), dtype=np.float)
        else:
            reduced_field_builder = np.zeros(self.numberOfParticles, dtype=np.float64)

        final_offset = 0
        for processor_id in range(self.numberOfProcessors):
            h5_field_unmasked = self.output[h5_field_name + str(processor_id)][:]

            h5_field = h5_field_unmasked[self.used_value_mask[processor_id]]
            nindices = h5_field.shape[0]

            reduced_field_builder[final_offset:final_offset + nindices] = h5_field

            final_offset += nindices

        sorted_reduced_field_data = reduced_field_builder[self.sort_indices]

        # The reduced data is (confusingly) stored in a h5 path given by number of processors + 1.
        # the sibling paths 0 - numberOfProcessors contain per-processor data, so it's a bit
        # counter-intuitive that numberOfProcessors+1 should be something different.
        # todo find a more sensible alternative.
        reduced_data_field_index = str(self.numberOfProcessors + 1)
        h5_path = h5_field_name + reduced_data_field_index
        self.output[h5_path][:, 0] = self.sorted_final_indices
        if reduce_a_three_vector:
            self.output[h5_path][:, 1:] = sorted_reduced_field_data
        else:
            self.output[h5_path][:, 1] = sorted_reduced_field_data

        return sorted_reduced_field_data


    def reduceInternallyManagedArraysAndSaveInH5File(self):
        for bin_name in tqdm.tqdm(self.spacetime_bin_data_arrays, desc="Reducing internally managed hdf5 arrays..."):
            # Passes False because so far, all the internally-managed bins contain 1D data. This may need changing later.
            sorted_particle_data_array = self.reduceArrayAndSaveInH5File(bin_name + "/Final/", False)
            sorted_particle_data_array_vtu_vtk = numpy_support.numpy_to_vtk(sorted_particle_data_array)
            sorted_particle_data_array_vtu_vtk.SetName(bin_name)
            yield (sorted_particle_data_array_vtu_vtk, sorted_particle_data_array, sorted_particle_data_array_vtu_vtk)

            # sorted_particle_data_array_residence_time = self.reduceArrayAndSaveInH5File(bin_name + PARTICLE_RESIDENCE_TIME_NAMETAG + "/Final/", False)
            # sorted_particle_data_array_vtu_vtk_residence_time = numpy_support.numpy_to_vtk(sorted_particle_data_array_residence_time)
            # sorted_particle_data_array_vtu_vtk_residence_time.SetName(bin_name + PARTICLE_RESIDENCE_TIME_NAMETAG)
            # yield (sorted_particle_data_array_vtu_vtk_residence_time, sorted_particle_data_array_residence_time, sorted_particle_data_array_vtu_vtk_residence_time)

    # todo this doesnt really belong here; it should be somehow shifted into the particle manager
    # (avoiding using self.output along the way)
    def getOutputFileArrayAliveParticlesOnly(self, array_name, time_step):
        alive_global = np.array(self.output["/alive/" + str(time_step)][:], dtype=np.bool)
        array_including_dead_particles = self.getOutputFileArray(array_name, time_step)
        return array_including_dead_particles[alive_global]


    def __getLocalArrayValues(self, global_array):
        return global_array[np.where(self.particles_partition == self.rank)]


    def maskAndRepartitionDataArrays(self, time_step):
        for bin_name, bin_array in self.spacetime_bin_data_arrays.items():
            masked_repartitioned_data = self.getOutputFileArrayAliveParticlesOnly(bin_name, time_step)
            # if number_of_local_particles > 0:
            #     writeLocalParticleTimestepData("/" + bin_name + "/", time_step,
            #                                    local_particle_data_slice,
            #                                    masked_repartitioned_data)

        # self.spacetime_bin_data_arrays[array_name]
        # return self.output[internal_h5_path + str(time_step)][:]
