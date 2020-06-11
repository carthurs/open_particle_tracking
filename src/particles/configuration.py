from __future__ import print_function
from builtins import object
import os
import json
import warnings

ConfigKeys = {
               "MAX_PARTICLE_REINJECTIONS": "maximum particle reinjections",
               "STEPS_BETWEEN_REINJECTION": "tracking steps between each reinjection",
               "STEPS_BEFORE_FIRST_REINJECTION": "steps before first reinjection",  # note - this affects the first reinjection - not the first injection.
               "SPACE_TIME_BINS": "space time bins",
               "BIN_NAME": "bin name",
               "BIN_FILE_NAMES": "bin file names",
               "BIN_TIME_INTERVALS": "bin time intervals",
               "BIN_TIME_INTERVAL_START": "start",
               "BIN_TIME_INTERVAL_END": "end",
               "BIN_DATA": "bin data",
               "INCLUDE_VORTICITY_COMPUTATIONS": "include vorticity computations",
               "MPI_SUPPORT_FOR_H5PY_AVAILABLE": "mpi support for h5py available",
               "MESH_MANUALLY_EXTRACTED_FROM_GEOMBC": "mesh manually extracted from geombc",  # Set True if you got the mesh .coordinantes and .connectivity from geombc.dat.1 using GeombcReader.py; False otherwise.
               "DATA_FILE_BASE_NAME": "data file base name",
               "INPUT_DATA_START_TIMESTEP": "input data start timestep",
               "INPUT_DATA_END_TIMESTEP": "input data end timestep",
               "TIMESTEPS_BETWEEN_RESTARTS_IN_INPUT_DATA": "timesteps between restarts in input data",
               "REAL_TIME_BETWEEN_RESTARTS_IN_INPUT_DATA": "real time between restarts in input data",
               "TRACKING_SIMULATION_STARTING_TIMESTEP": "tracking simulation starting timestep",
               "NUMBER_OF_CYCLES_TO_TRACK_FOR": "number of cycles to track for",
               "WALL_HAS_DISPLACEMENT_FIELD": "wall has displacement field",
               "STEPS_BETWEEN_REPARTITIONING_PARTICLES_AND_WRITING_OUTPUT": "steps between repartitioning particles and writing output",
               "REAL_TIME_THROUGH_CARDIAC_CYCLE_WHEN_SIMULATION_STARTS": "real time through cardiac cycle when simulation starts",
               "REAL_TIME_OF_FIRST_SYSTOLE_START": "real time of first systole start",
               "REAL_TIME_OF_FIRST_SYSTOLE_END": "real time of first systole end",
               "CARDIAC_CYCLE_LENGTH": "cardiac cycle length"
             }

'''
Manages the configuration specified by the user in a .json file.
The possible key names are listed in the dictionary ConfigKeys, above.

This class uses a default configuration file first, then over-rides
it on a per-value basis using the custom json config file, specified
by the user.
'''
class Configuration(object):
    def __init__(self, full_path_to_custom_config_file):
        self.__loadDefaultConfig()
        self.__overrideWithCustomConfig(full_path_to_custom_config_file)
        # self.cached_bin_names = None
        

    def __loadDefaultConfig(self):
        this_script_path = os.path.dirname(os.path.abspath(__file__))
        default_configuration_file_name = 'default_config.json'
        with open(os.path.join(this_script_path, default_configuration_file_name), 'r') as default_config:
            self.configuration = json.load(default_config)

    def __overrideWithCustomConfig(self, full_path_to_custom_config_file):
        try:
          with open(full_path_to_custom_config_file, 'r') as custom_config:
              self.configuration.update(json.load(custom_config))
        except IOError:
          warnings.warn("No custom particles_config.json found in working directory. Proceeding with default configuration.")


    def maxReinjections(self):
        return int(self.configuration[ConfigKeys["MAX_PARTICLE_REINJECTIONS"]])


    def stepsBetweenReinjections(self):
        return int(self.configuration[ConfigKeys["STEPS_BETWEEN_REINJECTION"]])


    def minimumStepsBeforeFirstReinjection(self):
        return int(self.configuration[ConfigKeys["STEPS_BEFORE_FIRST_REINJECTION"]])


    def numberOfParticleBins(self):
        return len(self.configuration[ConfigKeys["SPACE_TIME_BINS"]])


    def particleBinNames(self):
        for spacetime_bin in self.configuration[ConfigKeys["SPACE_TIME_BINS"]]:
            yield spacetime_bin[ConfigKeys["BIN_NAME"]]


    def particleBinTimeIntervals(self, bin_name):
        bin_exists = False
        for spacetime_bin in self.configuration[ConfigKeys["SPACE_TIME_BINS"]]:
            if spacetime_bin[ConfigKeys["BIN_NAME"]] is bin_name:
                bin_exists = True
                if ConfigKeys["BIN_TIME_INTERVALS"] in spacetime_bin:
                  for time_interval in spacetime_bin[ConfigKeys["BIN_TIME_INTERVALS"]]:
                      start_time = float(time_interval[ConfigKeys["BIN_TIME_INTERVAL_START"]])
                      end_time = float(time_interval[ConfigKeys["BIN_TIME_INTERVAL_END"]])
                      yield (start_time, end_time)
                else:
                  # If no time bin was specified, just enable the bin for the whole cardiac cycle
                  input_data_duration_in_seconds = float(self.configuration[ConfigKeys["INPUT_DATA_END_TIMESTEP"]] - self.configuration[ConfigKeys["INPUT_DATA_START_TIMESTEP"]]) / float(self.configuration[ConfigKeys["TIMESTEPS_BETWEEN_RESTARTS_IN_INPUT_DATA"]]) * self.configuration[ConfigKeys["REAL_TIME_BETWEEN_RESTARTS_IN_INPUT_DATA"]]
                  yield (0.0, input_data_duration_in_seconds)

        if not bin_exists:
            raise RuntimeError("No bin with name matching " + bin_name + " was found in particleBinTimeIntervals.")


    def particleBinMeshFileNames(self, bin_name):
        bin_exists = False
        for spacetime_bin in self.configuration[ConfigKeys["SPACE_TIME_BINS"]]:
            if spacetime_bin[ConfigKeys["BIN_NAME"]] is bin_name:
                bin_exists = True
                try:
                    for mesh_name in spacetime_bin[ConfigKeys["BIN_FILE_NAMES"]]:
                        yield mesh_name
                except KeyError:
                    print("No spatial localisation found for bin {bin_name}".format(bin_name=bin_name))
                    return

        if not bin_exists:
            raise RuntimeError("No bin with name matching " + bin_name + " was found in particleBinSpatialRegions.")

    def getDataNameForBin(self, bin_name):
        bin_exists = False
        for spacetime_bin in self.configuration[ConfigKeys["SPACE_TIME_BINS"]]:
            if spacetime_bin[ConfigKeys["BIN_NAME"]] is bin_name:
                bin_exists = True
                return spacetime_bin[ConfigKeys["BIN_DATA"]]

        if not bin_exists:
            raise RuntimeError("No bin with name matching " + bin_name + " was found in particleBinSpatialRegions.")


    def includeVorticityComputations(self):
        return self.configuration[ConfigKeys["INCLUDE_VORTICITY_COMPUTATIONS"]]

    def mpiSupportForH5pyAvailable(self):
        return self.configuration[ConfigKeys["MPI_SUPPORT_FOR_H5PY_AVAILABLE"]]


    def meshManuallyExtractedFromGeombc(self):
        return self.configuration[ConfigKeys["MESH_MANUALLY_EXTRACTED_FROM_GEOMBC"]]


    def fname(self):
        return self.configuration[ConfigKeys["DATA_FILE_BASE_NAME"]]


    def cycle_start(self):
        return self.configuration[ConfigKeys["INPUT_DATA_START_TIMESTEP"]]


    def cycle_stop(self):
        return self.configuration[ConfigKeys["INPUT_DATA_END_TIMESTEP"]]


    def cycle_step(self):
        return self.configuration[ConfigKeys["TIMESTEPS_BETWEEN_RESTARTS_IN_INPUT_DATA"]]


    def step_start(self):
        return self.configuration[ConfigKeys["TRACKING_SIMULATION_STARTING_TIMESTEP"]]


    def ncycles(self):
        return self.configuration[ConfigKeys["NUMBER_OF_CYCLES_TO_TRACK_FOR"]]


    def dt(self):
        return self.configuration[ConfigKeys["REAL_TIME_BETWEEN_RESTARTS_IN_INPUT_DATA"]]


    def disp(self):
        return self.configuration[ConfigKeys["WALL_HAS_DISPLACEMENT_FIELD"]]


    def repartition_frequency_in_simulation_steps(self):
        return self.configuration[ConfigKeys["STEPS_BETWEEN_REPARTITIONING_PARTICLES_AND_WRITING_OUTPUT"]]


    def simulationStartTime(self):
        return self.configuration[ConfigKeys["REAL_TIME_THROUGH_CARDIAC_CYCLE_WHEN_SIMULATION_STARTS"]]


    def firstSystoleStartTime(self):
        return self.configuration[ConfigKeys["REAL_TIME_OF_FIRST_SYSTOLE_START"]]


    def firstSystoleEndTime(self):
        return self.configuration[ConfigKeys["REAL_TIME_OF_FIRST_SYSTOLE_END"]]


    def cardiacCycleLength(self):
        return self.configuration[ConfigKeys["CARDIAC_CYCLE_LENGTH"]]

