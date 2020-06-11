from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
import vtk
import numpy as np
from vtk.util import numpy_support as ns
import argparse
import mpi4py.MPI
import os
import particles

# This code is setup for discovering which particles end in a user-defined
# (customised in the code, here) hit-zone sphere. Useful for just examining
# the properties of particles which leave through a particular outlet.
#
# (you MUST set the termination region sphere manually (command line
# argument) to determine precisely what "leaving near the aortic outlet" means)


# case with aortic root present aortic outlet
# centreOfOutletSphereHitzone = np.array([143.6, 80.8, 207.0])
# radiusOfOutletSphereHitzone = 15.0 # outlet diamter is 22

# volunteer case aortic outlet
# centreOfOutletSphereHitzone = np.array([193.9, 101.1, 224.8])
# radiusOfOutletSphereHitzone = 12.0 # outlet diamter is 17.3

# study case aortic outlet
# centreOfOutletSphereHitzone = np.array([152.0, 59.2, 160.8])
# radiusOfOutletSphereHitzone = 18.0 # outlet diamter is 26.4

# study case LCC outlet (middle of the 3 ascending vessels)
# centreOfOutletSphereHitzone = np.array([127.5, -168.6, 131.8])
# radiusOfOutletSphereHitzone = 7.0 # outlet diamter is ~11

# study case brachiocephalic outlet (first (most proximal) of the 3 ascending
# vessels)
# centreOfOutletSphereHitzone = np.array([106.7, -153.3, 171.2])
# radiusOfOutletSphereHitzone = 8.0 # outlet diamter is ~13

def writeToMeanParticleDataSummaryFile(meanParticleValueToWrite,
                                       parsed_command_line_input):
    fullResultsFilePath = parsed_command_line_input.output_file
    with open(fullResultsFilePath, 'a') as meanResultsFile:
        meanResultsFile.write(str(meanParticleValueToWrite) + " ")


def finaliseMeanParticleDataSummaryFile(parsed_command_line_input):
    fullResultsFilePath = parsed_command_line_input.output_file
    with open(fullResultsFilePath, 'a') as meanResultsFile:
        friendlyDataTag = parsed_command_line_input.friendly_data_tag
        meanResultsFile.write(friendlyDataTag + "\n")


def computeMeanValueOverParticlesWithMinimumResidenceTime(
        particleDataValuesArray, particleResidenceTimeArray,
        minimumResidenceTime, consoleDisplayMessage):
    # Display computed mean indices in the console:
    particle_data_values_array_numpy = ns.vtk_to_numpy(particleDataValuesArray)
    particle_data_nans_and_early_exiting_particles_removed = [
        particle_data_value for (particle_data_value, residence_time)
        in zip(particle_data_values_array_numpy, particleResidenceTimeArray)
        if residence_time > minimumResidenceTime and not
        np.isnan(particle_data_value)]

    computed_result =\
        np.mean(particle_data_nans_and_early_exiting_particles_removed)

    print(consoleDisplayMessage + ":", computed_result, "computed from array "\
        "with shape",\
        np.shape(particle_data_nans_and_early_exiting_particles_removed))

    return computed_result


def computeMeanOverParticlesWeightedByResidenceTime(
    particleDataValuesArray, includeParticleInCalculationBooleanList,
        consoleDisplayMessage, particle_residence_time_array,
        minimum_residence_time):

    particleDataValuesArray_numpy = ns.vtk_to_numpy(particleDataValuesArray)
    particle_residence_time_array_numpy = ns.vtk_to_numpy(particle_residence_time_array)
    if len(particleDataValuesArray_numpy) != len(particle_residence_time_array_numpy):
        raise RuntimeError('Mismatch of particle data and residence time arrays.')

    residenceTimeScaledParticleDataValuesArray = old_div(particleDataValuesArray_numpy, particle_residence_time_array_numpy)

    # Modify the particle inclusion boolean array to avoid including any
    # particles in the calculation which just got divided by zero:
    residenceScaledIncludeParticleInCalculationBooleanList = [original_value if residence_time > minimum_residence_time else False for (original_value, residence_time) in zip(includeParticleInCalculationBooleanList, particle_residence_time_array_numpy)]

    return computeMeanValueOverParticlesFinishingInHitzone(
        ns.numpy_to_vtk(residenceTimeScaledParticleDataValuesArray), residenceScaledIncludeParticleInCalculationBooleanList,
        consoleDisplayMessage + " with residence time weighting and minimum residence time {}".format(minimum_residence_time))


def computeMeanValueOverParticlesFinishingInHitzone(
        particleDataValuesArray, includeParticleInCalculationBooleanList,
        consoleDisplayMessage):
    particle_data_values_array_numpy = ns.vtk_to_numpy(particleDataValuesArray)

    # mean absolute helical flow index only of particles which leave near the
    # aortic outlet
    particle_data_particles_finishing_in_hitzone_and_not_nan = \
        [particle_data_value for (particle_data_value, useParticle) in
         zip(particle_data_values_array_numpy,
             includeParticleInCalculationBooleanList)
         if useParticle and not np.isnan(particle_data_value)]

    computed_result = np.mean(
        particle_data_particles_finishing_in_hitzone_and_not_nan)

    # Display computed mean indices in the console:
    print(consoleDisplayMessage + " (after unwanted particle removal (time/space thresholding or inf values removed)): ====>",\
        computed_result, "computed from array with shape",\
        np.shape(particle_data_particles_finishing_in_hitzone_and_not_nan),\
        "\n")

    return computed_result


def addArrayToOutputVtpFile(arrayToAdd, datasetToAddTo):
    return datasetToAddTo.GetPointData().AddArray(arrayToAdd)


def computeMeanValueOverParticlesAndAddToOutputFiles(
        inputArrayName, inputDataReader, particleResidenceTimeArray,
        minimumResidenceTime, includeParticleInCalculationBooleanList,
        outputVtpDataset, parsed_command_line_input):

        particleDataArray = inputDataReader.GetOutput().GetPointData().\
            GetArray(inputArrayName)

        if particleDataArray is not None:
            addArrayToOutputVtpFile(particleDataArray, outputVtpDataset)
            # Mean value of whatever per-particle metric we're working with,
            # but omitting any particles which were not in the domain for at
            # least the minimumResidenceTime
            print("\n\n")
            mean_over_particles = \
                computeMeanValueOverParticlesWithMinimumResidenceTime(
                    particleDataArray, particleResidenceTimeArray,
                    minimumResidenceTime, "Mean of {} with no particles removed".format(inputArrayName)
                )

            writeToMeanParticleDataSummaryFile(mean_over_particles,
                                               parsed_command_line_input)

            # Mean value of whatever per-particle metric we're working with,
            # but omitting any particles which did not leave the domain or
            # finish the simulation in the hitbox (or, as far as this function
            # is concerned, omitting any particles which do not have "True" in
            # includeParticleInCalculationBooleanList).
            if (includeParticleInCalculationBooleanList is not None and
               includeParticleInCalculationBooleanList.count(True) > 0):
                mean_over_particles_in_hitzone = \
                    computeMeanValueOverParticlesFinishingInHitzone(
                        particleDataArray,
                        includeParticleInCalculationBooleanList,
                        "Mean " + inputArrayName
                    )
                writeToMeanParticleDataSummaryFile(
                    mean_over_particles_in_hitzone,
                    parsed_command_line_input)
            elif includeParticleInCalculationBooleanList is None:
                print("Information: No hitzone specified.")
                writeToMeanParticleDataSummaryFile(np.nan,
                                                   parsed_command_line_input)
            else:
                print("Information: No particles were in the hitzone sphere.")
                writeToMeanParticleDataSummaryFile(np.nan,
                                                   parsed_command_line_input)


def parseCommandLineArguments():
    command_line_argument_parser = \
        argparse.ArgumentParser(description=(
            'Postprocess particle tracking results'
            ' for final-position particle visualisation,'
            ' and compute some mean quantitites on the'
            ' particles.')
        )
    command_line_argument_parser.add_argument(
        'input_vtu_file',
        help='.vtu output file from particle tracking simulation'
    )
    command_line_argument_parser.add_argument(
        '-o',
        dest='output_file', default='particle_means.txt',
        help='file to'
             ' contain summary data about metrics computed over the particles.'
             ' May be the same file as used by other runs of this script;'
             ' the data will be appended'
    )
    command_line_argument_parser.add_argument(
        '-t', dest='friendly_data_tag', default='',
        help='Human-readable tag to identify'
        ' the output data in the output_file passed using the -o flag'
    )
    command_line_argument_parser.add_argument(
        '-c', dest='particle_hitzone_centre',
        help='Centre of a sphere. Particle metrics'
        ' will be reported on just the particles which lie in this sphere,'
        ' in addition to the metrics which include all particles regardless of'
        ' location. Format: x y z, space-separated', type=float, nargs=3
    )
    command_line_argument_parser.add_argument(
        '-r', dest='particle_hitzone_radius',
        help='Radius of the sphere. Particle'
        ' metrics will be reported on just the particles which lie in this '
        'sphere, in addition to the metrics which include all particles '
        'regardless of location.', type=float
    )
    command_line_argument_parser.add_argument(
        '-i', dest='    h5_input_file_name',
        help='hdf5 input file containing tracking output data for'
        ' individual particles. All particles which did not end the '
        'simulation in the sphere will be removed from this data, '
        'and the result written to <H5_INPUT_FILE_NAME>-copy.h5.',
        type=str
    )
    # command_line_argument_parser.add_argument(
    #     '-s', dest='strip_nonhitzone_particles', action='store_true',
    #     help='If this flag appears on the command line, it removes all '
    #     'particles which do not reach the '
    #     'hitzone by the end of the simulation from the output data.',
    # )

    parsed_aruments = command_line_argument_parser.parse_args()
    parsed_arguments_dict = vars(parsed_aruments)
    print("\n---------------------------------------------")
    print("Running with the following configuration:")
    print("---------------------------------------------")
    for argument in parsed_arguments_dict:
        print(argument, "=", parsed_arguments_dict[argument])
    print("---------------------------------------------\n")

    return parsed_aruments

def vtu2finalposition(parsed_command_line_input, config_manager):

    centreOfOutletSphereHitzone = parsed_command_line_input.particle_hitzone_centre
    radiusOfOutletSphereHitzone = np.array(parsed_command_line_input.
                                           particle_hitzone_radius)

    useHitzoneSphere = (centreOfOutletSphereHitzone is not None) and (radiusOfOutletSphereHitzone is not None)

    minimum_residence_time = 0.0

    if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:

        # VTU File produced at the end of the particle tracking
        fname = parsed_command_line_input.input_vtu_file
        fname_no_ext = fname.split('.')[:-1]
        fname_no_ext = ''.join(fname_no_ext)

        # Read final particle file
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(fname)
        reader.Update()

        nparticles = reader.GetOutput().GetNumberOfPoints()

        # Read "Coordinates" array and build polydata with points
        # located at corresponding coordinates

        # Points
        final_coordinates = np.copy(ns.vtk_to_numpy(reader.GetOutput().
                                    GetPointData().GetArray("Coordinates")))
        final_coordinates_pts = vtk.vtkPoints()
        final_coordinates_pts.SetData(reader.GetOutput().GetPointData().
                                      GetArray("Coordinates"))

        # Vertices
        final_connectivity_cells = vtk.vtkCellArray()
        final_connectivity = np.zeros((nparticles, 2), dtype=np.int64)
        final_connectivity[:, 0] = 1
        final_connectivity[:, 1] = list(range(nparticles))
        final_connectivity = np.reshape(final_connectivity, (nparticles * 2,))
        final_connectivity_vtk = ns.numpy_to_vtkIdTypeArray(final_connectivity)
        final_connectivity_cells.SetCells(nparticles, final_connectivity_vtk)

        # New dataset
        dataset = vtk.vtkPolyData()
        dataset.SetPoints(final_coordinates_pts)
        dataset.SetVerts(final_connectivity_cells)

        # Copy other particles arrays (e.g. PRT, PLAP, FTLE)
        prt_vtk = reader.GetOutput().GetPointData().GetArray("PRT")
        dataset.GetPointData().AddArray(prt_vtk)
        prt_numpy_array = ns.vtk_to_numpy(prt_vtk)

        dataset.GetPointData().AddArray(reader.GetOutput().GetPointData().
                                        GetArray("FTLE"))
        dataset.GetPointData().AddArray(reader.GetOutput().GetPointData().
                                        GetArray("PLAP"))

        PARTICLE_RESIDENCE_TIME_NAMETAG = "_particle_residence_time"
        for custom_particle_bin_name in config_manager.particleBinNames():

            print("---------------------------------\n")

            particle_residence_time_array_for_this_bin = reader.GetOutput().GetPointData().\
                                        GetArray(custom_particle_bin_name + PARTICLE_RESIDENCE_TIME_NAMETAG)
            dataset.GetPointData().AddArray(particle_residence_time_array_for_this_bin)

            particle_data_array_for_this_bin = reader.GetOutput().GetPointData().\
                                        GetArray(custom_particle_bin_name)
            dataset.GetPointData().AddArray(particle_data_array_for_this_bin)

            particle_value_not_zero_or_infinite = [(not np.isinf(particle_value) and particle_value > 0.0 ) for particle_value in
                                              ns.vtk_to_numpy(particle_data_array_for_this_bin)]

            # Compute the mean residence time for this bin:
            computeMeanValueOverParticlesAndAddToOutputFiles(
                custom_particle_bin_name + PARTICLE_RESIDENCE_TIME_NAMETAG, reader, prt_numpy_array,
                minimum_residence_time, particle_value_not_zero_or_infinite,
                dataset, parsed_command_line_input)

            # Compute the mean value of the data accumulated in this bin:
            computeMeanValueOverParticlesAndAddToOutputFiles(
                custom_particle_bin_name, reader, prt_numpy_array,
                minimum_residence_time, particle_value_not_zero_or_infinite,
                dataset, parsed_command_line_input)

            # Compute the mean value per residence time for the particles
            # Note that this doesn't currently save anything to the output
            # files - it just prints the value to the console
            computeMeanOverParticlesWeightedByResidenceTime(
                            particle_data_array_for_this_bin,
                            particle_value_not_zero_or_infinite,
                            "Mean " + custom_particle_bin_name + PARTICLE_RESIDENCE_TIME_NAMETAG + " zero and infinite values removed",
                            particle_residence_time_array_for_this_bin,
                            minimum_residence_time
                        )

        # Write output
        dataset_writer = vtk.vtkXMLPolyDataWriter()
        dataset_writer.SetFileName(fname_no_ext + ".vtp")
        dataset_writer.SetInput(dataset)
        dataset_writer.Update()

    if useHitzoneSphere:
        if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
            # seconds; cutoff minimum time a particle stays in the bloodstream
            # for before we include it in the helical flow index calculation.

            numberOfParticles = np.size(final_coordinates, 0)

            if centreOfOutletSphereHitzone is not None and \
                    radiusOfOutletSphereHitzone is not None:
                finalParticleDistancesFromSphereHitzoneCentre = \
                    np.zeros((numberOfParticles,))
                for particleIndex in range(numberOfParticles):
                    finalParticleDistancesFromSphereHitzoneCentre[particleIndex] = \
                        np.linalg.norm(centreOfOutletSphereHitzone -
                                       final_coordinates[particleIndex, :])

                particle_finishes_in_hitzone = \
                    [True if finalDistance < radiusOfOutletSphereHitzone else False
                     for finalDistance
                     in finalParticleDistancesFromSphereHitzoneCentre]
            else:
                particle_finishes_in_hitzone = None

            computeMeanValueOverParticlesAndAddToOutputFiles(
                "PLAP", reader, prt_numpy_array,
                minimum_residence_time, particle_finishes_in_hitzone,
                dataset, parsed_command_line_input)

            # particle_finishes_in_hitzone2 = [True if finalDistance <
            #              radiusOfOutletSphereHitzone else False for
            #              finalDistance in
            #              finalParticleDistancesFromSphereHitzoneCentre]
            # particle_finishes_in_hitzone = [True for _ in
            # particle_finishes_in_hitzone2]

            # This does NOT have the absolute value of the local normalised helicity
            # taken before summing.
            # See below for the absolute value version.
            # See Morbiducci et al. 2009 "In Vivo Quantification of Helical Blood
            # Flow in Human Aorta by Time-Resolved
            #                             Three-Dimensional Cine Phase Contrast
            #                             Magnetic Resonance Imaging"
            computeMeanValueOverParticlesAndAddToOutputFiles(
                "Helical Flow Index", reader, prt_numpy_array,
                minimum_residence_time, particle_finishes_in_hitzone,
                dataset, parsed_command_line_input)

            # This has the absolute value of the local normalised helicity taken before
            # summing. See below for the absolute value version.
            # See Morbiducci et al. 2009 "In Vivo Quantification of Helical Blood
            # Flow in Human Aorta by Time-Resolved
            #                             Three-Dimensional Cine Phase Contrast
            #                             Magnetic Resonance Imaging"
            computeMeanValueOverParticlesAndAddToOutputFiles(
                "Absolute Helical Flow Index", reader, prt_numpy_array,
                minimum_residence_time, particle_finishes_in_hitzone,
                dataset, parsed_command_line_input)

            # This has the absolute value of the local normalised helicity taken before
            # summing. See below for the absolute value version.
            # See Morbiducci et al. 2009 "In Vivo Quantification of Helical Blood
            # Flow in Human Aorta by Time-Resolved
            #                             Three-Dimensional Cine Phase Contrast
            #                             Magnetic Resonance Imaging"
            computeMeanValueOverParticlesAndAddToOutputFiles(
                "Absolute Helical Flow Index Systole", reader, prt_numpy_array,
                minimum_residence_time, particle_finishes_in_hitzone,
                dataset, parsed_command_line_input)

            # This has the absolute value of the local normalised helicity taken
            # before summing. See above for the non-absolute value version.
            # See Morbiducci et al. 2009 "In Vivo Quantification of Helical Blood
            #                             Flow in Human Aorta by Time-Resolved
            #                             Three-Dimensional Cine Phase Contrast
            #                             Magnetic Resonance Imaging"
            #
            # DIASTOLE ONLY (with "systole" as defined by the user when running the
            #                original particle tracking):
            computeMeanValueOverParticlesAndAddToOutputFiles(
                "Absolute Helical Flow Index Diastole", reader, prt_numpy_array,
                minimum_residence_time, particle_finishes_in_hitzone,
                dataset, parsed_command_line_input)


            finaliseMeanParticleDataSummaryFile(parsed_command_line_input)

            # if True:  # parsed_command_line_input.strip_nonhitzone_particles:
            #     invariant_particle_index_vtk = reader.GetOutput().GetPointData().\
            #         GetArray("Helical Flow Index")
            #     invariant_particle_index_numpy =\
            #         ns.vtk_to_numpy(invariant_particle_index_vtk)
            #     invariant_particle_index_numpy_masked = [
            #         particle_index for particle_index, particle_in_hitzone in
            #         zip(invariant_particle_index_numpy, particle_finishes_in_hitzone)
            #         if particle_in_hitzone
            #     ]
            #     print invariant_particle_index_numpy_masked

            # Find the indices of the Trues in the particle arrays
            # (i.e. the global indices of the hitzone-finishing particles)
            hitzone_particle_indices = []
            start_index = 0
            try:
                while start_index < numberOfParticles:
                    index_of_particle_in_hitzone =\
                        particle_finishes_in_hitzone.index(True, start_index,
                                                           numberOfParticles)
                    hitzone_particle_indices.\
                        append(index_of_particle_in_hitzone)

                    start_index = index_of_particle_in_hitzone + 1
            except ValueError:
                print("ValueError")
            print("Found " + str(len(hitzone_particle_indices)) + " particles in the specified spherical hitzone.")
        else:
            # just for the ranks that will receive the MPI broadcast
            # of hitzone_particle_indices:
            hitzone_particle_indices = None

        hitzone_particle_indices = mpi4py.MPI.COMM_WORLD.bcast(hitzone_particle_indices, root=0)

        mpi4py.MPI.COMM_WORLD.Barrier()

        if parsed_command_line_input.h5_input_file_name is not None:
            h5_output_file_name = os.path.splitext(parsed_command_line_input.h5_input_file_name)[0] + \
                                          '-r' + str(radiusOfOutletSphereHitzone) + \
                                          '-c' + str(centreOfOutletSphereHitzone[0]) + \
                                          '-' + str(centreOfOutletSphereHitzone[1]) + \
                                          '-' + str(centreOfOutletSphereHitzone[2]) + \
                                          '.h5'

            import remove_nonhitzone_particles
            remove_nonhitzone_particles.remove_nonhitzone_particles(hitzone_particle_indices,
                                                                    mpi4py.MPI.COMM_WORLD,
                                                                    parsed_command_line_input.h5_input_file_name,
                                                                    h5_output_file_name
                                                                    )


if __name__ == "__main__":

    custom_config_file_name = 'particle_config.json'
    config_manager = particles.Configuration(custom_config_file_name)

    command_line_input = parseCommandLineArguments()
    vtu2finalposition(command_line_input, config_manager)
