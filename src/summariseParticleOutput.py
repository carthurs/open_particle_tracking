from __future__ import division
from __future__ import print_function
# Postprocess the .h5 file output of crimsonParticleTracking.py to summarise the data recorded on particles.
# Will read the particle_config.json file from the simulation folder (which should be the working dir for this
# script) in order to provide per-bin stats. Expects the relative paths from the working dir to the bins to match those
# given in the particle_config.json

from past.utils import old_div
import h5py
import particles
import numpy
import pandas
import matplotlib.pyplot
from matplotlib.backends.backend_pdf import PdfPages
import re
from multi_key_dict import multi_key_dict
import copy


def pairUpSystoleAndDiastoleBins(config_manager):
    systole_regex = re.compile('systole', re.IGNORECASE)
    diastole_regex = re.compile('diastole', re.IGNORECASE)

    systole_diastole_bin_pairs = {}
    paired_bin_joint_names = multi_key_dict()

    for bin_index, bin_name in enumerate(config_manager.particleBinNames()):
        if systole_regex.match(bin_name):
            for bin_index2, bin_name2 in enumerate(config_manager.particleBinNames()):
                bin_name_without_systole_substring = systole_regex.sub('', bin_name)  # removes all occurences of "systole" from the string
                bin_name_without_diastole_substring = diastole_regex.sub('', bin_name2)

                if bin_name_without_systole_substring == bin_name_without_diastole_substring:
                    print("Pairing bins {} and {}, because I think they're systole and diastole of the same spatial location. A combined analysis on these will additionally be performed.".format(bin_name, bin_name2))
                    systole_diastole_bin_pairs[bin_index] = bin_index2
                    systole_diastole_bin_pairs[bin_index2] = bin_index

                    paired_bin_joint_names[bin_index, bin_index2] = systole_regex.sub('systole+diastole', bin_name)  # same as bin_name_without_diastole_substring

    return systole_diastole_bin_pairs, paired_bin_joint_names


def generate_summaries(config_manager):

    data_file_base_name = "{}-particles-{}".format(config_manager.fname(), config_manager.cycle_start())
    data_file_name_h5 = "{}.h5".format(data_file_base_name)

    systole_diastole_bin_pair_indices, paired_bin_names = pairUpSystoleAndDiastoleBins(config_manager)
    number_of_systole_diastole_bin_pairs = old_div(len(systole_diastole_bin_pair_indices),2)

    if 2*number_of_systole_diastole_bin_pairs != len(systole_diastole_bin_pair_indices):
        raise RuntimeError("Odd number of bin index pairs computed. Code must be buggy...")

    figure_output_pdf = PdfPages(data_file_base_name + ".pdf")
    with h5py.File(data_file_name_h5, 'r') as simulationOutputFile:

        number_of_output_dataframe_rows = 3*config_manager.numberOfParticleBins() + 2*number_of_systole_diastole_bin_pairs  # space for combining bin pair data too

        output_data_column_names = ['Mean over Particles', 'Minimum over Particles', 'Maximum over Particles', 'Number of Particles Involved']
        output_data_row_names = ['none'] * number_of_output_dataframe_rows

        particle_summary_data = numpy.zeros((number_of_output_dataframe_rows, len(output_data_column_names)))

        simulation_end_data_field_name = None
        for bin_index, bin_name in enumerate(config_manager.particleBinNames()):

            # on the first pass through the enclosing loop, get the index of the final data field
            # that we need to access on all subsequent calls:
            if simulation_end_data_field_name is None:
                for field_name in list(simulationOutputFile[bin_name]['Final'].keys()):
                    pass
                simulation_end_data_field_name = field_name  # i.e. set this to the last field name from the previous for loop

            current_bin_data = numpy.array(simulationOutputFile[bin_name]['Final'][simulation_end_data_field_name])[:, 1]
            current_bin_residence_times = numpy.array(simulationOutputFile[bin_name + '_particle_residence_time']['Final'][simulation_end_data_field_name])[:, 1]

            nonzero_residency_time_mask = numpy.where(current_bin_residence_times > 0.001)
            current_bin_nonzero_residence_times = current_bin_residence_times[nonzero_residency_time_mask]
            current_bin_data_for_particles_with_nonzero_residence_time = current_bin_data[nonzero_residency_time_mask]

            data_per_unit_time_for_current_bin = old_div(current_bin_data_for_particles_with_nonzero_residence_time, current_bin_nonzero_residence_times)

            # append histogram for current dataset to the pdf:
            number_of_bins = 256
            x_axis_length = 500
            matplotlib.pyplot.hist(data_per_unit_time_for_current_bin, bins=numpy.linspace(0, x_axis_length, num=number_of_bins), density=True)
            matplotlib.pyplot.axis([0, x_axis_length, 0, 0.05])
            matplotlib.pyplot.xlabel("Time-Averaged {} Acquisition Rate".format(config_manager.getDataNameForBin(bin_name)))
            matplotlib.pyplot.ylabel("Probability Density over Particles")
            matplotlib.pyplot.title("Time-Averaged {} Acquisition Rate for bin {}".format(config_manager.getDataNameForBin(bin_name), bin_name))
            matplotlib.pyplot.plot()
            figure_output_pdf.savefig()
            matplotlib.pyplot.cla()

            x_axis_length = 1.5
            matplotlib.pyplot.hist(current_bin_nonzero_residence_times, bins=numpy.linspace(0, x_axis_length, num=number_of_bins),
                                   density=True, facecolor='g')
            matplotlib.pyplot.axis([0, x_axis_length, 0, 10])
            matplotlib.pyplot.xlabel(
                "Residence Time in {}".format(bin_name))
            matplotlib.pyplot.ylabel("Probability Density over Particles")
            matplotlib.pyplot.title(
                "Particle Residence Times in bin {}".format(bin_name))
            matplotlib.pyplot.plot()
            figure_output_pdf.savefig()
            matplotlib.pyplot.cla()


            print(bin_name, "mean per unit time:", numpy.mean(data_per_unit_time_for_current_bin), "min:", numpy.min(data_per_unit_time_for_current_bin), "max:", numpy.max(data_per_unit_time_for_current_bin))

            output_data_row_names[bin_index] = "Time-Averaged {} Acquisition Rate for bin {} (excluding zeros)".format(config_manager.getDataNameForBin(bin_name), bin_name)
            particle_summary_data[bin_index, 0] = numpy.mean(data_per_unit_time_for_current_bin)
            particle_summary_data[bin_index, 1] = numpy.min(data_per_unit_time_for_current_bin)
            particle_summary_data[bin_index, 2] = numpy.max(data_per_unit_time_for_current_bin)
            particle_summary_data[bin_index, 3] = len(data_per_unit_time_for_current_bin)

            shifted_index = bin_index + config_manager.numberOfParticleBins()
            output_data_row_names[shifted_index] = "residence times " + bin_name + " (excluding zeros)"
            particle_summary_data[shifted_index, 0] = numpy.mean(current_bin_nonzero_residence_times)
            particle_summary_data[shifted_index, 1] = numpy.min(current_bin_nonzero_residence_times)
            particle_summary_data[shifted_index, 2] = numpy.max(current_bin_nonzero_residence_times)
            particle_summary_data[shifted_index, 3] = len(current_bin_nonzero_residence_times)

            twice_shifted_index = bin_index + 2*config_manager.numberOfParticleBins()
            output_data_row_names[twice_shifted_index] = "total " + bin_name + " (excluding zeros)"
            particle_summary_data[twice_shifted_index, 0] = numpy.mean(current_bin_data_for_particles_with_nonzero_residence_time)
            particle_summary_data[twice_shifted_index, 1] = numpy.min(current_bin_data_for_particles_with_nonzero_residence_time)
            particle_summary_data[twice_shifted_index, 2] = numpy.max(current_bin_data_for_particles_with_nonzero_residence_time)
            particle_summary_data[twice_shifted_index, 3] = len(current_bin_data_for_particles_with_nonzero_residence_time)


        # Do the combined systole/diastole bin info:
        systole_diastole_bin_pair_indices_mutating = copy.deepcopy(systole_diastole_bin_pair_indices)  # we'll edit this data structure as we go - defensively, let's not break the original...
        output_row_placement_index_shift = 3*config_manager.numberOfParticleBins()
        for bin_index, bin_name in enumerate(config_manager.particleBinNames()):
            once_shifted_index = bin_index + 1*config_manager.numberOfParticleBins()
            twice_shifted_index = bin_index + 2*config_manager.numberOfParticleBins()

            if bin_index in systole_diastole_bin_pair_indices_mutating:  # checks amongst key entries in the dictionary
                # Recover the original total from the previously computed mean by multiplying by the number of particles:
                number_of_particles_involved_in_bin = particle_summary_data[twice_shifted_index, 3]
                total_bin_data = particle_summary_data[twice_shifted_index, 0] * number_of_particles_involved_in_bin
                
                # do the same for the other bin (the systole bin for diastole bins; the diastole bin for systole bins):
                other_bin_index = systole_diastole_bin_pair_indices_mutating[bin_index]
                other_bin_twice_shifted_index = other_bin_index + 2*config_manager.numberOfParticleBins()
                number_of_particles_involved_in_other_bin = particle_summary_data[other_bin_twice_shifted_index, 3]
                total_other_bin_data = particle_summary_data[other_bin_twice_shifted_index, 0] * number_of_particles_involved_in_other_bin

                total_particles_in_bin = number_of_particles_involved_in_bin + number_of_particles_involved_in_other_bin

                # Total plap output for this bin pair:
                combined_bin_mean_data = old_div((total_bin_data + total_other_bin_data), total_particles_in_bin)

                particle_summary_data[output_row_placement_index_shift, 0] = combined_bin_mean_data
                particle_summary_data[output_row_placement_index_shift, 1] = min(particle_summary_data[twice_shifted_index, 1], particle_summary_data[other_bin_twice_shifted_index, 1])
                particle_summary_data[output_row_placement_index_shift, 2] = max(particle_summary_data[twice_shifted_index, 2], particle_summary_data[other_bin_twice_shifted_index, 2])
                particle_summary_data[output_row_placement_index_shift, 3] = total_particles_in_bin

                output_data_row_names[output_row_placement_index_shift] = "total {} (excluding zeros)".format(paired_bin_names[bin_index])


                output_row_for_combined_bin_residence_time = output_row_placement_index_shift + number_of_systole_diastole_bin_pairs

                total_bin_residence_time = particle_summary_data[once_shifted_index, 0] * number_of_particles_involved_in_bin
                other_bin_once_shifted_index = other_bin_index + 1*config_manager.numberOfParticleBins()
                total_other_bin_residence_time = particle_summary_data[other_bin_once_shifted_index, 0] * number_of_particles_involved_in_other_bin

                combined_bin_mean_residence_time = old_div((total_bin_residence_time + total_other_bin_residence_time), total_particles_in_bin)
                particle_summary_data[output_row_for_combined_bin_residence_time, 0] = combined_bin_mean_residence_time
                particle_summary_data[output_row_for_combined_bin_residence_time, 1] = min(particle_summary_data[once_shifted_index, 1], particle_summary_data[other_bin_once_shifted_index, 1])
                particle_summary_data[output_row_for_combined_bin_residence_time, 2] = max(particle_summary_data[once_shifted_index, 2], particle_summary_data[other_bin_once_shifted_index, 2])
                particle_summary_data[output_row_for_combined_bin_residence_time, 3] = total_particles_in_bin

                output_data_row_names[output_row_for_combined_bin_residence_time] = "residence times {} (excluding zeros)".format(paired_bin_names[bin_index])

                # Remove this bin pair so we don't do it twice:
                systole_diastole_bin_pair_indices_mutating.pop(bin_index)
                systole_diastole_bin_pair_indices_mutating.pop(other_bin_index)

                output_row_placement_index_shift += 1


        figure_output_pdf.close()

        output_data_frame = pandas.DataFrame(particle_summary_data, columns=output_data_column_names, index=output_data_row_names)

        print(output_data_frame)

        data_file_name_xlsx = "{}.xlsx".format(data_file_base_name)
        with pandas.ExcelWriter(data_file_name_xlsx) as writer:
            output_data_frame.to_excel(writer)


if __name__ == "__main__":
    config_manager_in = particles.Configuration('particle_config.json')
    generate_summaries(config_manager_in)