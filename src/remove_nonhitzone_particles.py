from __future__ import print_function
from builtins import str
from builtins import object
import h5py
import numpy
import mpi4py
import mpi4py.MPI

# This code is very fragile. I really don't recommend
# modifying it - a complete re-write would be a much
# better idea.
#
# Refactoring it is on my todo list, but if you're
# seeing this message then it hasn't been done!
# (although you might check the bitbucket in case
# there's a newer version up there).

# It should do the job of removing particles that
# do not have the global indices provided by the
# user from the hdf5 file, however.data_shape



class Hdf5ArrayMetadata(object):
    def __init__(self, data_path, data_shape, data_type):
        self.data_path = data_path
        self.data_shape = data_shape
        if data_type == 'float32':
            self.data_type = 'f'
        elif data_type == 'int64':
            self.data_type = numpy.int64
        else:
            raise RuntimeError("Unknown metadata type.")


def debug_print(rank, comm, counter, tag):
    print('rank:', rank, 'count:', counter, tag)
    # comm.Barrier()
    counter += 1
    return counter


def remove_nonhitzone_particles(particle_indices_to_keep, mpi_communicator,
                                input_filename, output_filename):

    mpi_rank = mpi_communicator.Get_rank()
    mpi_number_of_processors = mpi_communicator.Get_size()

    particle_indices_to_keep_set = set(particle_indices_to_keep) #.intersection(set(range(0, 1763)))

    # Prepare a copy of the output file, with empty space where the to-be-edited
    # particle arrays will go:
    # output_filename = 'volunteer3D-particles-4000-copy-out.h5'
    # output_file = h5py.File(output_filename, 'w', driver='mpio', comm=mpi_communicator)

    filename = input_filename

    global_time_steps_to_process = numpy.array([])
    # file = h5py.File(filename, 'r', driver='mpio', comm=mpi_communicator)
    with h5py.File(filename, 'r', driver='mpio', comm=mpi_communicator) as file:
        with h5py.File(output_filename, 'w', driver='mpio', comm=mpi_communicator) as output_file:
            # -1 because there's all the timesteps plus a "Final" array;
            # we don't want to edit "Final".
            for idx, particle_index_array in enumerate(file.get('Index')):
                if idx > -1:
                    if particle_index_array != 'Final':
                        global_time_steps_to_process = numpy.append(global_time_steps_to_process, int(particle_index_array))
            # print global_time_steps_to_process

            # Partition the time-steps across the processors data:
            time_step_partition = numpy.array_split(global_time_steps_to_process, mpi_number_of_processors)
            this_processors_timesteps = time_step_partition[mpi_rank]

            # print "this_processors_timesteps:", this_processors_timesteps
            file_internal_indices_of_particles_to_keep_by_timestep = dict()
            for particle_index_array in this_processors_timesteps:
                particle_index_array_str = str(int(particle_index_array))

                file_data_for_this_step = file['Index' + '/' + particle_index_array_str][:]

                print("Gathering data on timestep", particle_index_array_str, "length is", len(file_data_for_this_step))
                file_internal_indices_of_particles_to_keep_by_timestep[particle_index_array_str] = []
                # print file.get('Index' + '/' + particle_index_array_str).value
                for array_index, invariant_particle_index in enumerate(file_data_for_this_step):
                    if invariant_particle_index in particle_indices_to_keep_set:
                        file_internal_indices_of_particles_to_keep_by_timestep[particle_index_array_str].append(array_index)
                print("Length was", len(file_internal_indices_of_particles_to_keep_by_timestep[particle_index_array_str]), "vs original", len(particle_indices_to_keep_set), "and", len(file_data_for_this_step))

            timestep_data_slice_sizes = dict()
            replacement_hdf5_data_arrays = dict()
            for root_entry in file:
                if root_entry != 'TimeWritten':
                    # for particle_index_array in file.get(root_entry):
                    #     if particle_index_array != 'Final':
                    for particle_index_array in this_processors_timesteps:
                        internal_path_to_data = root_entry + '/' + str(int(particle_index_array))
                        # print "Removing non-hitzone particle data from array", internal_path_to_data
                        try:
                            raw_numpy_array = file.get(internal_path_to_data).value
                            if root_entry != 'Topology':
                                # print file_internal_indices_of_particles_to_keep_by_timestep
                                val = file_internal_indices_of_particles_to_keep_by_timestep[str(int(particle_index_array))]
                                newdata = raw_numpy_array[val]
                            else:
                                number_of_particles_this_step = len(file_internal_indices_of_particles_to_keep_by_timestep[str(int(particle_index_array))])
                                newdata = numpy.arange(number_of_particles_this_step)
                                # print '==============>', newdata
                            replacement_hdf5_data_arrays[internal_path_to_data] = newdata
                            # del file[internal_path_to_data]
                            # output_file.create_dataset(internal_path_to_data, data=newdata)
                            # print internal_path_to_data, numpy.shape(newdata)
                            timestep_data_slice_sizes[particle_index_array] = numpy.shape(newdata)[0]
                        except AttributeError:
                            print("==========> 	Failure modifying array", internal_path_to_data)

            # Actually write the modified particle data arrays to the hdf5 file.
            # We do this separately as the calls to del and create_dataset
            # must be done collectively.

            array_metadata_by_timestep = []

            for root_entry in file:
                if root_entry != 'TimeWritten':
                    # for particle_index_array in file.get(root_entry):
                    #     if particle_index_array != 'Final':
                    for particle_index_array in this_processors_timesteps:
                        internal_path_to_data = root_entry + '/' + str(int(particle_index_array))
                        data_shape = numpy.shape(replacement_hdf5_data_arrays[internal_path_to_data])
                        data_type = file[internal_path_to_data].dtype

                        array_metadata = Hdf5ArrayMetadata(internal_path_to_data, data_shape, data_type)

                        array_metadata_by_timestep.append(array_metadata)

            gathered_metadata_uncombined_lists = mpi_communicator.allgather(array_metadata_by_timestep)
            gathered_array_metadata_by_timestep = []
            for gathered_list in gathered_metadata_uncombined_lists:
                gathered_array_metadata_by_timestep = gathered_array_metadata_by_timestep + gathered_list

            mpi_communicator.Barrier()

            # for array_metadatum in gathered_array_metadata_by_timestep:
                # internal_path_to_data = array_metadatum.data_path
                # mpi_communicator.Barrier()
                # del file[internal_path_to_data]
                # mpi_communicator.Barrier()
            # print "pre close", mpi_rank
            # file.flush()
            # gc.collect()
            # file.close()
            # print "closed file rank ", mpi_rank
            # mpi_communicator.Barrier()
            # file = h5py.File(filename, 'r+', driver='mpio', comm=mpi_communicator)
            # print "opened file rank ", mpi_rank

            for array_metadatum in gathered_array_metadata_by_timestep:
                internal_path_to_data = array_metadatum.data_path
                mpi_communicator.Barrier()
                # print "Creating dataset", internal_path_to_data, "of type", array_metadatum.data_type, "and shape", array_metadatum.data_shape
                # file.create_group(internal_path_to_data.split('/')[0])
                # print "done", mpi_rank, internal_path_to_data.split('/')[0]

                group_root_name = internal_path_to_data.split('/')[0]
                dataset_name = internal_path_to_data.split('/')[1]

                # print "dataset_name", dataset_name, "group_root_name", group_root_name

                # print "rank", mpi_rank, "done 1"

                # if max(array_metadatum.data_shape) > 0:
                group = output_file.require_group(group_root_name)
                group.create_dataset(dataset_name, shape=array_metadatum.data_shape, dtype=array_metadatum.data_type)
                # group.create_dataset(dataset_name, shape=array_metadatum.data_shape, dtype=array_metadatum.data_type, data=replacement_hdf5_data_arrays[internal_path_to_data])
                # print "rank", mpi_rank, "done 2"
                # file.create_dataset('bean', shape=(4, 3), dtype='f')
                mpi_communicator.Barrier()
                # print "rank", mpi_rank, "done 3"
            mpi_communicator.Barrier()
            # print "rank", mpi_rank, "done 4"

            for root_entry in file:
                if root_entry != 'TimeWritten':
                    # for particle_index_array in file.get(root_entry):
                    #     if particle_index_array != 'Final':
                    for particle_index_array in this_processors_timesteps:
                        internal_path_to_data = root_entry + '/' + str(int(particle_index_array))
                        # print "------------++++++===>>>", internal_path_to_data, numpy.shape(replacement_hdf5_data_arrays[internal_path_to_data])
                        # type(file[internal_path_to_data])
                        output_file[internal_path_to_data][...] = replacement_hdf5_data_arrays[internal_path_to_data]

            mpi_communicator.Barrier()
        print(mpi_rank, "output file closed.")
    print(mpi_rank, "input file closed.")
    # gc.collect()
    # file.flush()
    # file.close()

    gathered_timestep_data_slice_sizes = mpi_communicator.allgather(timestep_data_slice_sizes)
    gathered_joined_array_metadata_by_timestep = dict()
    for gathered_dict in gathered_timestep_data_slice_sizes:
        gathered_joined_array_metadata_by_timestep.update(gathered_dict)

    mpi_communicator.Barrier()

    # output_file.flush()
    # output_file.close()
    # Write XDMF File for Visualization in Paraview
    if mpi_rank == 0:
        filename2 = 'volunteer3D-particles-4000-copy.xdmf'
        xdmf_out = open(filename2, 'w')
        xdmf_out.write("""<?xml version="1.0"?>
      <Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
      <Domain>
        <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
          <Time TimeType="List">\n""")
        timesteps_str = ' '.join(str(int(i)) for i in sorted(gathered_joined_array_metadata_by_timestep.keys()))
        nsteps = len(global_time_steps_to_process)
        xdmf_out.write('<DataItem Format="XML" Dimensions="%d">%s</DataItem>\n</Time>' %(nsteps,timesteps_str) )
        # For each timestep point to grid topology and geometry, and attributes
        # nliveparticles = particle_manager.getNumberOfGlobalLiveParticles() # todo confirm that this line is redundant & remove it
        # for i, nliveparticles in zip(saved_steps, saved_nliveparticles):
        for val in sorted(gathered_joined_array_metadata_by_timestep.keys()):
            i = int(val)
            print(mpi_rank, "Writing", i)
            nliveparticles = int(gathered_joined_array_metadata_by_timestep[val])
            xdmf_out.flush()
            xdmf_out.write('<Grid Name="grid_%d" GridType="Uniform">\n' % i)
            xdmf_out.write('<Topology NumberOfElements="%d" TopologyType="Polyvertex">\n' % nliveparticles)
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/Topology/%d</DataItem>\n'
                           % (nliveparticles, output_filename, i))
            xdmf_out.write('</Topology>\n<Geometry GeometryType="XYZ">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 3">%s:/Coordinates/%d</DataItem>\n'
                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Geometry>\n')
        
            xdmf_out.write('<Attribute Name="PLAP" AttributeType="Scalar" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/PLAP/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')
            xdmf_out.write('<Attribute Name="PRT" AttributeType="Scalar" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/PRT/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')
            xdmf_out.write('<Attribute Name="alive" AttributeType="Scalar" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/alive/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')
            xdmf_out.write('<Attribute Name="partition" AttributeType="Scalar" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/partition/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')
            xdmf_out.write('<Attribute Name="velocity" AttributeType="Vector" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 3">%s:/velocity/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')
            xdmf_out.write('<Attribute Name="vorticity" AttributeType="Vector" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 3">%s:/vorticity/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')
            xdmf_out.write('<Attribute Name="helical_flow_index" AttributeType="Scalar" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/helical_flow_index/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')
            xdmf_out.write('<Attribute Name="absolute_helical_flow_index" AttributeType="Scalar" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/absolute_helical_flow_index/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')

            xdmf_out.write('<Attribute Name="absolute_helical_flow_index_systole" AttributeType="Scalar" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/absolute_helical_flow_index_systole/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')

            xdmf_out.write('<Attribute Name="absolute_helical_flow_index_diastole" AttributeType="Scalar" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/absolute_helical_flow_index_diastole/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')

            xdmf_out.write('<Attribute Name="Invariant Particle Index" AttributeType="Integer" Center="Cell">\n')
            xdmf_out.write('<DataItem Format="HDF" Dimensions="%d 1">%s:/Index/%d</DataItem>\n'
                                       % (nliveparticles, output_filename, i))
            xdmf_out.write('</Attribute>\n')

            xdmf_out.write('</Grid>\n')
            xdmf_out.flush()
        xdmf_out.write('</Grid>\n</Domain>\n</Xdmf>')
        xdmf_out.flush()
        xdmf_out.close()
        print("rank", mpi_rank, "reached the end")
        return

    # origdata = file.get('Coordinates/4000').value

    # newdata = numpy.delete(origdata, (0), axis=0)

    # del file['Coordinates/4000']
    # file.create_dataset('Coordinates/4000', data=newdata)

    # print origdata[0:5, :]
    # print file.get('Coordinates/4000').value[0:5, :]


if __name__ == '__main__':
    input_filename = 'volunteer3D-particles-4000-copy.h5'
    output_filename = 'volunteer3D-particles-4000-copy-out.h5'

    mpi_communicator = mpi4py.MPI.COMM_WORLD
    remove_nonhitzone_particles(
        [0, 1, 3, 51122, 7] + list(numpy.arange(200, 400)),
        mpi_communicator,
        input_filename,
        output_filename
        )
    print("rank", mpi_communicator.Get_rank(), "reached the end")
    mpi_communicator.Barrier()
    print("rank", mpi_communicator.Get_rank(), "passed the end barrier")
    mpi4py.MPI.Finalize()
