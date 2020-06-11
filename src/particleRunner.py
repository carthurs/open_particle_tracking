"""
File for running particle tracking with bins.

Assumes that all your bins are in folders in the current working dir
with names {bin-name}/presolver/the.coordinates etc.

Particles presolver files shoudl be in particle/presolver/the.xxx

Set up your particle_config.json completely before you run this, and all the bins
will be found automatically, so long as the bin file name roots (i.e.
the part before the .vtu) match the folder names {bin-name}

In theory, this should all be automatic.

You need to have the vis files ready to go in the current dir too.
"""

import sys
sys.path.append('/home/chris/workspace/simvascular_flowsolver_estimator/scripts/')
import GeombcReader
import mesh2vtk
import vis2hdf5_nosurface
import particles
import mpi4py, mpi4py.MPI
import crimsonParticleTracking
import numpy
import summariseParticleOutput

if __name__ == "__main__":

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()

    config_manager = particles.Configuration('particle_config.json')
    vis_file_basename = config_manager.fname()
    start_step = config_manager.cycle_start()
    end_step = config_manager.cycle_stop()
    step_between_restarts = config_manager.cycle_step()

    if rank == 0:
        pathToGeombcInputFile = r'geombc.dat.1'

        GeombcReader.extractArrayAndWriteToDisk(pathToGeombcInputFile, "extracted.coordinates", "co-ordinates", numpy.float64)
        GeombcReader.extractArrayAndWriteToDisk(pathToGeombcInputFile, "extracted.connectivity", "connectivity interior linear tetrahedron", numpy.int32)
        mesh_manually_extracted_true = True 
        mesh2vtk.mesh2vtk('extracted', 'extracted', config_manager, mesh_manually_extracted_true)

        mesh_manually_extracted_false = False
        mesh2vtk.mesh2vtk('particle/presolver/the', 'particles', config_manager, mesh_manually_extracted_false)
        # make the bin meshes into a vtu: (from the presolver folder, the file is called "the")
        for bin_name in config_manager.particleBinNames():
            for bin_file_name in config_manager.particleBinMeshFileNames(bin_name):
                if bin_file_name is not None:
                    bin_file_name_base = bin_file_name.split('.')[0]
                    input_file_name = '{}/presolver/the'.format(bin_file_name_base)
                    mesh2vtk.mesh2vtk(input_file_name, bin_file_name_base, config_manager, mesh_manually_extracted_false)

    comm.Barrier()
    vis2hdf5_nosurface.vis2hdf5_nosurface(vis_file_basename, start_step, end_step, step_between_restarts, 'extracted.vtu')

    # crimsonParticleTracking.runTrackingWithConfiguration(config_manager)

    # if rank == 0:
    #     summariseParticleOutput.generate_summaries(config_manager)
