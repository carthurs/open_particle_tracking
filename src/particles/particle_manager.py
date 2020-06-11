from builtins import range
from builtins import object
import numpy as np
import vtk
from vtk.util import numpy_support
import tqdm
import warnings

class ParticleManager(object):
    '''
    One day, this class should actually manage all the particles. But
    refactoring all the particle management tasks here will be time-consuming.
    For now, I'm creating it as a place where all /new/ particle management
    code can go (plus some minor easy refactors of existing particle code
    which I've moved in here).
    '''
    def __init__(self, comm, particle_vtu_file):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.numberOfProcessors = self.comm.Get_size()
        '''
        ***************** PARTICLES INJECTION PREPARATION ********************
        '''
        self.domain_particles = vtk.vtkXMLUnstructuredGridReader()
        self.domain_particles.SetFileName(particle_vtu_file)
        self.domain_particles.Update()

        '''
        We push the domain_particles through an append_filter (as the only
        input) This will become useful later when we start reinjecting
        particles - the new particles will be added as inputs to the
        vtkAppendFilter.
        '''
        original_particle_mesh_data = self.domain_particles.GetOutput()
        self.append_filter = vtk.vtkAppendFilter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            self.append_filter.AddInput(original_particle_mesh_data)
        else:
            self.append_filter.AddInputData(original_particle_mesh_data)
        self.append_filter.Update()

        self.alive_particles_coordinates = np.copy(
            numpy_support.vtk_to_numpy(
                self.append_filter.GetOutput().GetPoints().GetData()
            )
        )
        # self.particles_vtk_pts = vtk.vtkPoints().SetData(
        # numpy_support.numpy_to_vtk(self.alive_particles_coordinates))
        self.nparticles = self.append_filter.GetOutput().GetNumberOfPoints()

        # initially particles are 'tidy'
        self.particles_indices = np.arange(self.nparticles, dtype=np.int64)
        self.particles_offsets = np.zeros((self.numberOfProcessors + 1,),
                                          dtype=np.int64)

    def reinjectParticles(self):
        original_particle_mesh_data = self.domain_particles.GetOutput()
        if vtk.VTK_MAJOR_VERSION <= 5:
            self.append_filter.AddInput(original_particle_mesh_data)
        else:
            self.append_filter.AddInputData(original_particle_mesh_data)

        self.append_filter.Update()

        nparticles_previous = self.nparticles
        self.nparticles = self.append_filter.GetOutput().GetNumberOfPoints()
        self.number_of_newly_injected_particles = (self.nparticles -
                                                   nparticles_previous)

        newly_injected_particle_indices = np.arange(nparticles_previous,
                                                    self.nparticles,
                                                    dtype=np.int64)

        # Get the particle coordinates of the newly injected particles, but
        # we are careful to avoid overwriting the locations of the
        # already-advected particles:
        newly_injected_particles_coordinates = np.copy(
            numpy_support.vtk_to_numpy(
                self.append_filter.GetOutput().GetPoints().GetData()
            )[newly_injected_particle_indices]
        )

        self.alive_particles_coordinates = np.append(
            self.alive_particles_coordinates,
            newly_injected_particles_coordinates,
            axis=0
        )
        self.particles_indices = np.append(self.particles_indices,
                                           newly_injected_particle_indices)

        self.repartition()

    def getNumberOfNewlyInjectedParticles(self):
        return self.number_of_newly_injected_particles

    def repartition(self):
        if len(self.alive_particles_coordinates) == 0:
            warnings.warn("no particles to partition")
        self.particles_partition = np.zeros(
            (len(self.alive_particles_coordinates),),
            dtype=np.int64
        )
        if self.rank == 0:
            tqdm.tqdm.write("Repartitioning...")
            kdtree = self.__getKdTree()
            offset = 0
            for processor_index in range(self.numberOfProcessors):
                if processor_index < kdtree.GetNumberOfRegions():
                    points_in_regions = numpy_support.vtk_to_numpy(
                        kdtree.GetPointsInRegion(processor_index)
                    )
                    offset += points_in_regions.shape[0]
                    self.particles_partition[points_in_regions] = processor_index
                self.__setParticleOffsetSliceEndpointForProcessor(offset, processor_index)

        self.__broadcastData()

    def getLocalArrayValues(self, global_array):
        return global_array[np.where(self.particles_partition == self.rank)]

    def getLocalParticleIndices(self):
        return np.where(self.particles_partition == self.rank)

    def __getKdTree(self):
        particles_vtk_pts = vtk.vtkPoints()
        particles_vtk_pts.SetData(
            numpy_support.numpy_to_vtk(self.alive_particles_coordinates)
        )
        kdtree = vtk.vtkKdTree()
        kdtree.SetNumberOfRegionsOrLess(self.numberOfProcessors)
        kdtree.BuildLocatorFromPoints(particles_vtk_pts)
        return kdtree

    def setLocalAliveParticleBooleanMask(self, alive_mask):
        self.globalAliveParticleBooleanMask = alive_mask

    # def updateGlobalAliveParticleBooleanMask(self, geometry_mesh):
    #     probe_filter = vtk.vtkProbeFilter()
    #     probe_filter.SetValidPointMaskArrayName("alive")
    #     probe_filter.SetInput(global_particles)
    #     probe_filter.SetSource(geometry_mesh)
    #     probe_filter.Update()
    #     self.globalAliveParticleBooleanMask = numpy_support.vtk_to_numpy(
    #        probe_filter.GetOutput().GetPointData().GetArray("alive"))

    def getAliveParticleMask(self):
        return self.globalAliveParticleBooleanMask

    def getParticleProbe(self, domain, local_particles):
        probe = vtk.vtkProbeFilter()
        probe.SetValidPointMaskArrayName("alive")
        if vtk.VTK_MAJOR_VERSION <= 5:
            probe.SetInput(local_particles)
            probe.SetSource(domain)
        else:
            probe.SetInputData(local_particles)
            probe.SetSourceData(domain)
        probe.Update()
        return probe

    def getFromProbe(self, field_name, probe):
        return numpy_support.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(field_name))

    def maskArrayByParticleLocalAlivenessAndGetLocalValues(self, numpy_array):
        data_array_alive_only = numpy_array[
            self.globalAliveParticleBooleanMask
        ]
        data_array = self.getLocalArrayValues(data_array_alive_only)
        return data_array

    def getLocalAliveParticles(self):
        return self.getLocalArrayValues(self.globalAliveParticleBooleanMask)

    def addParticlePointDataArray(self, array):
        self.append_filter.GetOutput().GetPointData().AddArray(array)

    def getParticleCoordinatesSlice(self, slice):
        return self.alive_particles_coordinates[slice]

    def getLocalParticleCoordinates(self):
        return self.getLocalArrayValues(self.alive_particles_coordinates)

    def getLocalParticleCells(self, nlocalparticles):
        local_particles_topology = np.zeros((nlocalparticles, 2), dtype=np.int64)
        local_particles_topology[:, 0] = 1
        local_particles_topology[:, 1] = np.arange(nlocalparticles, dtype=np.int64)
        local_particles_topology = np.reshape(local_particles_topology, (nlocalparticles * 2, 1))

        local_particles_cells = vtk.vtkCellArray()
        local_particles_cells.SetCells(nlocalparticles,
                                       numpy_support.numpy_to_vtkIdTypeArray(local_particles_topology))

        return local_particles_cells

    def computeAndGetLocalParticlesVtk(self, local_particles_coordinates, nlocalparticles):
        local_particles_cells = self.getLocalParticleCells(nlocalparticles)
        local_particles_vtk = vtk.vtkPolyData()
        local_particles_pts = vtk.vtkPoints()
        local_particles_pts.SetData(numpy_support.numpy_to_vtk(local_particles_coordinates))
        local_particles_vtk.SetPoints(local_particles_pts)
        local_particles_vtk.SetVerts(local_particles_cells)

        return local_particles_vtk

    def setParticleCoordinates(self, coordinates):
        self.alive_particles_coordinates = coordinates

    def getParticleDataOutputPort(self):
        return self.append_filter.GetOutputPort()

    def getNumberOfParticles(self):
        return self.nparticles

    def getNumberOfGlobalLiveParticles(self):
        return self.particles_indices.shape[0]

    def getParticleOffsetSlice(self, rank):
        return slice(self.particles_offsets[rank],
                     self.particles_offsets[rank + 1])

    def getParticleOffsetStart(self, rank):
        return self.particles_offsets[rank]

    def getParticleOffsetEnd(self, rank):
        return self.particles_offsets[rank + 1]

    def __setParticleOffsetSliceEndpointForProcessor(self, offsets, rank):
        self.particles_offsets[rank + 1] = offsets

    def __broadcastData(self):
        # print ("rank", str(self.comm.Get_rank()),
        #        "broadcasting",
        #        np.shape(self.particles_offsets))
        self.comm.Barrier()
        self.comm.Bcast(self.particles_offsets, root=0)
        self.comm.Barrier()
        self.comm.Bcast(self.particles_partition, root=0)

    def getParticleLocalToGlobalMap(self):
        return self.particles_indices[
            np.where(self.particles_partition == self.rank)
        ]

    def setParticleIndices(self, indices):
        # pass
        self.particles_indices = indices
