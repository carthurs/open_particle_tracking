from __future__ import division
#######################################
# parallel generate path lines script #
#                                     #
# command line arguments:             #
# 1 - .h5 particle file               #
# 2 - start step                      #
# 3 - stop step                       #
# 4 - step internal                   #
# 5 - number of pathlines             #
#                                     #
#######################################

from builtins import str
from builtins import range
from past.utils import old_div
import sys
from os.path import dirname, sep
from mpi4py import MPI
import h5py
import numpy as np
import vtk
from vtk.util import numpy_support as ns

# input arguments
h5Filename  = sys.argv[1]
h5Folder = dirname(h5Filename)
start = sys.argv[2]
stop = sys.argv[3]
interval = sys.argv[4]
numberOfPathLines= int(sys.argv[5])

# start MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# open h5 file
h5file = h5py.File(h5Filename, 'r')

# read step numbers
indices = h5file["Index"]

# get and sort global timesteps
timeSteps = []
for key in list(indices.keys()):
    if key != "Final":
        timeSteps.append(key)
timeSteps.sort(key=int)

# determine maximum particle index from first global timestep
index = indices[timeSteps[0]][:]
maxIndex = np.max(index)
numberOfParticleIndices = maxIndex + 1

# if not all pathlines
if numberOfPathLines!= -1:

    # get initial coordinates of all the particles
    initialCoordinates = h5file["Coordinates/"+timeSteps[0]][:]

    # generate VTK points
    initialVTKcoordinates = vtk.vtkPoints()
    initialVTKcoordinates.SetData(ns.numpy_to_vtk(initialCoordinates))

    # generate VTK Kd Tree
    kdtree = vtk.vtkKdTree()
    kdtree.SetNumberOfRegionsOrMore(numberOfPathLines)
    kdtree.BuildLocatorFromPoints(initialVTKcoordinates)

    # pick the first point of each region
    decimatedIndex = np.zeros((kdtree.GetNumberOfRegions(),), dtype=np.int64)
    for i in range(kdtree.GetNumberOfRegions()):
        pointsInRegions = ns.vtk_to_numpy(kdtree.GetPointsInRegion(i))
        decimatedIndex[i] = pointsInRegions[0]

else:
    
    # include all possible indices
    decimatedIndex = np.arange(0, numberOfParticleIndices, dtype=np.int64)

# generate iterator for all time steps
timeStepIterator = range(int(start), int(stop), int(interval))
numberOfTimeSteps = len(timeStepIterator)

# count number of pathlines handled per processor
pathlinesPerProc = int(np.floor(old_div(len(decimatedIndex),size)))

# split decimated index across processors
n1 = rank * pathlinesPerProc
n2 = (rank + 1) * pathlinesPerProc

# if on last proc use end of array
if rank == (size - 1):
    localLineIndices = decimatedIndex[n1:]
else:
    localLineIndices = decimatedIndex[n1:n2]

# generate over-sized 2d array for each particle and the total number of time steps
# array initialised with -1
particleLines = np.zeros((numberOfParticleIndices, numberOfTimeSteps), dtype=int)
particleLines[:] = -1

# generate over-sized numpy array for particle points
particlePoints = np.zeros((numberOfParticleIndices*numberOfTimeSteps,3))

# initial counter for array
previousNumberOfParticles = 0

for i, step in enumerate(timeStepIterator):

    # get indices in this time step
    index = indices[str(step)][:]

    # get boolean mask to determine which indices are on this proc
    is_dec_in_idx = np.in1d(index, localLineIndices)

    # redefine array
    index = index[is_dec_in_idx]
    
    # count particles on this proc, and update total number
    currentNumberOfParticles = len(index)
    currentNumberOfParticles += previousNumberOfParticles

    # read coordinates
    coordinates = h5file["Coordinates/" + str(step)][:]

    # set coordinates
    particlePoints[previousNumberOfParticles:currentNumberOfParticles] = coordinates[is_dec_in_idx]

    # set line indices
    particleLines[index, i] = np.arange(previousNumberOfParticles, currentNumberOfParticles)

    previousNumberOfParticles = currentNumberOfParticles


vtkParticlePoints = vtk.vtkPoints()
vtkParticlePoints.SetData(ns.numpy_to_vtk(particlePoints[:currentNumberOfParticles]))

# create vtk vtkPolyData
vtkPolyData = vtk.vtkPolyData()
vtkPolyData.SetPoints(vtkParticlePoints)

# loop over each particle path line and add to vtk cell
vtkParticleCell = vtk.vtkCellArray()

for i in range(numberOfParticleIndices):

    # get the local indices of this line and remove any -1 values
    lineIndex = particleLines[i][:]
    lineIndex = lineIndex[lineIndex != -1]

    # count number of points in this line
    numberOfLinePoints = len(lineIndex)

    # if the number of points are > 1
    if numberOfLinePoints > 1:

        # generate line
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(numberOfLinePoints)

        for j in range(0, numberOfLinePoints):
            line.GetPointIds().SetId(j, lineIndex[j])  # 2nd argument is the id of the vtkPoint data

        vtkParticleCell.InsertNextCell(line)

# add lines to vtkPolyData
vtkPolyData.SetLines(vtkParticleCell)

# write out cleaned data
cleaner = vtk.vtkCleanPolyData()
if vtk.VTK_MAJOR_VERSION == 6:
    cleaner.SetInputData(vtkPolyData)
else:
    cleaner.SetInput(vtkPolyData)
cleaner.Update()

fileString = str(numberOfPathLines)+'-'+start+'-'+stop+'-'+interval

# if serial write out serial file
if size == 1:

    serialWriter = vtk.vtkXMLPolyDataWriter()
    serialWriter.SetFileName('pathlines-serial-'+fileString+'.vtp')
    if vtk.VTK_MAJOR_VERSION == 6:
        serialWriter.SetInputData(cleaner.GetOutput())
    else:
        serialWriter.SetInput(cleaner.GetOutput())
    serialWriter.Write()

# else write out parallel files
else:

    # write out local file, note that name HAS to be with the underscore
    localWriter = vtk.vtkXMLPolyDataWriter()
    localWriter.SetFileName('pathlines-'+fileString+'_'+str(rank)+'.vtp')
    if vtk.VTK_MAJOR_VERSION == 6:
        localWriter.SetInputData(vtkPolyData)
    else:
        localWriter.SetInput(vtkPolyData)
    localWriter.Write()

    # write out the master .pvtp file
    if rank == 0:
        parallelWriter = vtk.vtkXMLPPolyDataWriter()
        parallelWriter.SetFileName('pathlines-'+fileString+'.pvtp')
        parallelWriter.SetNumberOfPieces(size)
        if vtk.VTK_MAJOR_VERSION == 6:
            parallelWriter.SetInputData(vtkPolyData)
        else:
            parallelWriter.SetInput(vtkPolyData)
        parallelWriter.Write()


