'''
Used to specify the time-bins within which particle
data is reported. A simple example would be to
define one for systole and one for diastole.
'''


from builtins import str
from builtins import object
class ParticleDataTimeBinsSpecifier(object):
    def __init__(self):
        self.timeBins = dict()
        self.binSpatialLimits = dict()
        self.numberOfTimeBins = 0

    def names(self):
        for name in self.timeBins:
            yield name

    def startAndEndTimes(self):
        for name in self.timeBins:
            yield self.timeBins[name]

    def getTimeBin(self, binName):
        return self.timeBins[binName]

    def getBinSpatialLimits(self, binName):
        try:
            return self.binSpatialLimits[binName]
        except KeyError:
            raise RuntimeError("Bin " + binName + " does not have a mesh specifying its spatial limits.")

    def getNumberOfSpatialRegionsComprisingBin(self, binName):
        try:
            return len(self.binSpatialLimits[binName])
        except KeyError:
            raise RuntimeError("Bin " + binName + " does not have a mesh specifying its spatial limits.")


    def addTimeBinInterval(self, startTimeInSeconds,
                           endTimeInSeconds, binName=None):

        try:
            self.__addIntervalToExistingTimeBin(startTimeInSeconds,
                                         endTimeInSeconds,
                                         binName)
        except KeyError:
            # Unless the binName was given, generate a unique binName for it
            if not binName:
                binName = ("time_bin_" + str(self.numberOfTimeBins) +
                           "_" + str(startTimeInSeconds) + "_" +
                           str(endTimeInSeconds))

            # timeBins contains a list of pairs, not just pairs, so a single bin
            # can be constructed from disjoint time intervals
            self.timeBins[binName] = [(startTimeInSeconds, endTimeInSeconds)]

        self.numberOfTimeBins += 1

    '''
    Additionally limit the bin with name binName in space, as well
    as in time, by specifying a vtk mesh whose boundary gives the
    spatial limits of the bin.
    '''
    def addSpatialRegionToBin(self, binBoundaryVtkMesh, binName):
        try:
            self.binSpatialLimits[binName].append(binBoundaryVtkMesh)
        except KeyError:
            self.binSpatialLimits[binName] = [binBoundaryVtkMesh]

    def binHasSpatialLimits(self, binName):
        if binName in self.binSpatialLimits:
            return True
        else:
            return False

    def __addIntervalToExistingTimeBin(self, startTimeInSeconds,
                                     endTimeInSeconds, binName):
        if binName is not None:
            self.timeBins[binName].append((startTimeInSeconds, endTimeInSeconds))
        else:
            raise ValueError("A binName must be provided.")