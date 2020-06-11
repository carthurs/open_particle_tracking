from __future__ import division
from __future__ import print_function
from builtins import object
from past.utils import old_div
import numpy as np
import time

class CardiacCycleTimekeeper(object):
    '''
    Tracks the current time in the cardiac cycle, accounting for periodicity

    :param cycle_start: the time at which the cardiac cycle starts
    :type cycle_start: float

    :param cycle_stop: the time at which the cardiac cycle ends
    :type cycle_stop: float

    :param particleDataSpacetimeBinsSpecifiers: labels for different subintervals of the
     cardiac cycle which are of separate individual interest
    :type particleDataSpacetimeBinsSpecifiers: ParticleDataTimeBinsSpecifier
    '''
    def __init__(self, cycle_start, cycle_stop, cycle_step, ncycles,
                 startTime, timestepSize,
                 repartition_frequency_in_original_timesteps,
                 firstSystoleStartTime,
                 firstSystoleEndTime, cardiacCycleLength,
                 particleDataSpacetimeBinsSpecifiers, rank,
                 config_manager,
                 tracking_step_start):

        self.config_manager = config_manager
        self.cycle_start = cycle_start
        self.cycle_stop = cycle_stop
        self.cycle_step = cycle_step
        self.ncycles = ncycles

        self.tracking_step_start = tracking_step_start
        self.t_step = tracking_step_start

        self.currentStepIndex = 0
        self.numberOfCompletedReinjections = 0

        self.steps_between_reinjections = self.config_manager.stepsBetweenReinjections()
        self.minimum_steps_before_first_reinjection = self.config_manager.minimumStepsBeforeFirstReinjection()
        self.maxReinjections = self.config_manager.maxReinjections()

        self.repartition_frequency_in_original_timesteps = repartition_frequency_in_original_timesteps

        self.particleDataSpacetimeBinsSpecifiers = particleDataSpacetimeBinsSpecifiers

        self.rank = rank

        self.stepTimerStartTime = time.time()

        remainder = (self.cycle_stop - self.cycle_start) % self.cycle_step
        if (remainder != 0):
            raise RuntimeError('Error', 'Cycle_step does not exactly divide '
                               'the interval [cycle_start, cycle_stop].')

        self.total_simulation_steps = ((old_div((self.cycle_stop - self.cycle_start),
                                       self.cycle_step) + 1) * self.ncycles)

        print("simulation will run for {} steps. Now on step {}.".format(self.total_simulation_steps, self.currentStepIndex))

        self.startTime = startTime
        self.currentTime = self.startTime
        self.timestepSize = timestepSize
        self.firstSystoleStartTime = firstSystoleStartTime
        self.firstSystoleEndTime = firstSystoleEndTime
        self.cardiacCycleLength = cardiacCycleLength

        self.singleCycleSystoleStartTime = (self.firstSystoleStartTime -
                                            self.startTime)
        self.singleCycleSystoleEndTime = (self.firstSystoleEndTime -
                                          self.startTime)

        if (self.firstSystoleStartTime > self.firstSystoleStartTime):
            raise RuntimeError('Error', 'Systole start time was after'
                               ' systole end time.')
        if ((self.firstSystoleStartTime - self.startTime) >
           self.cardiacCycleLength):
            raise RuntimeError('Error', 'Systole start time was after'
                               ' the end of the specified cardiac cycle.')
        if ((self.firstSystoleEndTime - self.startTime) >
           self.cardiacCycleLength):
            raise RuntimeError('Error', 'Diastole start time was after'
                               ' the end of the specified cardiac cycle.')

    def step(self):
        '''Updates the current time by one time-step'''
        if self.rank == 0:
            print("Stepping from time", self.currentTime, "took:", time.time() - self.stepTimerStartTime)
        self.currentTime += self.timestepSize
        self.stepTimerStartTime = time.time()
        self.currentStepIndex += 1
        self.t_step += self.cycle_step

    def getTStep(self):
        return self.t_step

    def isFinalTimestep(self):
        print("total_simulation_steps", self.total_simulation_steps, "currentStepIndex", self.currentStepIndex)
        isFinalStep = (self.total_simulation_steps == 
                       self.currentStepIndex + 1)
        return isFinalStep

    def getTimestepSize(self):
        '''Returns the time-step'''
        return self.timestepSize

    def getCurrentTimeIntervalLabels(self):
        '''
        Returns the labels which apply to the current point in the cardiac
        cycle.

        The user may have provided any number of labels for various
        sub-intervals of the cardiac cycle. Any one time instant may have
        one or more labels.

        :return: A list of strings naming the intervals to which the current
         time point belongs.
        '''
        pass

    def isSystole(self):
        return self.__isSystoleInternal(self.currentTime)

    def __isSystoleInternal(self, at_time):
        singleCycleTime = (at_time - self.startTime) % self.cardiacCycleLength
        if (self.singleCycleSystoleStartTime <= singleCycleTime and
           singleCycleTime <= self.singleCycleSystoleEndTime):
            return True
        else:
            return False

    # Takes a single particle's residence time
    # Returns:
    # dictonary with bin names as keys, and residence times
    # in each bin as the mapped values.
    def getBinResidenceTimesForExitingParticle(self,
                                               particle_residence_time):
        '''
        This function depends critically on the state of this class.
        It requires knowledge of the time-period during which the particle is
        resident in the domain; the duration is provided by the caller, but the
        exit time is taken from the CardiacCycleTimekeeper's state.
        In short: this function only works if it is called at the time the
        particle exits the domain.
        '''
        particle_entrance_time = self.currentTime - particle_residence_time
        bin_residence_times = dict()
        for bin_name in self.particleDataSpacetimeBinsSpecifiers.names():
            bin_residence_times[bin_name] = 0.0
            for timepoint in np.arange(particle_entrance_time,
                                   self.currentTime,
                                   self.timestepSize):
                if self.__timeLiesInBin(timepoint, bin_name):
                    bin_residence_times[bin_name] += self.timestepSize

        return bin_residence_times


    def currentTimeLiesInBin(self, bin_name):
        return self.__timeLiesInBin(self.currentTime, bin_name)


    def __timeLiesInBin(self, timepoint, bin_name):

        # This is a list of 2-tuples
        time_bin_start_and_end_times = self.particleDataSpacetimeBinsSpecifiers.getTimeBin(bin_name)
        
        singleCycleTime = (timepoint - self.startTime) % self.cardiacCycleLength

        for bin_subinterval in time_bin_start_and_end_times:
            if (bin_subinterval[0] <= singleCycleTime and
                singleCycleTime <= bin_subinterval[1]):
                return True

        # If we get here then singleCycleTime did not lie in any of the
        # sub-bin intervals for this time bin:
        return False


    TAG_SYSTOLE = "systole"
    TAG_DIASTOLE = "diastole"

    def getSystolicAndDiastolicResidenceTimesForExitingParticle(self,
                                                    particle_residence_time):
        '''
        This function depends critically on the state of this class.
        It requires knowledge of the time-period during which the particle is
        resident in the domain; the duration is provided by the caller, but the
        exit time is taken from the CardiacCycleTimekeeper's state.
        In short: this function only works if it is called at the time the
        particle exits the domain.
        '''
        particle_entrance_time = self.currentTime - particle_residence_time
        systolic_residence_time = 0
        for timepoint in np.arange(particle_entrance_time,
                                   self.currentTime,
                                   self.timestepSize):
            if self.__isSystoleInternal(timepoint):
                systolic_residence_time += self.timestepSize
        diastolic_residence_time = (particle_residence_time -
                                    systolic_residence_time)

        residence_times = {CardiacCycleTimekeeper.TAG_SYSTOLE:
                           systolic_residence_time,
                           CardiacCycleTimekeeper.TAG_DIASTOLE:
                           diastolic_residence_time}
        return residence_times

    def getTotalSimulationSteps(self):
        return self.total_simulation_steps

    def getTime(self):
        return self.currentTime

    def repartitionThisStep(self):
        repartitionStep = ((self.t_step - self.tracking_step_start) % self.repartition_frequency_in_original_timesteps == 0)

        if repartitionStep and not self.isFinalTimestep():
            return True
        else:
            return False

    def reinjectThisStep(self):
        passedFirstInjectionStep = (self.currentStepIndex >=
                                    self.minimum_steps_before_first_reinjection)

        isReinjectionStep = (self.numberOfCompletedReinjections <
                              (self.currentStepIndex - self.minimum_steps_before_first_reinjection)
                               // self.steps_between_reinjections)

        maxReinjectionsReached = (self.numberOfCompletedReinjections >=
                                  self.maxReinjections)

        # don't reinject on the final step or it crashes. todo fix this.
        finalTimestep = self.isFinalTimestep()

        if (passedFirstInjectionStep and isReinjectionStep and not
            finalTimestep and not maxReinjectionsReached):
            self.numberOfCompletedReinjections += 1
            return True
        else:
            return False
