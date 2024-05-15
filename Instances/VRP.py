#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:13:42 2023

@author: stijnvanderwal

This code contains the necessaryframework for computing the speed 
that can be driven with a certain input.
"""
import numpy

"""
Here the universal used constants are declared. 
All constants are in SI units by default
"""
maximumPower = 350 #W
maximumVelocity = 25/3.6 # ms^-1
airDensity = 1.18 #kg/m^3 
airDragCoefficient = 1.18
frontalArea = 0.83 #m^2
rollResistanceCoefficient = 0.01
gravitationalAcceleration = 9.81 #ms^-2

def binarySearchZero(function, lowerBound, upperBound, bikeLoad,roadSlope, windSpeed, windAngle, tolerance = 1e-5, maxIterations = 10000):
    """
    Binary search algorithm, designed for this code.
    
    Input:
        function:       A monotone function for which we want to determine the root
        lowerbound:     the lower bound for the search region
        upperbound:     the upper bound for the search region
        bikeLoad:       the total weight of the bike
        roadSlope:      the slope of the traversed road
        windSpeed:      wind speed in meters per second
        windAngle:      the angle between the true wind and driving direction in radians
        tolerance:      the range within which the algorithm stops searching
        maxIterations:  the maximum number of search iterations
    
    Output:
        The speed in meters per second for which the input function returns zero
    """
    if function(upperBound, bikeLoad,roadSlope, windSpeed, windAngle) <= 0:
        return 25/3.6
    
    iterations = 0
    
    while abs(upperBound-lowerBound)>=tolerance and iterations <= maxIterations:
        iterations+=1
        midpoint = (upperBound+lowerBound)/2
        if function(midpoint, bikeLoad,roadSlope, windSpeed, windAngle)==0:
            return midpoint
        elif function(midpoint, bikeLoad,roadSlope, windSpeed, windAngle)>= 0:
            upperBound = midpoint
        else:
            lowerBound = midpoint
    return (upperBound+lowerBound)/2

def airDragForce(bikeSpeed, windSpeed, windAngle):
    """Calculate the air drag force
    Input and output in SI units"""
    
    apparantWind = numpy.sqrt(bikeSpeed**2 + bikeSpeed*windSpeed*numpy.cos(windAngle) + windSpeed**2)
    return (airDensity*airDragCoefficient*frontalArea)/2 * apparantWind*(bikeSpeed*windSpeed * numpy.cos(windAngle))

def rollResistance(bikeLoad, roadSlope):
    """Calculate Roll resistance
    Input and output in SI units"""
   
    return rollResistanceCoefficient * bikeLoad * gravitationalAcceleration * numpy.cos(numpy.arctan(roadSlope))

def gravForce(bikeLoad,roadSlope):
    """Calculate Gravitational Force
    Input and output in SI units"""
    
    return bikeLoad * gravitationalAcceleration * numpy.sin(numpy.arctan(roadSlope))

def powerBalance(velocity, bikeLoad,roadSlope, windSpeed, windAngle):
    """
    This function computes the differenct between the power needed to drive 
    a certain speed and a predefined maximum power. It accounts for 
    5% dissipation of power through mechanical losses and assumes 350 Watt.
    
    Input:
        velocity:       in meters per second, at which the vehicle is driving
        bikeLoad:       in kg, total weight of the vehicle  
        windSpeed:      in meters per second, at which the wind is blowing
        windAngle:      in radians, angle bewteen true wind and driving direction
        
    Output:
        net force in Newton on the bicycle
        """
    
    fixedForce = rollResistance(bikeLoad, roadSlope) + gravForce(bikeLoad, roadSlope)
    AR = airDragForce(bikeSpeed=velocity, windSpeed=windSpeed, windAngle=windAngle)
    #with 5% loss for friction in the bike
    return (fixedForce + AR) * velocity/0.95 - 350

def computeSpeed(mass, slope, windSpeed, windAngle):
    """
    This function returns fastests speed that can be driven 
    with the input parameters.
    
    Input:
        mass in kg
        slope in %
        windspeed in ms^-1
        windangle in radians
        
    Output:
        speed in ms^-1
        
        """
    
    return binarySearchZero(powerBalance,0, 25/3.6 , mass,slope, windSpeed, windAngle,tolerance= 1e-5)
