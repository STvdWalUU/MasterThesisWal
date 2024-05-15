#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:22:41 2024

@author: stijnvanderwal
"""

import numpy as np
import csv
import random
import math
import scipy.stats as stats

import VRP
from SimulatorClasses import customer
    
def count_lines(filename):
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        line_count = sum(1 for _ in reader)
    return line_count

def TravelTimeComputer(loadLevel, numberLoadlevels, travelDistance,slope, windSpeed, windAngle):
    """
    Parameters
    ----------
    loadLevel : int
        should be at least zero and at most the number of load levels.
    numberLoadlevels : int
        total number of loadlevels in which the load is divided.
    travelDistance : float
        distance in meters.
    slope : float
        fraction between vertical displacement over horizontal displacement.
    windSpeed : float
        in ms^-1.
    windAngle : float
        in radians.

    Returns
    -------
    float
        travel time in minutes.

    """
    mass = 140 + (loadLevel+0.5)* (290-140)/numberLoadlevels
    driveSpeed = VRP.computeSpeed(mass, slope, windSpeed, windAngle)
    
    return travelDistance/driveSpeed/60

def csvImporter(filename):
    """

    Parameters
    ----------
    filename : string
        filepath to CSV file that is to be converted.

    Returns
    -------
    customers : list
        list of customers deducted from the csv file.
    distanceMatrix : 2D-array
        distances based on coordinates in csv file.

    """

    numberCustomers = count_lines(filename)
    customers = []
    distanceMatrix = np.zeros([numberCustomers,numberCustomers])
    i = 0
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        next(csv_reader)

        for line in csv_reader:
            #0= id; 1= x; 2=y; 3=h; 4=demand; 5=twa; 6=twb; 7=service time
            if int(line[0])==0:
                cust = customer(0, float(line[1]), float(line[2]), float(line[3]), 0, 0, 1e5,0)
            else:
                cust = customer(int(line[0]), float(line[1]), float(line[2]), float(line[3]),int(float(line[4])), float(line[5]), float(line[6]), int(float(line[7])))
            customers.append(cust)
            for j in range(numberCustomers):
                try:
                    distanceMatrix[i,j] = max(float(line[j+8])*1000, distanceMatrix[j,i])
                except:
                    pass
            i+=1
    
    return customers, distanceMatrix

def computeTravelTimeMatrix(customers, distance_matrix, wind_speed_forecast, wind_speed_sigma, wind_direction_forecast, wind_direction_sigma, number_observations, number_loadlevels):
    """
    This function will return a 'flattened' matrix (in 2D) that can be exported 
    using .TXT and understood by the C# algorith for SA-ILP. It takes in a
    predefined list of customers and a distance matrix. These, together with
    generated wind forecasts (Monte-Carle like), determine the 'observations'
    for the travel time between each pair of customers and each possible loadlevel.

    Parameters
    ----------
    customers : list
        list of customer objects.
    distance_matrix : 2D-array
    wind_speed_forecast : float
        in ms^-1.
    wind_speed_sigma : float
        standard deviation in ms^-1.
    wind_direction_forecast : float
        in radians.
    wind_direction_sigma : float
        standard deviation in radians.
    number_observations : int
        desired number of observations.
    number_loadlevels : int

    Returns
    -------
    travel_time_matrix : 2D-array
        flattened matrix representing the travel time matrix observations.
    """
    try:
        scale = wind_speed_sigma**2 / wind_speed_forecast
        shape = wind_speed_forecast / scale
    except ZeroDivisionError:
        shape = 0
    
    nr_edges = len(customers)**2
    nr_columns = number_observations * number_loadlevels
    travel_time_matrix = np.zeros((nr_edges, nr_columns))
    
    for n in range(nr_columns):
        if shape == 0:
            wind_speed = wind_speed_forecast
        else:
            wind_speed = max(0, random.gammavariate(shape, scale))
        wind_direction = random.normalvariate(wind_direction_forecast, wind_direction_sigma)
        
        load_level = n // number_observations
        
        for cust_start, start_customer in enumerate(customers):
            for cust_end, end_customer in enumerate(customers):
                if cust_start != cust_end:
                    travel_distance = distance_matrix[cust_start, cust_end]
                    heading = math.atan2((end_customer.y - start_customer.y), (end_customer.x - start_customer.x))
                    wind_angle = wind_direction - heading
                    slope = (end_customer.h - start_customer.h) / travel_distance
                    travel_time_matrix[cust_start * len(customers) + cust_end, n] = round(TravelTimeComputer(load_level, number_loadlevels,travel_distance, slope, wind_speed, wind_angle), 2)
    
    return travel_time_matrix

def computeTravelTimeMatrixWithWeights(customers, distance_matrix, wind_speed_forecast, wind_speed_sigma, wind_direction_forecast, wind_direction_sigma, number_observations, number_loadlevels):
    """
    This function will return a 'flattened' matrix (in 2D) that can be exported 
    using .TXT and understood by the C# algorith for SA-ILP. It takes in a
    predefined list of customers and a distance matrix. These, together with
    predefined wind parameters determine a select set of 'observations'.
    """
    nr_edges = len(customers)**2
    nr_columns = number_observations * number_loadlevels
    travel_time_matrix = np.zeros((nr_edges, nr_columns))
    windDirections = [wind_direction_forecast, wind_direction_forecast-wind_direction_sigma, wind_direction_forecast+wind_direction_sigma]
    windSpeed = [wind_speed_forecast,wind_speed_forecast-wind_speed_sigma, wind_speed_forecast + wind_speed_sigma]
    n=0
    for wind_direction in windDirections:
        for wind_speed in windSpeed:
            for load_level in range(number_loadlevels):
                for cust_start, start_customer in enumerate(customers):
                    for cust_end, end_customer in enumerate(customers):
                        if cust_start != cust_end:
                            travel_distance = distance_matrix[cust_start, cust_end]
                            heading = math.atan2((end_customer.y - start_customer.y), (end_customer.x - start_customer.x))
                            wind_angle = wind_direction - heading
                            slope = (end_customer.h - start_customer.h) / travel_distance
                            travel_time_matrix[cust_start * len(customers) + cust_end, n] = round(TravelTimeComputer(load_level, number_loadlevels,travel_distance, slope, wind_speed, wind_angle), 2)
                n+=1
    return travel_time_matrix

def computeTravelTimeMatrix4D(seed, customers, distance_matrix, wind_speed_forecast, 
                              wind_speed_sigma, wind_direction_forecast, wind_direction_sigma, 
                              number_observations, number_loadlevels):
    """
    Creates a 4d array that can be sliced and written to txt file to demand.

    Parameters
    ----------
    seed : int
    customers : list 
        list of customer objects.
    distance_matrix : 2D-array
    wind_speed_forecast : float
        in ms^-1.
    wind_speed_sigma : float
        standard deviation in ms^-1.
    wind_direction_forecast : float
        in radians.
    wind_direction_sigma : float
        standard deviation in radians.
    number_observations : int
        number of observations for matrix.
    number_loadlevels : int

    Returns
    -------
    travel_time_matrix : 4D-array
        output matrix such that item [i,j,l,k] is road (i,j), loadlevel l and observation k.

    """
    random.seed(seed)
    
    try:
        #this is the k-theta-formulation from wikipedia (not the alpha-beta formulation !!!)
        scale =  (wind_speed_sigma**2) / wind_speed_forecast # beta
        shape = wind_speed_forecast / scale # alpha
    except ZeroDivisionError:
        shape = 0
    
    travel_time_matrix = np.zeros((len(customers), len(customers), number_loadlevels, number_observations))
    
    for k in range(number_observations):
        if k%(number_observations//10) == 0:
            print(f"{round(k/number_observations,2)*100}%")
        if shape == 0:
            wind_speed = wind_speed_forecast
        else:
            wind_speed = max(0, random.gammavariate(shape, scale))
        wind_direction = random.normalvariate(wind_direction_forecast, wind_direction_sigma)
        for cust_start, start_customer in enumerate(customers):
            for cust_end, end_customer in enumerate(customers):
                if cust_start != cust_end:
                    for loadLevel in range(number_loadlevels):
                        travel_distance = distance_matrix[cust_start, cust_end]
                        heading = math.atan2((end_customer.y - start_customer.y), (end_customer.x - start_customer.x))
                        wind_angle = wind_direction - heading
                        slope = (end_customer.h - start_customer.h) / travel_distance
                        travel_time_matrix[cust_start,cust_end,loadLevel,k] = round(TravelTimeComputer(loadLevel, number_loadlevels,travel_distance, slope, wind_speed, wind_angle), 2)   
    return travel_time_matrix

def computeTravelTimeMatrix4DWithWeights(customers, distance_matrix, wind_speed_forecast, 
                              wind_speed_sigma, wind_direction_forecast, wind_direction_sigma, 
                              number_loadlevels):
    """
    Creates a 4d array that can be sliced and written to txt file to demand.

    Parameters
    ----------
    seed : int
    customers : list 
        list of customer objects.
    distance_matrix : 2D-array
    wind_speed_forecast : float
        in ms^-1.
    wind_speed_sigma : float
        standard deviation in ms^-1.
    wind_direction_forecast : float
        in radians.
    wind_direction_sigma : float
        standard deviation in radians.
    number_observations : int
        number of observations for matrix.
    number_loadlevels : int

    Returns
    -------
    travel_time_matrix : 4D-array
        output matrix such that item [i,j,l,k] is road (i,j), loadlevel l and observation k.

    """
    windSpeeds = [wind_speed_forecast + 1.5*wind_speed_sigma, wind_speed_forecast + 0.5*wind_speed_sigma, wind_speed_forecast -0.5*wind_speed_sigma, wind_speed_forecast - 1.5*wind_speed_sigma,]
    windDirections = [wind_direction_forecast - wind_direction_sigma, wind_direction_forecast, wind_direction_forecast + wind_direction_sigma]
    
    travel_time_matrix = np.zeros((len(customers), len(customers), number_loadlevels, 12))
    weights = computeWeights(wind_speed_forecast, wind_speed_sigma)
    
    for k in range(len(windSpeeds)*len(windDirections)):
        print(k)
        for cust_start, start_customer in enumerate(customers):
            for cust_end, end_customer in enumerate(customers):
                if cust_start != cust_end:
                    for loadLevel in range(number_loadlevels):
                        travel_distance = distance_matrix[cust_start, cust_end]
                        heading = math.atan2((end_customer.y - start_customer.y), (end_customer.x - start_customer.x))
                        wind_angle = windDirections[k%3] - heading
                        slope = (end_customer.h - start_customer.h) / travel_distance
                        travel_time_matrix[cust_start,cust_end,loadLevel,k] = round(TravelTimeComputer(loadLevel, number_loadlevels,travel_distance, slope, windSpeeds[k//3], wind_angle), 2)   
    return travel_time_matrix, weights

def gamma_cdf_difference(x1, x2, shape, scale):
    cdf_x1 = stats.gamma.cdf(x1, a=shape, scale=scale)
    cdf_x2 = stats.gamma.cdf(x2, a=shape, scale=scale)
    return cdf_x2 - cdf_x1

def computeWeights(windSpeedForecast,windSpeedSigma):
    directionWeights = [0.25, 0.36, 0.25]
    speedWeights = [0,0,0,0]
    scale =  (windSpeedSigma**2) / windSpeedForecast # beta
    shape = windSpeedForecast / scale # alpha
    speedWeights[0] = gamma_cdf_difference(windSpeedForecast-2*windSpeedSigma, windSpeedForecast-windSpeedSigma, shape, scale)
    speedWeights[1] = gamma_cdf_difference(windSpeedForecast-1*windSpeedSigma, windSpeedForecast, shape, scale)
    speedWeights[2] = gamma_cdf_difference(windSpeedForecast, windSpeedForecast+windSpeedSigma, shape, scale)
    speedWeights[3] = gamma_cdf_difference(windSpeedForecast+windSpeedSigma, windSpeedForecast+ 2*windSpeedSigma, shape, scale)

    weights = np.zeros(12)
    fractions = []
    for i in range(12):
        fractions.append(directionWeights[i%3]*speedWeights[i//3])
        
    for i in range(12):
        fractions[i] /= sum(fractions)
        weights[i] += 1
    
    while sum(weights) <= 99:
        #print(weights)
        max_index = np.argmax(fractions - weights/sum(weights))
        if max_index%3 ==0:
            weights[max_index] += 1
            weights[max_index+2] += 1
        elif max_index%3 ==2:
            weights[max_index] += 1
            weights[max_index-2] += 1
        else:
            weights[max_index] += 1
    
    #print(np.floor(100*weights), sum(np.floor(100*weights)))
        
    return weights

def computeTravelTimeMatrix4DWithTwoModes(seed, customers, distance_matrix, wind_speed_forecast, 
                              wind_speed_sigma, wind_direction_forecast_one, wind_direction_sigma_one,  wind_direction_forecast_two, wind_direction_sigma_two,
                              number_observations, number_loadlevels):
    """
    Creates a 4d array that can be sliced and written to txt file to demand. 
    It uses two forecasted directions with their own standard deviations

    Parameters
    ----------
    seed : int
    customers : list 
        list of customer objects.
    distance_matrix : 2D-array
    wind_speed_forecast : float
        in ms^-1.
    wind_speed_sigma : float
        standard deviation in ms^-1.
    wind_direction_forecast_one : float
        in radians.
    wind_direction_sigma_one: float
        standard deviation in radians.
        wind_direction_forecast_two : float
            in radians.
        wind_direction_sigma_two : float
            standard deviation in radians.
    number_observations : int
        number of observations for matrix.
    number_loadlevels : int

    Returns
    -------
    travel_time_matrix : 4D-array
        output matrix such that item [i,j,l,k] is road (i,j), loadlevel l and observation k.

    """
    random.seed(seed)
    
    try:
        #this is the k-theta-formulation from wikipedia (not the alpha-beta formulation !!!)
        scale =  (wind_speed_sigma**2) / wind_speed_forecast # beta
        shape = wind_speed_forecast / scale # alpha
    except ZeroDivisionError:
        shape = 0
    
    travel_time_matrix = np.zeros((len(customers), len(customers), number_loadlevels, number_observations))
    
    for k in range(number_observations):
        if k%(number_observations//10) == 0:
            print(f"{round(k/number_observations,2)*100}%")
        if shape == 0:
            wind_speed = wind_speed_forecast
        else:
            wind_speed = max(0, random.gammavariate(shape, scale))
        wind_direction_one = random.normalvariate(wind_direction_forecast_one, wind_direction_sigma_one)
        wind_direction_two = random.normalvariate(wind_direction_forecast_two, wind_direction_sigma_two)
        wind_direction = random.choice([wind_direction_one, wind_direction_two])
        for cust_start, start_customer in enumerate(customers):
            for cust_end, end_customer in enumerate(customers):
                if cust_start != cust_end:
                    for loadLevel in range(number_loadlevels):
                        travel_distance = distance_matrix[cust_start, cust_end]
                        heading = math.atan2((end_customer.y - start_customer.y), (end_customer.x - start_customer.x))
                        wind_angle = wind_direction - heading
                        slope = (end_customer.h - start_customer.h) / travel_distance
                        travel_time_matrix[cust_start,cust_end,loadLevel,k] = round(TravelTimeComputer(loadLevel, number_loadlevels,travel_distance, slope, wind_speed, wind_angle), 2)   
    return travel_time_matrix

def computeTravelTimeMatrix3DDeterministic(customers, distance_matrix, wind_speed_forecast, 
                                           wind_direction_forecast, number_loadlevels):
    """
    This function will return the 3D matrix that is used as input to create
    the flattened 2D matrices for exporting to the SA-ILP algorithm using a 
    .txt-file. It will use the forecast as a deterministic given and produce
    travel times as such. The shape and form is such that it can be understood 
    by the SA-ILP-KDE algrotihm.
    """
    
    travel_time_matrix = np.zeros((len(customers), len(customers), number_loadlevels))

    for cust_start, start_customer in enumerate(customers):
        for cust_end, end_customer in enumerate(customers):
            if cust_start != cust_end:
                for loadLevel in range(number_loadlevels):
                    travel_distance = distance_matrix[cust_start, cust_end]
                    heading = math.atan2((end_customer.y - start_customer.y), (end_customer.x - start_customer.x))
                    wind_angle = wind_direction_forecast - heading
                    slope = (end_customer.h - start_customer.h) / travel_distance
                    travel_time_matrix[cust_start,cust_end,loadLevel] = round(TravelTimeComputer(loadLevel, number_loadlevels,travel_distance, slope, wind_speed_forecast, wind_angle), 2)   
    
    return travel_time_matrix

def write_travel_time_matrix_to_file(travel_time_matrix, number_customers, number_loadlevels, 
                                     number_observations, file_path, WIND_SPEED_FORECAST, 
                      WIND_DIRECTION_FORECAST):
    """
    writes a travel time matrix and several paramters to a txt file with prespecified name.
    

    Parameters
    ----------
    travel_time_matrix : 2D-array
        input matrix that needs to be written into txt file.
    number_customers : int
    number_loadlevels : int
    number_observations : int
    file_path : string
        path at which the file needs to be saved.
    WIND_SPEED_FORECAST : float
    WIND_DIRECTION_FORECAST : float

    Returns
    -------
    None.

    """
    with open(file_path, 'w') as file:
        # Write metadata to the file
        file.write(f"{number_customers}, {number_loadlevels}, {number_observations}, wind:{WIND_SPEED_FORECAST} ms^-1, {WIND_DIRECTION_FORECAST} deg \n")
        
        # Write the travel time matrix data to the file
        np.savetxt(file, travel_time_matrix, delimiter=",", fmt="%.2f")

def write_travel_time_matrix_to_file_weights(travel_time_matrix, number_customers, number_loadlevels, 
                                     number_observations, file_path, WIND_SPEED_FORECAST, 
                      WIND_DIRECTION_FORECAST, weights):
    """
    writes a travel time matrix and several paramters to a txt file with prespecified name.
    

    Parameters
    ----------
    travel_time_matrix : 2D-array
        input matrix that needs to be written into txt file.
    number_customers : int
    number_loadlevels : int
    number_observations : int
    file_path : string
        path at which the file needs to be saved.
    WIND_SPEED_FORECAST : float
    WIND_DIRECTION_FORECAST : float

    Returns
    -------
    None.

    """
    with open(file_path, 'w') as file:
        # Write metadata to the file
        file.write(f"{number_customers}, {number_loadlevels}, {number_observations}, wind:{WIND_SPEED_FORECAST} ms^-1, {WIND_DIRECTION_FORECAST} deg \n")
        for i in range(len(weights)-1): 
            file.write(f"{int(weights[i])},")
        file.write(f"{int(weights[-1])}")
        file.write("\n")
        
        # Write the travel time matrix data to the file
        np.savetxt(file, travel_time_matrix, delimiter=",", fmt="%.2f")
        
        