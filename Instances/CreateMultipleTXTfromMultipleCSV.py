#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:22:36 2024

This file can be used to create a series of TXT files from a folder of instances.
The TXT files are ready to be used in the SA-ILP algorithm with config VRPLTTfinalKDE.
By adjusting the settings, it is possible to choose which settings are included.

@author: stijnvanderwal
"""

import os
import txtfromcsv
import numpy as np

"""
Here the user can specify the sets of paramters for which a txt file will be created.
By giving a list of numbers, the code creates a txt file for each combination
The name of the TXT file will represent the combination of variables and is 
structured as follows: InstanceName_Ws{windspeed}_Wd{windDirection}_o{nrObservations}.txt
"""
output_folder = "" #this folder contains the instance .csv files
input_folder = "" #user specified folder

WIND_SPEED_FORECAST = [6.75] # in ms^-1
WIND_DIRECTION_FORECAST = [0] # in radians
DATASET_SIZES = [5,20]
nrLoadlevels = 10
nrObservations = max(DATASET_SIZES)

"""Change this seed to obtain different sets of randomized results. 
In the experiments 41 is used for data creation. While 42 is used for data simulation."""
seed = 41

def create_folder(folder_path): 
    os.makedirs(folder_path, exist_ok=True)
    
 
def convert4Dto2D(travelMatrix4D):
    """
    This function converts a 4D matrix with the specified structure to a 2D matrix 
    with the structure that can be understood by the SA-ILP-KDE algorithm.

    Parameters
    ----------
    travelMatrix4D : 4D-array
        contains floats representing the travel time for road (i,j) loadlevel l 
        and observation k as [i,j,l,k].

    Returns
    -------
    travelTimeMatrix2D : 2D-array
        flattened version of the 4D-array.

    """
    nrCustomers, nrLoadlevels, nrObservations = travelMatrix4D.shape[0],travelMatrix4D.shape[2],travelMatrix4D.shape[3]

    nr_edges = nrCustomers**2
    nr_columns = nrLoadlevels * nrObservations
    travelTimeMatrix2D = np.zeros((nr_edges, nr_columns))
    
    for n in range(nr_edges):
        for m in range(nr_columns):
            if n//nrCustomers != n%nrCustomers:
                travelTimeMatrix2D[n,m] = travelMatrix4D[n//nrCustomers, n%nrCustomers, m//nrObservations, m%nrObservations]
    return travelTimeMatrix2D

def convert3Dto2D(travelMatrix3D):
    """
    convert a 3D matrix to 2D. used for creating the txt file for the travel time
    with forecast wind (FC TT).

    Parameters
    ----------
    travelMatrix3D : 3D-array
        road (i,j) and load level l: [i,j,l]

    Returns
    -------
    travelTimeMatrix2D : 2D-array
        flattened version of the 3D matrix that can be understood by the SA-ILP-KDE.

    """
    nrCustomers, nrLoadlevels, nrObservations = travelMatrix3D.shape[0],travelMatrix3D.shape[2],1

    nr_edges = nrCustomers**2
    nr_columns = nrLoadlevels * nrObservations
    travelTimeMatrix2D = np.zeros((nr_edges, nr_columns))
    
    for n in range(nr_edges):
        for m in range(nr_columns):
            if n//nrCustomers != n%nrCustomers:
                travelTimeMatrix2D[n,m] = travelMatrix3D[n//nrCustomers, n%nrCustomers, m//nrObservations]
    return travelTimeMatrix2D

def createFCMatrix(customers, distances, output_folder, input_file, 
                   WIND_SPEED_FORECAST, WIND_DIRECTION_FORECAST):
    """
    FC stands for 'forecast'. This function computes what the travel times 
    would be if the wind would be exactly the forecast and creates the appropriate 
    txt-file that can be used in the SA-ILP-KDE algorithm.
    """
    travel_time_matrix_Forecast = txtfromcsv.computeTravelTimeMatrix3DDeterministic(customers, distances, WIND_SPEED_FORECAST , WIND_DIRECTION_FORECAST ,nrLoadlevels)
    travelMatrixToPrint = convert3Dto2D(travel_time_matrix_Forecast)
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + f"_final_Ws{WIND_SPEED_FORECAST}_Wd{WIND_DIRECTION_FORECAST}_oFC.txt")
    txtfromcsv.write_travel_time_matrix_to_file(travelMatrixToPrint, len(customers), 
                                                nrLoadlevels, 'FC', output_file, WIND_SPEED_FORECAST, 
                          WIND_DIRECTION_FORECAST)

# Function to generate .txt files using TXTfromCSV class
def generate_txt_file(seed, output_folder, input_file, WIND_SPEED_FORECAST, 
                      WIND_SPEED_SIGMA, WIND_DIRECTION_FORECAST, WIND_DIRECTION_SIGMA):
    """
    This function creates and writes the prescribed files for all sizes in the sizes array.

    Parameters
    ----------
    seed : int
        seed for the random generator for data, should be different than the seed 
        for simulation.
    output_folder : string
        path to the output folder where the TXT files need to be stored.
    input_file : string
        path to the CSV files representing the instances.
    WIND_SPEED_FORECAST : float
        forecast wind in ms^-1 for which the txt files are generated.
    WIND_SPEED_SIGMA : float
        standard deviation of wind speed in ms^-1.
    WIND_DIRECTION_FORECAST : float
        forecast wind direction in radians.
    WIND_DIRECTION_SIGMA : TYPE
        standard deviation of wind direction in radians.

    Raises
    ------
    Exception
        whenever standard deviation is larger than 7 i.e. probably input in degrees 
        instead of radians.

    Returns
    -------
    None.

    """

    customers, distances = txtfromcsv.csvImporter(input_file)
    if abs(WIND_DIRECTION_SIGMA) >= 7:
        raise Exception()
    # this returns a 4D matrix
    #travel_time_matrix = txtfromcsv.computeTravelTimeMatrix4D(seed, customers, distances, 
     #                       WIND_SPEED_FORECAST, WIND_SPEED_SIGMA, WIND_DIRECTION_FORECAST, WIND_DIRECTION_SIGMA, 
      #                      nrObservations, nrLoadlevels)
    travel_time_matrix = txtfromcsv.computeTravelTimeMatrix4DWithTwoModes(seed, customers, distances, 
                            WIND_SPEED_FORECAST, WIND_SPEED_SIGMA, WIND_DIRECTION_FORECAST, WIND_DIRECTION_SIGMA, np.round(WIND_DIRECTION_FORECAST + (90/180*np.pi),2), WIND_DIRECTION_SIGMA, 
                            nrObservations, nrLoadlevels)
    for size in DATASET_SIZES:
        travelMatrixToPrint = convert4Dto2D(travel_time_matrix[:,:,:,0:size])
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + f"_final_Ws{WIND_SPEED_FORECAST}_Wd{WIND_DIRECTION_FORECAST}_o{size}_multimodal.txt")
        txtfromcsv.write_travel_time_matrix_to_file(travelMatrixToPrint, len(customers), nrLoadlevels, size, 
                                                    output_file, WIND_SPEED_FORECAST, 
                                                    WIND_DIRECTION_FORECAST)
        
    createFCMatrix(customers, distances, output_folder, input_file, 
                   WIND_SPEED_FORECAST, WIND_DIRECTION_FORECAST)

# The function makes sure we use each instance from the input folder
def generate_txt_files_from_folder(seed, output_folder, input_folder, WIND_SPEED_FORECAST, 
                                   WIND_SPEED_SIGMA, WIND_DIRECTION_FORECAST, WIND_DIRECTION_SIGMA):
    create_folder(output_folder)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if os.path.splitext((file))[0].split("_")[-1] == 'Store':
                continue
            print("\n")
            print(f"Busy converting {file}")
            input_file = os.path.join(root, file)
            if os.path.exists(os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + f"_final_Ws{WIND_SPEED_FORECAST}_Wd{np.round(WIND_DIRECTION_FORECAST,2)}_o5_multimodal.txt")):
                print(f"_Ws{WIND_SPEED_FORECAST}_Wd{WIND_DIRECTION_FORECAST} already exists for {os.path.splitext(os.path.basename(input_file))[0]}")
                continue
            generate_txt_file(seed, output_folder, input_file, WIND_SPEED_FORECAST, 
                                  WIND_SPEED_SIGMA, WIND_DIRECTION_FORECAST, WIND_DIRECTION_SIGMA)

def main():
    for windDirection in WIND_DIRECTION_FORECAST:
        for windSpeed in WIND_SPEED_FORECAST:
            print(f"\t Now starting for windspeed:{windSpeed} ms^-1, dir:{np.round(windDirection/np.pi*180,0)} degrees")
            # These formulas for standard deviation are based on: https://doi.org/10.1175/1520-0450(1988)027%3C0550:SDOWSA%3E2.0.CO;2
            if windSpeed<5:
                windSsigma = 0.4*windSpeed
                windDsigma = 1/windSpeed # 0.32/windSpeed#*(180/np.pi)
            else:
                windSsigma = 0.08*windSpeed
                windDsigma = 0.1 #rad
            generate_txt_files_from_folder(seed, output_folder, input_folder, 
                                           windSpeed,windSsigma, windDirection, windDsigma)

if __name__ == "__main__":
    main()
