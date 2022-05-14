# Visualizing & Understanding Performance of Randomized Optimization Methods: Production of Data 

#Import Necessary Packages 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import os 

from mpl_toolkits import mplot3d 
from collections import defaultdict 
from numpy import savez_compressed 
from numpy import load
from pathlib import Path 

from ipynb.fs.full.Input_Functions import *

#Set Working Directory and Folder for Data to be Stored In

##Working Directory must have folder named "Iteration_Algorithm" for data to be stored in
root = Path(r"C:\Users\Rebec\Documents\University of Washington\Argonne") 
path_data = root / "Data"

#Produce array for m value comparision
def nearest_m(m): 
    
    """
    Create an (1,3) array of M values to visually compare.

    Parameters
    ----------
    m : scalar value
        Input paramter.

    Returns
    -------
    m_array
    """
        
    m_array=[]

    #Create an array with the two nearby elements on a log scale
    m_value=[m/10,m,m*10] 
    m_array=np.append(m_array,m_value) 
    return m_array 

## Produce dictionary with trajectory data
def op_algorithm(x,test_func,gr_test_func, test_func_name, it_algorithm,it_name,scaled_dir, dk_name, iter_num, seed_num,m,minima):
    
    """
    Produce dictionary with trajectory data for all seeds and M values. 

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    test_func : function
        The chosen test function.
        
    gr_test_func : function
        The gradient function for the chosen test function.
    
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.

    it_algorithm : function
        The chosen optimization function.
    
    iter_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.
    
    scaled_dir : function
        The chosen scaled direction function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.
        
    iter : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    m : float
        Input parameter value.
        
    minima : n size array
        The minima of the function if known.


    Returns
    -------
    'DataDictionary.npy'
    """
        
    n=len(x) 
    func_dict={}

    #Produce array of M values 
    m_vector = nearest_m(m)

    #Assign various seeds and m values in order to run optimization algorithm
    for s in range(0,seed_num):
        for q in range (0,len(m_vector)):
            m_value =m_vector[q]

            #Input values into function for the iterative algorithm
            traj_values = it_algorithm(iter_num,x,s,test_func,gr_test_func,scaled_dir,m_value) 

            #Save output to dictionary for each varying seed
            func_dict[('{0}'.format(it_name),'{0}'.format(test_func_name),'{0}'.format(dk_name),'{0}'.format(s),'{0}'.format(m_vector[q]))]=traj_values 

    #Save dictionary of arrays into a npy file 
    np.save(os.path.join(path_data,'DataDictionary.npy'),func_dict) 