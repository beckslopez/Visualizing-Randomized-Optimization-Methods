# ReadMe for Argonne Summer 2021 Research Experience: Visualizing and Understanding Randomized Optimization Methods

--------------------------------------------------------------------------------------
# Introduction
--------------------------------------------------------------------------------------
This module is written to produce a variety of visualizations to provide useful information for any test function to be used with an randomized optimization algorithm. There are essentially four files to be considered: 

-Input_Functions.py: Provides options for test functions we have already worked with and their gradient as well as iterative optimization algorithms and direction methods
			Others can write their own in similar format to use other methods or functions

-Optimization_Algorithm.py: Runs the actual iterative optimization algorithm and produces a dictionary with data values

-Visualization_Production.py: Produces various plots for function using data from dictionary

-Visual_Analysis.py : Assigns parameter values and calls other files 

The purpose is to enter Visual_Analysis and essentially run all other functions from other files (Input_Functions.py, Optimization_Algorithm.py, Visualization_Production.py) from this one. 

--------------------------------------------------------------------------------------
# Requirements
--------------------------------------------------------------------------------------
There are a few things one must do before running Visual_Analysis.py in the command terminal.

1) One must go into Optimization_Algorithm.py and Visualization_Production.py to change the directory (Directory Name):  

	root = Path(r"Directory Name")

2) Within your directory, create folders named 'Data' and 'Figures' to store your DataDictionary.npy file and png figure images.

3) Assign parameter values within Visualization_Production.py.

After this, one should be able to simply run Visual_Analysis.py from the command prompt.
