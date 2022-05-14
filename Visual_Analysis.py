#Load Functions from Other Files
from Optimization_Algorithm import *
from Visualization_Production import *
from Input_Functions import *

#Enter Starting Point
x=(-2,2,-1) 

#Chosen Test Function
test_func=convex_funn 
#Gradient Function for Chosen Test Function
gr_test_func= gr_convex_funn
##Name to be Assigned to Titles/Files for Test Function
test_func_name ='TestFunction'

#Chosen Iterative Optimization Method
it_algorithm=adam #iteration method
##Name to be Assigned to Titles/Files for Optimization Method
it_name= 'IterativeMethod'

#Chosen Randomized Direction Method
scaled_dir=standard_dk
##Name to be Assigned to Titles/Files for Scaled Direction Method
dk_name='RandomizedDir'

#Number of Iterations
iter_num = 1000

#Number of Runs/Seeds
seed_num=15 
#M Value
m=1e-01 
#Minima/Solution of Test Function
minima = [1,1,1]
#Chosen number of Contonurs for Plots
contour_num= 50

#Produce Data for Trajectories
op_algorithm(x,test_func,gr_test_func, test_func_name, it_algorithm, it_name, scaled_dir, dk_name, iter_num, seed_num,m,minima) 

#Produce Visualizations
visual_analysis(x,test_func,test_func_name, it_name, scaled_dir, dk_name, iter_num, seed_num,minima, m, contour_num)
