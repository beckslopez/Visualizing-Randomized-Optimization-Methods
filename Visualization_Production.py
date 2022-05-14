#Visualizing & Understanding Performance of Randomized Optimization Methods: Production of Visuals

##Imported Packages
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d  
import matplotlib.colors as colors 
import cv2 
import os

from mpl_toolkits import mplot3d 
from collections import defaultdict
from itertools import combinations 

from numpy import savez_compressed 
from numpy import load 
from pathlib import Path
from array import array
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

#Set Working Directory

root = Path(r"C:\Users\Rebec\Documents\University of Washington\Argonne")

#Path for the data
path_data = root / "Data" 
#Path for figures
path_Figures = root / "Figures" 

#Import data 
traj_dict= np.load(os.path.join(path_data,'DataDictionary.npy'), allow_pickle= 'TRUE')
#Change from np array to dictionary
traj_array = dict(enumerate(traj_dict.flatten(), 1)) 
#Take the first element, which removes the enumerated dictionary
data_dict=traj_array[1] 

#Array of Viusal Marker Options for Plotting

#Sequential color maps from MatPlotLib 
sequential_color_maps=['Greys_r', 'Purples_r', 'Greens_r', 'Blues_r','Reds_r','Oranges_r',
            'YlOrBr_r', 'YlOrRd_r', 'OrRd_r', 'PuRd_r', 'RdPu_r', 'BuPu_r',
            'GnBu_r', 'PuBu_r', 'YlGnBu_r', 'PuBuGn_r', 'BuGn_r', 'YlGn_r'] 

#Solid colors corresponding to the sequential color maps
solid_colors=['k','m','g','b','r','orange']

#Marker shapes for the plots
mark_array=['*','.','^','D'] 


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

#Statistical Analysis of Trajectories
def tensor(it_name,test_func, test_func_name,dk_name,seed_num,m,iter_num):
    """
    Produce stastical analysis measures for trajectories.

    Parameters
    ----------
  
    iter_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.
    
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.       
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    m : float
        Input parameter value.

    iter : integer
        The number of iterations.
        

    Returns
    -------
    tensor,mean_traj,std_traj,med_traj,mean_solu
    """
    x_list=[]
    solution_list=[] 
    
    #For all seeds up to seed_num
    for s in range(0,seed_num): 
            f_list=[] 
            #Search dictionary for data of trajectory
            x_bold =data_dict[('{0}'.format(it_name),'{0}'.format(test_func_name),'{0}'.format(dk_name),'{0}'.format(s),'{0}'.format(m))] 
            x_list.append(x_bold)
            
            #Find the solution value for each iteration for each seed
            for k in range(0,iter_num): 
                f_value=test_func(x_bold[k]) 
                f_list.append(f_value)
            solution_list.append(f_list)

    #Returns (iter_num,n,seed_num) array 
    tensor = np.dstack((x_list)) 
    #Returns (iter_num,1,seed_num) array 
    solution = np.dstack((solution_list)) 
    
    #Returns (iter_num,n) array for the average trajectory
    mean_traj = [np.mean(k,axis=1) for k in tensor] 
    #Returns (iter_num,n) array for standard deviation for each elemnet
    std_traj = [np.std(k,axis=1) for k in tensor] 
    #Returns (iter_num,n) array for median trajectory
    med_traj = [np.median(k,axis=1) for k in tensor] 
    #Returns (iter_num,1) array for the average solution
    mean_solu = [np.mean(k,axis=1) for k in solution] 
    
    return tensor,mean_traj,std_traj,med_traj,mean_solu

#Produce Confidence ellipses to demonstrate covariance
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D()         .rotate_deg(45)         .scale(scale_x, scale_y)         .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    
    # Get the path
    path = ellipse.get_path()
    # Get the list of path vertices
    vertices = path.vertices.copy()
    
    # Transform the vertices so that they have the correct coordinates
    xx,yy=vertices.T

    return ax.add_patch(ellipse), xx,yy#ax.add_patch(ellipse)

## General Visualization with all seeds and mean trajectory
def gen_visual(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num,seed_num,minima,m, contour_num):
    """
    Produce general plot with all trajectories for all seeds with mean trajectory.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.
    
    chosen_pair : (1,2) array of indices
        The chosen 2D slice.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter value.

    contour_num: integer
        Amount of Contour lines on plot.
        

    Returns
    -------
    'Gen_ItName_FuncName_DkName_ChosenPair_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1) 
    
    min_x1,max_x1,min_x2,max_x2=[],[],[],[]
    
    #Multiple Seeds on Same Plot
    for s in range(0, seed_num):
        x_bold = data_dict[('{0}'.format(it_name),'{0}'.format(test_func_name),'{0}'.format(dk_name),'{0}'.format(s), '{0}'.format(m))] #search dictionary for needed array
        
        #Select two dimensional slice of values from every possible combination
        x_slice = x_bold[0:iter_num+1,chosen_pair]    
        it_x1 = x_slice[0:iter_num+1,0] 
        it_x2 = x_slice[0:iter_num+1,1] 
        
        #Create (1,seed_num) arrays with the minimum and maximum values of each x for each seed
        min_x1=np.append(min_x1, np.amin(it_x1)) 
        max_x1=np.append(max_x1, np.amax(it_x1))
        min_x2=np.append(min_x2, np.amin(it_x2))
        max_x2=np.append(max_x2, np.amax(it_x2))
        
        
        #Arrows needed for quiver plot
        arrow_x1 = it_x1[1:] - it_x1[:-1] 
        arrow_x2 = it_x2[1:] - it_x2[:-1] 
        #Plotting the iterations with resulting point values
        ax.scatter(it_x1,it_x2,color = 'r', marker = '.')
        #Plotting arrows with intermediate values
        ax.quiver(it_x1[:-1], it_x2[:-1], arrow_x1, arrow_x2, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
        
    #Mean Trajectory on Same Plot
    
    #Retrieve mean for certain condiitions from tensor
    x_mean=np.array(tensor(it_name,test_func, test_func_name,dk_name,seed_num,m,iter_num)[1]) 
    
    #Select two dimensional slice of values from every possible combination  
    x_mean_slice = x_mean[0:iter_num+1,chosen_pair]      
        
    it_mx1 = x_mean_slice[0:iter_num+1,0] 
    it_mx2 = x_mean_slice[0:iter_num+1,1]
        
    #Arrows needed for quiver plot
    arrow_mx1 = it_mx1[1:] - it_mx1[:-1]
    arrow_mx2 = it_mx2[1:] - it_mx2[:-1] 
    ax.scatter(it_mx1,it_mx2,color = 'k', marker = '*',label= 'Average Trajectory')
    ax.quiver(it_mx1[:-1], it_mx2[:-1], arrow_mx1, arrow_mx2, scale_units = 'xy', angles = 'xy', scale = 1, color = 'k', alpha = .3) 
    
    #Finds scalar minimum and maximum values for each x value
    x1_range_min=np.minimum(np.amin(min_x1),minima[0]) 
    x1_range_max=np.maximum(np.amax(max_x1),minima[1]) 
    x2_range_min=np.minimum(np.amin(min_x2),minima[0]) 
    x2_range_max=np.maximum(np.amax(max_x2),minima[1])

    #Provides values for x_1 over a range
    x_1 = np.linspace((x1_range_min-.25),(x1_range_max+.25),contour_num) 
    #Provides values for x_2 over a range
    x_2 = np.linspace((x2_range_min-.25),(x2_range_max+.25),contour_num) 
    #Provides us with arrays so we can find every possible combination for x_1,x_2
    X1,X2 = np.meshgrid(x_1, x_2) 


    X=[] 
    
    #For each value on our grid
    for q in range(0,contour_num): 
        for p in range(0,contour_num):
            x_points=minima
            x_points[chosen_pair[0]]=X1[0,p]
            x_points[chosen_pair[1]]= X2[q,0]
            
            #Taking the test function values for mesh grid
            X=np.append(X,test_func(x_points)) 
    #Reshape to array of size (contour number, contour number)
    X=X.reshape(contour_num,contour_num)

    #Creates contour lines on plot
    ax.contour(X1,X2,X, contour_num, cmap = 'jet',zorder=-1) 
    
    #Plotting starting point 
    ax.plot(x[chosen_pair[0]],x[chosen_pair[1]],color='green', marker = 'X',markersize=15)  
    #Plotting minima
    ax.plot(minima[chosen_pair[0]],minima[chosen_pair[1]],color='gold', marker = '*',markersize=20) 
    
    #Plot Title and Subtitle
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize='large') 
    ax.set_title('General Plot with {} Iterations and {} Seeds'.format( iter_num,seed_num), fontsize='medium') 
    
    #Set Axis labels
    plt.xlabel("$x_{}$".format(chosen_pair[0])) 
    plt.ylabel("$x_{}$".format(chosen_pair[1])) 
    
    #Set Plot Legend
    plt.legend(fontsize='x-large')
    
    #Save Figure
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_{}_x{}x{}_{}_{}_{}.png".format('Gen',
                                                                                   it_name,
                                                                                   test_func_name,
                                                                                   dk_name,chosen_pair[0],
                                                                                   chosen_pair[1],iter_num,
                                                                                   seed_num,np.format_float_scientific(m,trim='-'))),dpi=100) #Save plot to folder
    #Close out plot since they are saved
    plt.close('all') 

## Plot: Comparision between Iterations & Seeds
def iter_comp(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num,seed_num,minima,m, contour_num=50):
    """
    Produce plot with all trajectories for all seeds differentiating between seeds and displays colormap over iter with mean trajectory.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.
    
    chosen_pair : (1,2) array of indices
        The chosen 2D slice.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter value.

    contour_num: integer
        Amount of Contour lines on plot.
        

    Returns
    -------
    'IterComp_ItName_FuncName_DkName_ChosenPair_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1) 
    
    min_x1,max_x1,min_x2,max_x2=[],[],[],[]
       
    #Multiple Seeds on Same Plot
    for s in range(0, seed_num):

        x_bold =data_dict[('{0}'.format(it_name),'{0}'.format(test_func_name),'{0}'.format(dk_name),'{0}'.format(s), '{0}'.format(m))] 
        #Select two dimensional slice of values from every possible combination
        x_slice = x_bold[0:iter_num+1,chosen_pair]         
  
        it_x1 = x_slice[0:iter_num+1,0] 
        it_x2 = x_slice[0:iter_num+1,1]
        
        min_x1=np.append(min_x1, np.amin(it_x1)) #Create an (1,seed_num) array with the minimum value of each x_1 column for each seed
        max_x1=np.append(max_x1, np.amax(it_x1)) #Create an (1,seed_num) array with the maximum value of each x_1 column for each seed
        min_x2=np.append(min_x2, np.amin(it_x2)) #Create an (1,seed_num) array with the minimum value of each x_ 2 column for each seed
        max_x2=np.append(max_x2, np.amax(it_x2)) #Create an (1,seed_num) array with the maximum value of each x_2 column for each seed
            
        ax.scatter(it_x1,it_x2,cmap = plt.cm.get_cmap(sequential_color_maps[((s+1)%len(sequential_color_maps))]),  marker = '.')#Plotting the iterations with resulting point values
 
    #Mean Trajectory on Same Plot
    x_mean=np.array(tensor(it_name,test_func, test_func_name,dk_name,seed_num,m,iter_num)[1]) 
    x_mean_slice = x_mean[0:iter_num+1,chosen_pair]  
        
    it_mx1 = x_mean_slice[0:iter_num+1,0] 
    it_mx2 = x_mean_slice[0:iter_num+1,1] 
    
    ax.scatter(it_mx1,it_mx2, cmap= sequential_color_maps[0], c=iter_k,  marker = '*', s= 35, label ='Average Trajectory')#Plotting the iterations with resulting point values
   
    #Finds scalar minimum and maximum values for each x value
    x1_range_min=np.minimum(np.amin(min_x1),minima[0]) 
    x1_range_max=np.maximum(np.amax(max_x1),minima[1]) 
    x2_range_min=np.minimum(np.amin(min_x2),minima[0]) 
    x2_range_max=np.maximum(np.amax(max_x2),minima[1])

    #Provides values for x_1 over a range
    x_1 = np.linspace((x1_range_min-.25),(x1_range_max+.25),contour_num) 
    #Provides values for x_2 over a range
    x_2 = np.linspace((x2_range_min-.25),(x2_range_max+.25),contour_num) 
    #Provides us with arrays so we can find every possible combination for x_1,x_2
    X1,X2 = np.meshgrid(x_1, x_2) 


    X=[] 
    
    #For each value on our grid
    for q in range(0,contour_num): 
        for p in range(0,contour_num):
            x_points=minima
            x_points[chosen_pair[0]]=X1[0,p]
            x_points[chosen_pair[1]]= X2[q,0]
            
            #Taking the test function values for mesh grid
            X=np.append(X,test_func(x_points)) 
    #Reshape to array of size (contour number, contour number)
    X=X.reshape(contour_num,contour_num)

    #Creates contour lines on plot
    ax.contour(X1,X2,X, contour_num, cmap = 'jet',zorder=-1) 
    
    #Plotting starting point 
    ax.plot(x[chosen_pair[0]],x[chosen_pair[1]],color='red', marker = 'X',markersize=15)  
    #Plotting minima
    ax.plot(minima[chosen_pair[0]],minima[chosen_pair[1]],color='gold', marker = '*',markersize=20) 
    
    #Plot Title and Subtitle
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize="large") 
    ax.set_title('Comparsion Along Seeds and Iterations with {} Iterations and {} Seeds'.format( iter_num,
                                                                                                seed_num), fontsize='medium') 
    
    #Set Axis labels
    plt.xlabel("$x_{}$".format(chosen_pair[0])) 
    plt.ylabel("$x_{}$".format(chosen_pair[1])) 
    
    #Set Plot Legend
    plt.legend(fontsize='x-large')
    
    #Save Figure
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_{}_x{}x{}_{}_{}_{}.png".format('IterComp',
                                                                                   it_name,
                                                                                   test_func_name,
                                                                                   dk_name,
                                                                                   chosen_pair[0],
                                                                                   chosen_pair[1],
                                                                                   iter_num,
                                                                                   seed_num,
                                                                                   np.format_float_scientific(m,trim='-'))),dpi=100) 
    
    #Close out plot since they are saved
    plt.close('all')
  
 ## Plot: Iteration Comparision in 3 Dimensions
def iter_comp_3d(x,it_name,test_func,test_func_name,dk_name,iter_num,seed_num,minima,m, contour_num):
    """
    Projects 2D Slice Iteration comparisions up to 3D.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter value.

    contour_num: integer
        Amount of Contour lines on plot.
        

    Returns
    -------
    'Iter3d_ItName_FuncName_DkName_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8),constrained_layout=True)
    ax = plt.axes(projection ="3d")
    
    min_x1,max_x1,min_x2,max_x2,min_x3,max_x3=[],[],[],[],[],[]

    #Multiple Seeds on Same Plot
    for s in range(0, seed_num):

        x_bold =data_dict[('{0}'.format(it_name),'{0}'.format(test_func_name),'{0}'.format(dk_name),'{0}'.format(s), '{0}'.format(m))] #search dictionary for needed array
        
        #Take column of x_1 values
        it_x1 = x_bold[0:iter_num+1,0] 
        #Take column of x_2 values
        it_x2 = x_bold[0:iter_num+1,1] 
        #Take column of x_2 values
        it_x3 = x_bold[0:iter_num+1,2] 
        
        #Creates (1,seed_num) arrays with the minimum and maximum values for each x
        min_x1=np.append(min_x1, np.amin(it_x1))
        max_x1=np.append(max_x1, np.amax(it_x1)) 
        min_x2=np.append(min_x2, np.amin(it_x2)) 
        max_x2=np.append(max_x2, np.amax(it_x2)) 
        min_x3=np.append(min_x3, np.amin(it_x3)) 
        max_x3=np.append(max_x3, np.amax(it_x3)) 

    #Mean Trajectory on Same Plot
    x_mean=np.array(tensor(it_name,test_func, test_func_name,dk_name,seed_num,m,iter_num)[1]) 

    it_mx1 = x_mean[0:iter_num+1,0] 
    it_mx2 = x_mean[0:iter_num+1,1] 
    it_mx3 = x_mean[0:iter_num+1,2] 

    #Returns scalar minium and maximum values for each x
    x1_range_min=np.minimum(np.amin(min_x1),minima[0])
    x1_range_max=np.maximum(np.amax(max_x1),minima[0]) 
    x2_range_min=np.minimum(np.amin(min_x2),minima[1])
    x2_range_max=np.maximum(np.amax(max_x2),minima[1])
    x3_range_min=np.minimum(np.amin(min_x3),minima[2]) 
    x3_range_max=np.maximum(np.amax(max_x3),minima[2])


    #Multiple Seeds on Same Plot
    for s in range(0, seed_num):

        x_bold =data_dict[('{0}'.format(it_name),'{0}'.format(test_func_name),'{0}'.format(dk_name),'{0}'.format(s), '{0}'.format(m))] #search dictionary for needed array

        it_x1 = x_bold[0:iter_num+1,0] 
        it_x2 = x_bold[0:iter_num+1,1] 
        it_x3 = x_bold[0:iter_num+1,2] 

        ax.scatter(it_x1,it_x2,zs=x3_range_min-.25,cmap = plt.cm.get_cmap(sequential_color_maps[((s+1)%len(sequential_color_maps))]),  marker = '.',zdir='z')#Plotting the iterations with resulting point values
        ax.scatter(it_x1,it_x3,zs=x2_range_max+.25,cmap = plt.cm.get_cmap(sequential_color_maps[((s+1)%len(sequential_color_maps))]),  marker = '.',zdir='y')#Plotting the iterations with resulting point values
        ax.scatter(it_x2,it_x3,zs=x1_range_min-.25,cmap = plt.cm.get_cmap(sequential_color_maps[((s+1)%len(sequential_color_maps))]),  marker = '.',zdir='x')#Plotting the iterations with resulting point values


    ax.scatter(it_mx1,it_mx2,zdir='z',zs=x3_range_min-.2,  cmap= sequential_color_maps[0], c=iter_k,  marker = '*', s= 35, zorder=2,label ='Average Trajectory')#Plotting the iterations with resulting point values
    ax.scatter(it_mx1,it_mx3,zs=x2_range_max+.2,zdir='y', cmap= sequential_color_maps[0], c=iter_k,  marker = '*', s= 35,zorder=2)#Plotting the iterations with resulting point values
    ax.scatter(it_mx2,it_mx3,zs=x1_range_min-.2, zdir='x', cmap= sequential_color_maps[0], c=iter_k,  marker = '*', s= 35,zorder=2)#Plotting the iterations with resulting point values

    #Provides values for x_1 over a range
    x_1 = np.linspace((x1_range_min-.25),(x1_range_max+.25),contour_num) 
    #Provides values for x_2 over a range
    x_2 = np.linspace((x2_range_min-.25),(x2_range_max+.25),contour_num) 
    #Provides values for x_3 over a range
    x_3 = np.linspace((x3_range_min-.25),(x3_range_max+.25),contour_num) 

    X,Y,Z=[],[],[] 
    
    X1,X2 = np.meshgrid(x_1, x_2) #Provides us with arrays so we can find every possible combination for x_1,x_2
    
    for q in range(0,contour_num):
        for p in range(0,contour_num):
            z_points=[1]*n
            z_points[0]=X1[0,p]
            z_points[1]= X2[q,0]
            Z=np.append(Z,test_func(z_points)) 
    Z=Z.reshape(contour_num,contour_num)
    
    ax.contour(X1,X2,Z, contour_num, cmap = 'jet',zdir='z',offset=x3_range_min-.25) #Creates contour lines on plot

    X1,X3 = np.meshgrid(x_1,x_3) #Provides us with arrays so we can find every possible combination for x_1,x_2

    for q in range(0,contour_num): #take every possible combinition of q & p for mesh grid
        for p in range(0,contour_num):
            y_points=[1]*n
            y_points[0]=X1[0,p]
            y_points[2]= X3[q,0]
            Y=np.append(Y,test_func(y_points)) #Taking the test function values for mesh grid
    Y=Y.reshape(contour_num,contour_num)
    
    ax.contour(X1,Y,X3, contour_num, cmap = 'jet',zorder=-1,zdir='y',offset=x2_range_max+.25) #Creates contour lines on plot

    X2,X3 = np.meshgrid(x_2, x_3) #Provides us with arrays so we can find every possible combination for x_1,x_2

    for q in range(0,contour_num): #take every possible combinition of q & p for mesh grid
        for p in range(0,contour_num):
            x_points=[1]*n
            x_points[1]=X2[0,p]
            x_points[2]= X3[q,0]
            X=np.append(X,test_func(x_points)) #Taking the test function values for mesh grid
    X=X.reshape(contour_num,contour_num)
    
    ax.contour(X,X2,X3, contour_num, cmap = 'jet',zorder=-1,zdir='x', offset=x1_range_min-.25) #Creates contour lines on plot
    
    #Plotting starting point
    ax.plot(x[0],x[1],color='red', marker = 'X',markersize=15,zdir='z',zs=x3_range_min-.25)     
    ax.plot(x[0],x[2],color='red', marker = 'X',markersize=15,zdir='y',zs=x2_range_max+.25) 
    ax.plot(x[1],x[2],color='red', marker = 'X',markersize=15,zdir='x',zs=x1_range_min-.25) 

    #Plotting minima on respective planes
    ax.plot(minima[0],minima[1],color='gold', marker = '*',markersize=20, zdir='z',zs=x3_range_min-.25)
    ax.plot(minima[0],minima[2],color='gold', marker = '*',markersize=20, zdir='y',zs=x2_range_max+.25)
    ax.plot(minima[1],minima[2],color='gold', marker = '*',markersize=20, zdir='x',zs=x1_range_min-.25)

    #Set Limits
    ax.set_xlim(x1_range_min-.25, x1_range_max+.25)
    ax.set_ylim(x2_range_min-.25, x2_range_max+.25)
    ax.set_zlim(x3_range_min-.25, x3_range_max+.25)
    
    #Set Title and Subtitle
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize="large") #Set title of plot with number of iterationsk
    ax.set_title('Comparsion Along Seeds and Iterations with {} Iterations and {} Seeds'.format( iter_num,seed_num), fontsize='medium') #Set Subtitle
    
    #Label Axis
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    
    #Add legend
    plt.legend(fontsize='x-large')
    
    #Save figure image to directory
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_{}_{}_{}_{}.png".format('Iter3d',
                                                                            it_name,
                                                                            test_func_name,
                                                                            dk_name,
                                                                            iter_num,
                                                                            seed_num,
                                                                            np.format_float_scientific(m,trim='-'))),dpi=100, bbox_inches='tight')
    
    #Close out plot
    plt.close('all') 

## Plot: Closer look at Iteration Comparison
def iter_zoom(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num,seed_num,minima,m, contour_num=50):
    """
    Produce plot with close observation around mimina or chosen region.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.
    
    chosen_pair : (1,2) array of indices
        The chosen 2D slice.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter value.

    contour_num: integer
        Amount of Contour lines on plot.
        

    Returns
    -------
    'Iterzoom_ItName_FuncName_DkName_ChosenPair_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8), constrained_layout= True)
    ax = fig.add_subplot(1, 1, 1) 
    
    min_x1,max_x1,min_x2,max_x2=[],[],[],[]
  
    #Mean Trajectory 
    x_mean=np.array(tensor(it_name,test_func, test_func_name,dk_name,seed_num,m,iter_num)[1]) 
    
    #Look at two dimensional slice
    x_mean_slice = x_mean[0:iter_num+1,chosen_pair] 
        
    it_mx1 = x_mean_slice[0:iter_num+1,0] 
    it_mx2 = x_mean_slice[0:iter_num+1,1]
        
    #Arrows needed for quiver plot
    arrow_mx1 = it_mx1[1:] - it_mx1[:-1]
    arrow_mx2 = it_mx2[1:] - it_mx2[:-1] 
    
    
    ax.scatter(it_mx1,it_mx2, cmap= sequential_color_maps[0], c=iter_k,  marker = '*', s= 35, label ='Average Trajectory')#Plotting the iterations with resulting point values
    ax.quiver(it_mx1[:-1], it_mx2[:-1], arrow_mx1, arrow_mx2, scale_units = 'xy', angles = 'xy', scale = 1, color =solid_colors[0], alpha=.15) #Plotting arrows with intermediate values

    #Returns scalar minimum and maximum values for x
    x1_range_min=np.minimum(np.amin(it_mx1),minima[0])
    x1_range_max=np.maximum(np.amax(it_mx1),minima[1]) 
    x2_range_min=np.minimum(np.amin(it_mx2),minima[0]) 
    x2_range_max=np.maximum(np.amax(it_mx2),minima[1])


    #Provides values for x_1 over a range
    x_1 = np.linspace((x1_range_min-.25),(x1_range_max+.25),contour_num) 
    #Provides values for x_2 over a range
    x_2 = np.linspace((x2_range_min-.25),(x2_range_max+.25),contour_num) 
    #Provides us with arrays so we can find every possible combination for x_1,x_2
    X1,X2 = np.meshgrid(x_1, x_2) 


    X=[] 
    
    #For each value on our grid
    for q in range(0,contour_num): 
        for p in range(0,contour_num):
            x_points=minima
            x_points[chosen_pair[0]]=X1[0,p]
            x_points[chosen_pair[1]]= X2[q,0]
            
            #Taking the test function values for mesh grid
            X=np.append(X,test_func(x_points)) 
    #Reshape to array of size (contour number, contour number)
    X=X.reshape(contour_num,contour_num)

    #Creates contour lines on plot
    ax.contour(X1,X2,X, contour_num, cmap = 'jet',zorder=-1) 
    
    #Plotting starting point 
    ax.plot(x[chosen_pair[0]],x[chosen_pair[1]],color='red', marker = 'X',markersize=15)  
    #Plotting minima
    ax.plot(minima[chosen_pair[0]],minima[chosen_pair[1]],color='gold', marker = '*',markersize=20) 
    
    #Set Axis Limits
    ax.set_xlim(minima[chosen_pair[0]]-.1, minima[chosen_pair[0]]+1)
    ax.set_ylim(minima[chosen_pair[1]]-.1, minima[chosen_pair[1]]+1)
    
    #Plot Title and Subtitle
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize="large") #Set title of plot with number of iterationsk
    ax.set_title('Comparsion Along Seeds and Iterations with {} Iterations and {} Seeds'.format( iter_num,seed_num), fontsize='medium')
    
    #Set Axis Label
    plt.xlabel("$x_{}$".format(chosen_pair[0]))
    plt.ylabel("$x_{}$".format(chosen_pair[1])) 
    
    #Add legent
    plt.legend(fontsize='x-large') 
    
    #Save Figure Image to Directory
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_{}_x{}x{}_{}_{}_{}.png".format('IterZoom',
                                                                                   it_name,
                                                                                   test_func_name,
                                                                                   dk_name,
                                                                                   chosen_pair[0],
                                                                                   chosen_pair[1],
                                                                                   iter_num,
                                                                                   seed_num,
                                                                                   np.format_float_scientific(m,trim='-'))),dpi=100) 
    
     #Close out plot since it is saved
    plt.close('all')

# Plot: Standard Deviation and Mean Trajectories
def std_dir_comp(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num,seed_num,minima,m,contour_num):
    """
    Produce plot with mean trajectory and standard deviation two dimensional confidence intervals.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.
    
    chosen_pair : (1,2) array of indices
        The chosen 2D slice.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter value.

    contour_num: integer
        Amount of Contour lines on plot.
        

    Returns
    -------
    'StdDir_ItName_FuncName_DkName_ChosenPair_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1) 
    
    min_x1,max_x1,min_x2,max_x2=[],[],[],[]
    min_x1_std, max_x1_std, min_x2_std, max_x2_std= [],[],[],[]  
        
    #Mean Trajectory on Same Plot
    x_mean=np.array(tensor(it_name,test_func, test_func_name,dk_name, seed_num,m,iter_num)[1])
    
    #Select two dimensional slice 
    x_mean_slice = x_mean[0:iter_num+1,chosen_pair]     

    it_mx1 = x_mean_slice[0:iter_num+1,0] 
    it_mx2 = x_mean_slice[0:iter_num+1,1] 
    
    #Create (1, j*seed_num) arrays with minimum and maximum values for mean x
    min_x1=np.append(min_x1, np.amin(it_mx1)) 
    max_x1=np.append(max_x1, np.amax(it_mx1)) 
    min_x2=np.append(min_x2, np.amin(it_mx2)) 
    max_x2=np.append(max_x2, np.amax(it_mx2))
    
    #Plotting the iterations with resulting point values
    ax.scatter(it_mx1,it_mx2,cmap= sequential_color_maps[3], c=iter_k, marker = mark_array[1], label ='Average Trajectory for {0}'.format(dk_name))

     #Derive stardard deviation from tensor
    x_std=np.array(tensor(it_name,test_func, test_func_name,dk_name, seed_num,m,iter_num)[2])
    #Select two dimensional slice 
    x_std_slice=x_std[0:iter_num+1,chosen_pair]  
    
    #Get bottom limit for variation by subtracting std from x_bold
    x_min=x_mean-x_std 
    #Get top limit for variation by adding std to x_bold
    x_max=x_mean+x_std

    x_min_slice = x_min[0:iter_num+1,chosen_pair]    
    x_max_slice = x_max[0:iter_num+1,chosen_pair]   

    #Extract limits for plot
    x1_bottom= x_min_slice[0:iter_num+1,0]
    x2_bottom = x_min_slice[0:iter_num+1,1]  
    x1_top= x_max_slice[0:iter_num+1,0] 
    x2_top = x_max_slice[0:iter_num+1,1] 
    
    #Create (1, j*seed_num) arrays with minimum and maximum values for std x
    min_x1_std=np.append(min_x1_std, np.amin(x1_bottom)) 
    max_x1_std=np.append(max_x1_std, np.amax(x1_top)) 
    min_x2_std=np.append(min_x2_std, np.amin(x2_bottom)) 
    max_x2_std=np.append(max_x2_std, np.amax(x2_top)) 

    #Plotting rectangles for standard deivation
    #For each iteration
    for k in range(0,iter_num,5):
        ax.add_patch(Rectangle(xy=(x1_bottom[k],x2_bottom[k]) ,width=x_std_slice[k,0]*2, height=x_std_slice[k,1]*2, facecolor=solid_colors[3],
                 edgecolor='None', alpha=0.25, fill=True,zorder=-2)) 

    #Find minimum and maximum of each element of x_bold
    x1_range_min=min(np.amin(min_x1),minima[0],np.amin(min_x1_std)) 
    x1_range_max=max(np.amax(max_x1),minima[1],np.amax(max_x1_std)) 
    x2_range_min=min(np.amin(min_x2),minima[0],np.amin(min_x2_std)) 
    x2_range_max=max(np.amax(max_x2),minima[1],np.amax(max_x2_std))
    
    #Provides values for x_1 over a range
    x_1 = np.linspace((x1_range_min-.25),(x1_range_max+.25),contour_num) 
    #Provides values for x_2 over a range
    x_2 = np.linspace((x2_range_min-.25),(x2_range_max+.25),contour_num) 
    #Provides us with arrays so we can find every possible combination for x_1,x_2
    X1,X2 = np.meshgrid(x_1, x_2) 


    X=[] 
    
    #For each value on our grid
    for q in range(0,contour_num): 
        for p in range(0,contour_num):
            x_points=minima
            x_points[chosen_pair[0]]=X1[0,p]
            x_points[chosen_pair[1]]= X2[q,0]
            
            #Taking the test function values for mesh grid
            X=np.append(X,test_func(x_points)) 
    #Reshape to array of size (contour number, contour number)
    X=X.reshape(contour_num,contour_num)

    #Creates contour lines on plot
    ax.contour(X1,X2,X, contour_num, cmap = 'jet',zorder=-1) 
    
    #Plotting starting point 
    ax.plot(x[chosen_pair[0]],x[chosen_pair[1]],color='red', marker = 'X',markersize=15)  
    #Plotting minima
    ax.plot(minima[chosen_pair[0]],minima[chosen_pair[1]],color='gold', marker = '*',markersize=20) 
    
    #Plot Title and Subtitle
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize='large')
    ax.set_title('Comparision of Standard Deivation along Average Trajectories with {} Iterations and {} Seeds'.format( iter_num,seed_num), fontsize='medium')
    
    #Set Axis Label
    plt.xlabel("$x_{}$".format(chosen_pair[0])) 
    plt.ylabel("$x_{}$".format(chosen_pair[1])) 
    
    #Add legend
    plt.legend(fontsize='x-large') 
    
    #Save Figure Image to Directory
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_x{}x{}_{}_{}_{}.png".format('StdDir',
                                                                                it_name,
                                                                                test_func_name,
                                                                                chosen_pair[0],
                                                                                chosen_pair[1], 
                                                                                iter_num,
                                                                                seed_num,
                                                                                np.format_float_scientific(m,trim='-'))),dpi=100) #Save plot 
    
    #Close out plot
    plt.close('all')  

def manual_cmap(k,iter_num):
    """
    Normalized colormap for edgecolor compatitibility.

    Parameters
    ----------
    k : array-like, shape (n, )
        The starting point.

    iter_num : integer
        The number of iterations.       

    Returns
    -------
    rgb_value
    """
    rgb_value= plt.cm.gist_rainbow_r((np.clip(k,0,iter_num)-0)/iter_num )
    return rgb_value

    
# Provides Covariance and Mean Trajectories for Both Iterative Methods
def covar_dir_comp(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num,seed_num,minima,m, contour_num):
    """
    Produce plot with covariance confidence ellipses for several points along iterations.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.
    
    chosen_pair : (1,2) array of indices
        The chosen 2D slice.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter.

    contour_num: integer
        Amount of Contour lines on plot.
        

    Returns
    -------
    'Cov_ItName_FuncName_DkName_ChosenPair_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1) 
    
    min_x1,max_x1,min_x2,max_x2=[],[],[],[] 

    #Mean Trajectory on Same Plot
    x_mean=np.array(tensor(it_name,test_func, test_func_name,dk_name, seed_num,m,iter_num)[1]) 
    #Select two dimensional slice
    x_mean_slice = x_mean[0:iter_num+1,chosen_pair]     


    it_mx1 = x_mean_slice[0:iter_num+1,0] 
    it_mx2 = x_mean_slice[0:iter_num+1,1] 

    #Create (1,seed_num) arrays for minimum and maximum values for each component of x
    min_x1=np.append(min_x1, np.amin(it_mx1)) 
    max_x1=np.append(max_x1, np.amax(it_mx1)) 
    min_x2=np.append(min_x2, np.amin(it_mx2)) 
    max_x2=np.append(max_x2, np.amax(it_mx2)) 

    ax.scatter(it_mx1,it_mx2,cmap= sequential_color_maps[3], c=iter_k, marker = mark_array[0], label ='Average Trajectory for {0}'.format(dk_name))

    #Derive stardard deviation from tensor
    x_std=np.array(tensor(it_name,test_func, test_func_name,dk_name, seed_num,m,iter_num)[2]) 
     #Select two dimensional slice 
    x_std_slice=x_std[0:iter_num+1,chosen_pair]  
    
    min_ell_x1,min_ell_x2, max_ell_x1,max_ell_x2 =[],[],[],[]

    #Plotting covariance ellipses 
    #Currently set for each 25th point, as it is very time consuming, but can be adjusted
    for k in range(0,iter_num,25):
        x_cov=np.array(tensor(it_name,test_func, test_func_name,dk_name, seed_num,m,iter_num)[0])
        x_bold_cov= x_cov[0:iter_num+1,chosen_pair] 

        x_1=x_bold_cov[k,0]
        x_2=x_bold_cov[k,1]
        cov = np.cov(x_1, x_2)
        if np.sqrt(cov[0, 0] * cov[1, 1])== 0:
            pass
        else:    
            con_ell=confidence_ellipse(x_bold_cov[k,0], x_bold_cov[k,1], ax, facecolor='None', edgecolor= manual_cmap(k,iter_num), linewidth = 2, alpha=1, fill=True,zorder=-1)
            con_ell[0]

            #Create (1,seed_num) arrays for minimum and maximum values for each component of covariance of x
            min_ell_x1=np.append(min_ell_x1, np.amin(con_ell[1])) 
            min_ell_x2=np.append(min_ell_x2, np.amin(con_ell[2])) 
            max_ell_x1=np.append(max_ell_x1, np.amax(con_ell[1])) 
            max_ell_x2=np.append(max_ell_x2, np.amax(con_ell[2])) 

    #Plotting starting point
    ax.plot(x[chosen_pair[0]],x[chosen_pair[1]],color='red', marker = 'X',markersize=15)   
    #Plot Minima/Solution
    ax.plot(minima[chosen_pair[0]],minima[chosen_pair[1]],color='gold', marker = '*',markersize=20) 

    #Set title/subtitle
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize='large') 
    ax.set_title('Covariance along Average Trajectories with {} Iterations and {} Seeds'.format( iter_num,seed_num), fontsize='medium')
    
    #Axis label
    plt.xlabel("$x_{}$".format(chosen_pair[0]))
    plt.ylabel("$x_{}$".format(chosen_pair[1]))
    
    #Add legend
    plt.legend(fontsize='x-large') 

    #Find minimum and maximum values for range of contour lines
    x1_range_min=min(np.amin(min_x1),minima[0])+np.amin(min_ell_x1)
    x1_range_max=max(np.amax(max_x1),minima[1])+np.amax(max_ell_x1)
    x2_range_min=min(np.amin(min_x2),minima[0])+np.amin(min_ell_x2)
    x2_range_max=max(np.amax(max_x2),minima[1])+np.amax(max_ell_x2)
    
    
    #Provides values for x_1 over a range
    x_1 = np.linspace((x1_range_min-.25),(x1_range_max+.25),contour_num) 
    #Provides values for x_2 over a range
    x_2 = np.linspace((x2_range_min-.25),(x2_range_max+.25),contour_num) 
    #Provides us with arrays so we can find every possible combination for x_1,x_2
    X1,X2 = np.meshgrid(x_1, x_2) 


    X=[] 

    for q in range(0,contour_num): 
        for p in range(0,contour_num):
            x_points=[1]*n
            x_points[chosen_pair[0]]=X1[0,p]
            x_points[chosen_pair[1]]= X2[q,0]
            X=np.append(X,test_func(x_points)) 
    X=X.reshape(contour_num,contour_num)

    #Creates contour lines on plot
    ax.contour(X1,X2,X, contour_num, cmap = 'jet',zorder=-2) 
    
    #Save figure image to directory
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_x{}x{}_{}_{}_{}.png".format('CovDir',it_name,test_func_name,chosen_pair[0],chosen_pair[1], iter_num,seed_num,np.format_float_scientific(m,trim='-'))),dpi=100) #Save plot 
    #Close out Plot   
    plt.close('all')
    
## Compasion between the mean trajectory for various M values

def means_mvalues(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num,seed_num,minima,m, contour_num):
    """
    Produce plot with mean trajectories for various M values for visual comparision.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.
    
    chosen_pair : (1,2) array of indices
        The chosen 2D slice.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    contour_num: integer
        Amount of Contour lines on plot.
        

    Returns
    -------
    'MValue_ItName_FuncName_DkName_ChosenPair_IterNum_SeedNum.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1) 
    
    min_x1,max_x1,min_x2,max_x2=[],[],[],[]
    
    #Multiple m_values on Same Plot
    
    #Compute Vector of M Values
    m_vector=nearest_m(m) 
    
    for q in range (0,len(m_vector)):
        m_value =m_vector[q] 
        
        #Retrieve mean x_bold for each m
        x_mean=np.array(tensor(it_name,test_func, test_func_name,dk_name, seed_num,m_value,iter_num)[1])            
        x_mean_slice = x_mean[0:iter_num+1,chosen_pair]   
        
        it_mx1 = x_mean_slice[0:iter_num+1,0] 
        it_mx2 = x_mean_slice[0:iter_num+1,1]  
        
        #Create an (1, j*seed_num) array with the minimum and maximum values for each x
        min_x1=np.append(min_x1, np.amin(it_mx1)) 
        max_x1=np.append(max_x1, np.amax(it_mx1)) 
        min_x2=np.append(min_x2, np.amin(it_mx2))
        max_x2=np.append(max_x2, np.amax(it_mx2))

        #Arrows needed for quiver plot
        arrow_mx1 = it_mx1[1:] - it_mx1[:-1] 
        arrow_mx2 = it_mx2[1:] - it_mx2[:-1] 
        ax.scatter(it_mx1,it_mx2, marker = mark_array[q+1], cmap = plt.cm.get_cmap(sequential_color_maps[q+1]),c=iter_k, label='M Value of {0}'.format(m_vector[q])) 
        ax.quiver(it_mx1[:-1], it_mx2[:-1], arrow_mx1, arrow_mx2, scale_units = 'xy', angles = 'xy', scale = 1,color= solid_colors[q+1], alpha = .1)
    
    #Finds scalar minimum and maximum values for each x value
    x1_range_min=np.minimum(np.amin(min_x1),minima[0]) 
    x1_range_max=np.maximum(np.amax(max_x1),minima[1]) 
    x2_range_min=np.minimum(np.amin(min_x2),minima[0]) 
    x2_range_max=np.maximum(np.amax(max_x2),minima[1])

    #Provides values for x_1 over a range
    x_1 = np.linspace((x1_range_min-.25),(x1_range_max+.25),contour_num) 
    #Provides values for x_2 over a range
    x_2 = np.linspace((x2_range_min-.25),(x2_range_max+.25),contour_num) 
    #Provides us with arrays so we can find every possible combination for x_1,x_2
    X1,X2 = np.meshgrid(x_1, x_2) 


    X=[] 
    
    #For each value on our grid
    for q in range(0,contour_num): 
        for p in range(0,contour_num):
            x_points=minima
            x_points[chosen_pair[0]]=X1[0,p]
            x_points[chosen_pair[1]]= X2[q,0]
            
            #Taking the test function values for mesh grid
            X=np.append(X,test_func(x_points)) 
    #Reshape to array of size (contour number, contour number)
    X=X.reshape(contour_num,contour_num)

    #Creates contour lines on plot
    ax.contour(X1,X2,X, contour_num, cmap = 'jet',zorder=-1) 
    
    #Plotting starting point 
    ax.plot(x[chosen_pair[0]],x[chosen_pair[1]],color='green', marker = 'X',markersize=15)  
    #Plotting minima
    ax.plot(minima[chosen_pair[0]],minima[chosen_pair[1]],color='gold', marker = '*',markersize=20) 
    
    #Plot Title and Subtitle
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize='large') 
    ax.set_title('M Values along Mean Trajectories with {} Iterations and {} Seeds'.format( iter_num,seed_num), fontsize='medium')   
    
    #Set Axis labels
    plt.xlabel("$x_{}$".format(chosen_pair[0])) 
    plt.ylabel("$x_{}$".format(chosen_pair[1])) 
    
    #Set Plot Legend
    plt.legend(fontsize='x-large')
  
    #Save plot image to directory
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_{}_x{}x{}_{}_{}.png".format('MValue',it_name,test_func_name,dk_name,chosen_pair[0],chosen_pair[1],iter_num,seed_num)),dpi=100) 
    #Close plot after saving
    plt.close('all') 

## Plot: Standard Deviation as Function of Iteration Count
def iter_std(x,it_name,test_func,test_func_name,dk_name,iter_num,seed_num,m):
    """
    Produce one dimensional plot that plots standard deviation for each component of x over iteration count.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter value.
        

    Returns
    -------
    'IterStd_ItName_FuncName_DkName_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1) 
    
    #Retrieve std for certain condiitions from tensor
    x_std=np.array(tensor(it_name,test_func, test_func_name,dk_name,seed_num,m,iter_num)[2])
         
    #For each element in x_bold
    for x_n in range(0,n):
        xn_std = x_std[0:iter_num+1,x_n]  
        ax.plot(iter_k,xn_std, color= solid_colors[x_n+3], label ='Standard Deviation for x_{}'.format(x_n))
    
    #Set title and subtitle for plot
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize="large") 
    ax.set_title('Standard Deviation as a Function of Iteration Count', fontsize='medium') 
    
    #Set Axis Labels
    plt.xlabel("$Iteration Count$") #Set x-axis label
    plt.ylabel("$Standard Deviation$") #Set y-axis label
    
    #Add legend to plot
    plt.legend(fontsize='x-large')
    
    #Add Gridlines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    #Save figure image to directory
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_{}__{},{},{}.png".format('IterStd',
                                                                             it_name,
                                                                             test_func_name,
                                                                             dk_name,
                                                                             iter_num,
                                                                             seed_num,
                                                                             np.format_float_scientific(m,trim='-'))),dpi=100) 
    #Close out plot since it is saved
    plt.close('all') 

## Plot: Average Solution as Function of Iteration Count
def iter_fx(x,it_name,test_func,test_func_name,dk_name,iter_num,seed_num,m):
    """
    Produce one dimensional plot that plots the average solution value of x over iteration count.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter value.
        

    Returns
    -------
    'IterSol_ItName_FuncName_DkName_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))

    #Reshape the figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1) 
     
#     for j in range(0,len(dk_names)):
#         dk_name=dk_names[j] #Cycle through direction methods
    
    #Retrieve mean solution from tensor
    fx=np.array(tensor(it_name,test_func, test_func_name,dk_name,seed_num,m,iter_num)[4][0]) 
    ax.plot(iter_k,fx, color= solid_colors[3], label ='Average Function Values for {}'.format(dk_name))
    
    #Set title and subtitle for plot
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize="large") 
    ax.set_title('Average Solution as a Function of Iteration Count', fontsize='medium') 
    
    #Set Axis labels
    plt.xlabel("Iteration Count")
    plt.ylabel("Average Solution")
    
    #Add legend tp plot
    plt.legend(fontsize='x-large')
    
    #Add Gridlines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    #Save figure image to directory
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_{},{},{}.png".format('IterSol',it_name,test_func_name,iter_num,seed_num,np.format_float_scientific(m,trim='-'))),dpi=100) #Save plot 
    
    #Close out plot since it is saved
    plt.close('all') 

## Plot: Calculating the Distance between Points then Mean Tracjectory and Vice Versa
def mean_diff(x,it_name,test_func,test_func_name,dk_name,iter_num,seed_num,m):
    """
    Produce one dimensional plot that plots Distance between Points then Mean Tracjectory and Vice Versa over iteration count.

    Parameters
    ----------
    x : array-like, shape (n, )
        The starting point.

    it_name : string
        The name to be assigned to image file name and titles for the chosen optimization function.

    test_func : function
        The chosen test function.
        
    test_func_name : string
        The name to be assigned to image file name and titles for the chosen test function.
        
    dk_name : string
        The name to be assigned to image file name and titles for the chosen scaled direction function.

    iter_num : integer
        The number of iterations.
        
    seed_num : integer
        The number of seeds which is the number of runs/trajectories.
        
    minima : n size array
        The minima of the function if known.

    m : float
        Input parameter value.
        

    Returns
    -------
    'MeanDiff_ItName_FuncName_DkName_IterNum_SeedNum_M.png'
    """
    n=len(x)
    
    #Make list of all k values
    iter_k=list(range(0,iter_num))
    
    #Reshape the figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1) 
    
     #Retrieve mean  from tensor
    x_mean=np.array(tensor(it_name,test_func, test_func_name,dk_name,seed_num,m,iter_num)[1])
    x_mean_diff=[0]
    
    for x in range(0,iter_num-1):
        x_diff=np.linalg.norm(x_mean[x+1] - x_mean[x])
        x_mean_diff= np.append(x_mean_diff,x_diff)
    
    x_bold_diff=[]
    
    #For all seeds
    for s in range(0,seed_num):
            diff_list=[]
            x_diff=[0]
            x_bold =data_dict[('{0}'.format(it_name),'{0}'.format(test_func_name),'{0}'.format(dk_name),'{0}'.format(s),'{0}'.format(m))]
            
            #Find the solution value for each iteration for each seed
            for k in range(0,iter_num-1): 
                diff=np.linalg.norm(x_bold[k+1]-x_bold[k]) 
                x_diff.append(diff)
            diff_list.append(x_diff)
            
    #Returns (iter_num,n,seed_num) array 
    x_bold_diff = np.dstack((diff_list)) 
    
    #Returns (iter_num,n) array for the average trajectory
    x_diff_mean = [np.mean(k,axis=1) for k in x_bold_diff] 
    
    #Plotting the iterations with resulting point values
    ax.plot(iter_k,x_mean_diff, linewidth=2, color= solid_colors[3], label ='Mean Trajectory then Distance')
    ax.plot(iter_k,x_diff_mean[0], linewidth=2, color= solid_colors[5], label ='Distances then Mean Distances')
    
    #Set title and subtitle
    plt.suptitle('{} Using the {} Iterative Algorithm:'.format( test_func_name, it_name),y=.95, fontsize="large")
    ax.set_title('Mean Distance Between Points as Function of Iteration Count', fontsize='medium') 
    
    #Set Axis Labels
    plt.xlabel("Iteration Count") 
    plt.ylabel("Difference")
    
    #Add legend
    plt.legend(fontsize='x-large')
    
    #Add grid lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    #Save figure image to directory
    plt.savefig(os.path.join(path_Figures,"{}_{}_{}_{}__{},{},{}.png".format('MeanDiff',
                                                                             it_name,
                                                                             test_func_name,
                                                                             dk_name,
                                                                             iter_num,
                                                                             seed_num,
                                                                             np.format_float_scientific(m,trim='-'))),dpi=100) #Save plot 
    
    #Close out figures
    plt.close('all') 


#Produce all visual plots for each of the test functions
def visual_analysis(x,test_func,test_func_name, it_name, scaled_dir, dk_name, iter_num, seed_num,minima,m=1.0, contour_num=50):
    
    n=len(x)
    #Provides a list of indexes for x_bold
    x_list=list(range(0,int(n))) 

    #Provides all possible combination of element indices
    x_pairs= list(combinations(x_list, 2)) 

    #For all potential combinations of 2-D slices 
    for p in range(0,len(x_pairs)):
        
        chosen_pair=x_pairs[p]
        
        #Run all functions to produce visual plots
        gen_visual(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num,seed_num,minima,m, contour_num)
        iter_comp(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num, seed_num,minima,m, contour_num)
        iter_zoom(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num, seed_num,minima,m, contour_num)
        std_dir_comp(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num, seed_num,minima,m,contour_num)
        covar_dir_comp(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num, seed_num,minima,m, contour_num)
        means_mvalues(x,it_name,test_func,test_func_name,dk_name,chosen_pair,iter_num, seed_num,minima,m, contour_num)
        iter_std(x,it_name,test_func,test_func_name,dk_name,iter_num,seed_num,m)
        iter_fx(x,it_name,test_func,test_func_name,dk_name,iter_num,seed_num,m)
        mean_diff(x,it_name,test_func,test_func_name,dk_name,iter_num,seed_num,m)

        #If it is a 3D function
        if n==3:
            iter_comp_3d(x,it_name,test_func,test_func_name,dk_name,iter_num, seed_num,minima,m, contour_num)





