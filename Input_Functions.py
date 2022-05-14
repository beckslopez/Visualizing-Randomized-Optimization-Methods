# Visualizing & Understanding Performance of Randomized Optimization Methods: List of Potenticial Input Functions

# Imported Packages
import numpy as np

#N Dimensional Chained Rosenbrock Function
def chained_rosen(x):
    n=len(x)
    if (n%2)== 0: #if n is an even dimension
        f_x=[] #Begin with an empty array
        for i in range(0,int((n/2))): 
            f_x=np.append(f_x, 100*(x[2*i]**2-x[2*i+1])**2 +(x[2*i]-1)**2) #Add new element to array, reminder that python has index 0 which is equivalent to x_1 so we substract an extra 1 to get proper values  
        return sum(f_x) #Returns scalar, final value is the summation of each element in the array
    else: #If n is not an even dimension, it will not run the function,
        print("For this function, n must be an even dimension.") 

#N Dimensional Alternate Chained Rosenbrock Function  
def alt_rosen(x):
    n=len(x)
    f_x=[] #Begin with an empty array
    for i in range(0,int(n-1)): 
        f_x=np.append(f_x, 100*(x[i]**2-x[i+1])**2 +(x[i]-1)**2) #Add new element to array, reminder that python has index 0 which is equivalent to x_1 so we substract an extra 1 to get proper values  
    return sum(f_x) #Returns scalar, final value is the summation of each element in the array

#N Dimensional Convex Nestrov Quadratic Function
def convex_nev(x):
    n=len(x)
    f_x=[] #Begin with an empty array
    for i in range(0,int(n-1)):
        f_x= np.append(f_x,(x[i]-x[i+1])**2) #Summation componenet of the function
    f_n = (1/2)*(x[0]**2 + sum(f_x))-x[0] #Add the elements in the array to the first components of the function
    return f_n #Returns scalar

#N Dimensional Convex Funnel Function   
def convex_funn(x):
    n=len(x)
    f_x=[] #Begin with an empty array
    for i in range(0,int(n)):
        f_x= np.append(f_x,(x[i]-1)**2) #Summation component of the function where python has a 0 index equal to x_1
    f_n = np.log(1+10* sum(f_x)) #Add the elements in the array to the first components of the function
    return f_n #Returns scalar

def ackley(x):
    n=len(x)
    #returns the point value of the given coordinate
    c1_sum=sum(tuple(a ** 2 for a in x ))
    c1 = -0.2*np.sqrt((1.0/n)*c1_sum)
    c2_sum= sum(tuple(np.cos(2*np.pi*b) for b in x))   
    c2 = (1.0/n)*c2_sum
    f = np.exp(1) + 20 - 20*np.exp(c1) - np.exp(c2)
    #returning the value
    return f


# # Gradient Functions 

#Gradient Calculation of Chained Rosenbrock Function
def gr_chained_rosen(x): 
    n=len(x)
    grad_cr=[] 
    if (n%2)== 0: #If n is an even dimension
        for i in range(0,int(n/2+1)): 
            if (i%2)== 0: #If the index is even
                grad_cr = np.append(grad_cr, 400*x[i]**3-400*x[i]*x[i+1] +2*x[i]-2) #Add new element to array for even indices  
            else: #If the index is odd
                grad_cr = np.append(grad_cr, 200*x[i] - 200*x[i-1]**2 ) #Add new element to array for odd indices
    else: #If n is not an even dimension, it will not run the function,
        print("For this function, n must be an even dimension.") 
    return grad_cr #Returns a (n,) array

#Gradient Calculation of Alternative Chained Rosenbrock Function
def gr_alt_rosen(x,n):
    n=len(x)
    grad_ar=[] #Begin with an empty array
    for i in range(0,int(n)): 
        if i==0: #For element x_1
            grad_ar = np.append(grad_ar, 400*x[i]**3-400*x[i]*x[i+1]+2*x[i]-2) #Add new element to array for x_1
        elif i >0 and i<(n-1):
            grad_ar = np.append(grad_ar, 400*x[i]**3 -400*x[i]*x[i+1]+202*x[i]- 200*x[i]**2 -2) #Add new element to gradient array
        else:#For element x_n                        
            grad_ar = np.append(grad_ar, 200*x[i] - 200*x[i-1]**2 ) #Add new element to gradient array
    return grad_ar #Returns a (n,) array

#Gradient Calculation of Convex Nestrov Quadratic Function
def gr_convex_nev(x,n):
    n=len(x)
    grad_cn=[] #Begin with an empty array
    for i in range(0,int(n)):
        if i==0: #For element x_1
            grad_cn = np.append(grad_cn,2*x[i]-x[i+1]-1) #Add new element to gradient array
        elif i >0 and i<(n-1): 
            grad_cn = np.append(grad_cn,2*x[i]-x[i-1]-x[i+1]) #Add new element to gradient array
        else: #For element x_n   
            grad_cn = np.append(grad_cn,x[i]-x[i-1]) #Add new element to gradient array
    return grad_cn #Returns a (n,) array

#Gradient Calculation of Convex Funnel Function

def sign_x(x_i):#Vector of signs 
    if (x_i-1)>0: #If greater than 0 then return +1
        return 1
    elif (x_i-1) <0: #If less than 0 then return -1
        return -1 
    else: #If (x_i-1) returns 0, return an error
        print("For this function, one of the conditions is that x is not equal to 1!")
              
def gr_convex_funn(x):
    n=len(x)
    signs=[] #Begin with an empty array
    f_x=[] #Begin with an empty array
    
    for i in range(0,int(n)):
        f_x= np.append(f_x,(x[i]-1)**2) #Summation component of the function where python has a 0 index equal to x_1
        
    for i in range(0,int(n)):   
        signs = np.append(signs,sign_x(x[i])) #Create vector of signs (pos vs neg)
   
    grad_cf = 10/(1+10*(sum(f_x)))*signs #Gradient of f with respect to x
    return grad_cf #Returns a (n,) array

def gr_ackley(x):
    n=len(x)
    g = np.zeros(n)

    eps = np.finfo(float).eps

    c1_sum=sum(tuple(a ** 2 for a in x ))
    c1 = np.maximum(eps,np.sqrt((1.0/n)*c1_sum))
    c2_sum= sum(tuple(np.cos(2*np.pi*b) for b in x))                         
    c2 = (1.0/n)*c2_sum


    for i in range(n):
        g[i] += 4*x[i]*np.exp(-0.2*c1)/(n*c1)
        g[i] += (2*np.pi/n)*np.sin(2*np.pi*x[i])*np.exp(c2)

    return g


# ## Method for Calculating Randomized Scaled Direction

def standard_dk(x,gr_test_func):
    n=len(x)
    gradient = gr_test_func(x) #Returns (n,) array for gradient
    xi_k = np.random.normal(0, 1,size =(n,1))#Returns (n,1) array, where xi_k to be a random vector with a normal distribution where the mean is 0 with a standard deviation of 1 
    gradient = gradient.reshape(n,1) #Returns a (n,1) array
    t_gradient = gradient.T #Returns a (1,n) array which is the transpose of the gradient
    d_k = (np.matmul(t_gradient, xi_k))* xi_k #Calculate scaled direction d_k
    d_k = np.squeeze(d_k,axis=1) #Returns (n,1) array 
    return d_k 

def rcd_dk(x,gr_test_func):
    n=len(x)
    gradient = gr_test_func(x) #Returns (n,) array for gradient
    xi_k = np.random.randint(0, n)#Returns random integer from 1 to n
    e_k = np.identity(n)[:,xi_k] #Returns (n,) array which is the xi_k indexed column from the identity matrix of size n
    e_k = e_k.reshape(n,1)#Returns (n,1) array
    gradient = gradient.reshape(n,-1) #Returns a (n,1) array
    t_gradient = gradient.T #Returns a (1,n) array which is the transpose of the gradient
    d_k = (np.matmul(t_gradient, e_k))* e_k #Calculate scaled direction d_k
    d_k = np.squeeze(d_k,axis=1)#Returns (n,1) array 
    return d_k 


# ## N Dimensional Itertive Optimization Algorithm

def standard_it(iter,x,s,test_func,gr_test_func,scaled_dir,m): 
# iter: number of iterations 
# n: number of dimensions
# x: starting point input as array
# s: chosen seed
# test_func: chosen test function
# gr_test_func: gradient function for chosen test function
# scaled_dir: method chosen for calculating the scaled direction
# m: value to input to alpha_l
    n=len(x)
    k = 0 #Begin at k=0 
    np.random.seed(s) #Assign a chosen seed
    it_x, it_k, it_f = np.empty(0),np.empty(0), np.empty(0) #Create empty arrays to record ending points
    while k < iter: #While k remains under the number of iterations, continue to run
        it_x = np.append(it_x, x) #Add first element of new point to list
        
        alpha_k = m/(k+1) #Returns a scalar for our non negative stepsize along d_k
        d_k = scaled_dir(x,gr_test_func)#Returns (n,1) array for the randomized scaled direction 
        x= x - alpha_k*d_k #Returns (n,1) array for running a standard iterative optimization algorithm
        k +=1 #Add one to k
    
    return (it_x.reshape(iter,n)) #Returns [iter,n] array, which is an array of points for trajectory values

def adam(iter, x, s, test_func, gr_test_func, scaled_dir, m):
    # iter: number of iterations
    # n: number of dimensions
    # x: starting point input as array
    # s: chosen seed
    # test_func: chosen test function
    # gr_test_func: gradient function for chosen test function
    # scaled_dir: method chosen for calculating the scaled direction
    # m: value to input to alpha_l
    n=len(x)
    beta_1 = 0.9
    beta_2 = 0.999  # initialize the values of the parameters
    epsilon = 1e-8

    m_t = np.zeros(n)
    v_t = np.zeros(n)

    k = 0  # Begin at k=1
    np.random.seed(s)  # Assign a chosen seed
    it_x, it_k, it_f = np.empty(0), np.empty(0), np.empty(0)  # Create empty arrays to record ending points
    while k < iter:  # While k remains under the number of iterations, continue to run
        it_x = np.append(it_x, x)  # Add first element of new point to list
        g_t = scaled_dir(x, gr_test_func)
        m_t = beta_1 * m_t + (1 - beta_1) * g_t
        v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)
        m_cap = m_t / (1 - (beta_1 ** (k+1)))
        v_cap = v_t / (1 - (beta_2 ** (k+1)))
        x = x - (m*m_cap)/(np.sqrt(v_cap)+epsilon)

        k += 1

    return (it_x.reshape(iter, n))  # Returns [iter,n] array, which is an array of points for trajectory values

#Array of Names of Test Functions for Dictionary of Iterations
test_functions=[chained_rosen,
                alt_rosen,
                convex_nev,
                convex_funn,
                ackley]

testfunc_names = ['Chained Rosenbrock Function',
                  'Alternative Chained Rosenbrock Function',
                  'Nesterov Convex Quadratic Function',
                  'Convex Funnel Function',
                'Ackley Function']

#Corresponding Gradient Vector Functions for Test Functions
gradient_functions=[gr_chained_rosen,
                    gr_alt_rosen, 
                    gr_convex_nev, 
                    gr_convex_funn,
                    gr_ackley]

#Methods to chose the Randomized Scaled Component
dir_functions=[standard_dk, 
               rcd_dk]
dir_names=['Standard','RCD']

#Iterative Algorithms
it_algs=[standard_it,adam]
it_names=['Standard','Adam']





