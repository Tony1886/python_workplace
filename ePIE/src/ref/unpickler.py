# Written by Matthew Kwiecien in 2014
import pickle
from deap import tools, creator, base
import numpy as np
from ePIE_GA import dims

## Creates the classes used by the DEAP genetic evolutionary algorithm package
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

## The dims function will generate a modified cartesian grid to use as input to the ePIE algorithm
def dims(xn, yn, stepsize, individual, list_filename, list_file_loc):
    
    # Initializing local variables
    col_s1=[]
    f_nums=[]
    s1_float=[]
    s1 = ""
    ind_string=""
    fn=list_filename+".csv"
    
    # Gathering parameters for rotation, scaling, and shearing the cartesian grid
    I_THETA = individual[0]
    XSHEAR = individual[1]
    YSHEAR = individual[2]
    SCALE = individual[3]
    
    y = np.linspace(0, yn, yn)
    x = np.linspace(0, xn, xn)
    (xg,yg)=np.meshgrid(x,y)
    xg = (xg - (xn/2))*stepsize
    yg = (yg - (yn/2))*stepsize

    for i in range(xn):
        for j in range(yn):
            s1+="{:e} {:e},".format(xg[j,i],yg[j,i])
            
    s1_split=s1.split(",")
    for i in range(7482):
            s1_float.append(map(float,s1_split[i].split(" ")))
        
    for i in range(7482):
        for j in range(2):
            col_s1.append(s1_float[i][j])
    
    # Rotation of grid
    rot = np.reshape(col_s1, (-1,2))
    xrot=[]
    yrot=[]
    for i in range(7482):
        xrot.append(np.cos(I_THETA)*rot[i][0] + np.sin(I_THETA)*rot[i][1])
        yrot.append(-np.sin(I_THETA)*rot[i][0] + np.cos(I_THETA)*rot[i][1])
        
    # Scaling grid
    xscale = [i*SCALE for i in xrot]
    yscale = [i*SCALE for i in yrot]

    # Shear grid
    xshear_x=[]
    yshear_x=[]
    xshear_y = []
    yshear_y = []
    for i in range(7482):
        xshear_x.append(xscale[i]+XSHEAR*yscale[i]) 
        yshear_x.append(yscale[i])
            
  for i in range(7482):
        xshear_y.append(xshear_x[i])
        yshear_y.append(YSHEAR*xshear_x[i] + yshear_x[i]) 

    f_nums=zip(xshear_y, yshear_y)
    for j in range(7482):
        ind_string+="{:e} {:e},".format(f_nums[j][0],f_nums[j][1])    

    # Save grid to file
    f=open(list_file_loc+fn,'w')
    f.write(ind_string.strip(','))
    f.close()

## The paramSave function will write to a file a list of input parameters
def paramSave(best_qxy_x, best_qxy_y, best_z, best_energy, output_filename):
    fn = output_filename+"_params.csv"
    best_qxy=best_qxy_x,best_qxy_y
    f = open('/local/kwiecien/ePIE_codes/bparams/'+fn, 'w')
    f.write('QXY='+str(best_qxy)+'\n'+'z='+str(best_z)+'\n'+'energy='+str(best_energy))

## The up function will unpickle the file that holds the hall of fame from the ePIE_GA output
## and create the cartesian grid and parameter file and save them.
def up(file_range):
    
    # Iterates through each data set
    for i in range(file_range):
        fileset = 'mda{:d}'.format(309+(2*i))
        hof=fileset+'_optim3_hof.p'
        output_filename = fileset+'_8_13'
        input_filename='/local/kwiecien/ptycho/src/data/'+hof
    	params = pickle.load(open(input_filename, 'r'))
    	individual = params[0]
    	   
    	# Creates and saves the best grid and set of parameters for a given data set
    	dims(58,129,70e-9,individual,output_filename)
    	paramSave(individual[4], individual[5], individual[6], individual[7], output_filename)
        print hof,fileset,input_filename,output_filename





