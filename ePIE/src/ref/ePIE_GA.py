## Written by Matthew Kwiecien in 2014
import random
from deap import algorithms, base, creator, tools
import ipdb
import os
import numpy as np
import random
import subprocess
import pickle
import time

# Number of data sets to iterate through
N_SETS = 31
param_dict={}
for i in range(N_SETS):
    # Pull the ith dataset information and store in a dictionary
    setnum=(2*i)
    fileset ='mda{:d}'
    fileset = fileset.format(309+setnum)
    filestart = '{:d}'
    filestart = filestart.format(i*7482)
    jobID = fileset+'_optim3'
    list_filename = fileset + '_list'
    param_dict[i]=fileset, filestart, jobID, list_filename
    print "STARTING FILESET -- "+fileset+" -- "
    
    # Parameters
    xn=58
    yn=129
    stepsize=70e-9
    N=128
    N_POP = 13
    
    # Range of values to try for fitness
    Z_VAL = .45
    Z_MIN, Z_MAX = Z_VAL - (.01*Z_VAL), Z_VAL + (.01*Z_VAL)
    E_VAL = 2.535
    E_MIN, E_MAX = E_VAL - (.01*E_VAL), E_VAL + (.01*E_VAL)
    QXY_X,QXY_Y, = 74, 185
    QXY_X_MIN,QXY_X_MAX = QXY_X - (0.05*QXY_X), QXY_X + (0.05*QXY_X)
    QXY_Y_MIN,QXY_Y_MAX = QXY_Y - (0.05*QXY_Y), QXY_Y + (0.05*QXY_Y)
    SCALE_MIN, SCALE_MAX = .9,1.1
    XSHEAR_MIN, XSHEAR_MAX = 0, .05
    YSHEAR_MIN, YSHEAR_MAX = 0, .05 
    THETA=(np.pi)/72
    
    # Location of input files
    h5_file_loc='/local/kwiecien/mar12_andreasen/hdf5_andreasen/'+fileset+'/fscan_13_#06d.h5'
    list_file_loc='/local/kwiecien/ePIE_codes/'
    error_loc='/local/kwiecien/ptycho/src/data/'
    
    # Number of attempts per data set
    iterations='15'

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

    ## The epie function will run the ePIE algorithm on a given set of parameters determined by the individual being used at the
    ## given step in the genetic algorithm and then output the error for that set of parameters.  
    def epie(list_file_loc, error_loc, jobID, iterations, filestart, h5_file_loc, qxy_x, qxy_y, z, energy):
        
        # A pipe is opened in the shell and the ePIE is run
        p=subprocess.Popen(["mpirun", "-n", "1", "./ePIE", \
            "-jobID="+jobID , "-blind=1", "-fp="+h5_file_loc , \
            "-fs="+filestart , "-lf="+list_file_loc+list_filename+".csv" , \
            "-beamsize=0.15e-6", "-qxy="+qxy_x+","+qxy_y , "-scanDims=1,7482", \
            "-step=70e-9,70e-9", "-probeModes=4", "-i="+iterations , \
            "-sqrtData", "-fftShiftData", "-rotate90=1", "-threshold=0", \
            "-size=128", "-detectorSize=4096", "-energy="+energy , "-dx=150e-6", \
            "-z="+z , "-overlap=0"], stdout=subprocess.PIPE)
        
        # Data is retrieved from the pipe
        out, err = p.communicate()

        # The program will hold until the subprocess finishes, then print out the error, and output the error
        p.wait()
        if p.returncode==0:
            pass
            error_metric = out
            print "Evaluated with ERROR of", error_metric
            return error_metric

    ## The minimumError function will call the dims function to generate a cartesian grid, then call the epie function 
    ## to find the error for the given cartesian grid, and return that error
    def minimumError(individual, xn=xn, yn=yn, stepsize=stepsize, list_filename=list_filename, list_file_loc=list_file_loc, error_loc=error_loc, jobID=jobID, iterations=iterations, filestart=filestart, h5_file_loc=h5_file_loc):
        qxy_x = str(individual[4])
        qxy_y = str(individual[5])
        z = str(individual[6])
        energy = str(individual[7]) 
        dims(xn, yn, stepsize, individual, list_filename, list_file_loc)

        return epie(list_file_loc, error_loc, jobID, iterations, filestart, h5_file_loc, qxy_x, qxy_y, z, energy),
    
    ## The pickler function will take the hall of fame, or set of parameters that result in the minimum error across the
    ## whole simulation, and save them into a .p file
    def pickler(jobID, hof):
        pickle.dump( hof, open("/local/kwiecien/ptycho/src/data/"+jobID+"_hof.p", 'w'))
    

    # DEAP package class creation
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox=base.Toolbox()

    # Defining the range of values to vary as attributes of different individuals in a given generation
    toolbox.register("rotation", random.uniform, -THETA, 0)
    toolbox.register("x_shear", random.uniform, XSHEAR_MIN, XSHEAR_MAX)
    toolbox.register("y_shear", random.uniform, YSHEAR_MIN, YSHEAR_MAX)
    toolbox.register("scale", random.uniform, SCALE_MIN, SCALE_MAX)
    toolbox.register("qxy_x_param", random.randint, round(QXY_X_MIN), round(QXY_X_MAX))
    toolbox.register("qxy_y_param", random.randint, round(QXY_Y_MIN), round(QXY_Y_MAX))
    toolbox.register("z_param", random.uniform, Z_MIN, Z_MAX)
    toolbox.register("energy_param", random.uniform, E_MIN, E_MAX)
    
    # Creating individuals with the above attributes and defining them as a generation
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.rotation, toolbox.x_shear, toolbox.y_shear, toolbox.scale, toolbox.qxy_x_param, toolbox.qxy_y_param, toolbox.z_param, toolbox.energy_param), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=N_POP)
    
    # Establishing evolutionary parameters and tools for evaluating generations
    toolbox.register("evaluate", minimumError)
    toolbox.register("mate", tools.cxUniform, indpb = .5)
    toolbox.register("mutate", tools.mutGaussian, mu=[0,0,0,0,0,0,0,0], sigma=[THETA, XSHEAR_MAX, YSHEAR_MAX, SCALE_MAX, 1, 1, Z_MAX, E_MAX], indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("map", map)
    
    ### main function that will run the genetic algorithm for a specified number of individuals and generations, 
    ### and will save the best individuals (set of parameters for ePIE algorithm) into a .p file
    def main():
        
        # Parameters for mutation in genetic algorithm
        CXPB, MUTPB, NGEN = .5, .05, 10
        
        # Initializing local variables
        pop = toolbox.population()
        hof = tools.HallOfFame(1) 
        pop, log = algorithms.eaSimple(pop,toolbox,CXPB,MUTPB,NGEN, halloffame = hof, verbose = True)
        return pickler(jobID, hof)
    
    main()

