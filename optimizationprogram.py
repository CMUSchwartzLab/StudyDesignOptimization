import warnings
warnings.filterwarnings("ignore")

import os
import unittest
import numpy as np
from sys import argv
import math
import random
import time
from optimization_sim import *

#from smt.applications import EGO
#from smt.applications.ego import Evaluator
#from smt.utils.sm_test_case import SMTestCase
#from smt.problems import Branin, Rosenbrock, HierarchicalGoldstein
#from smt.sampling_methods import FullFactorial
from multiprocessing import Pool
from smt.sampling_methods import LHS
from typing_extensions import NewType
import itertools
from smt.surrogate_models import (
    KRG,
    GEKPLS,
    KPLS,
    QP,
    MixIntKernelType,
    MixHrcKernelType,
)
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    MixedIntegerSamplingMethod,
    MixedIntegerKrigingModel
)

from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
    CategoricalVariable,
)
#CONSTANTS
eps = 1e-9
n_samples = 10
budget = 4
n_latins = 2
lamb = 0
cost_max = 1000000
NUM_CORES = 8
mesh_size = 5
lowerbounds = [250,1,0,0,0,0]
upperbounds = [5000,100,0.99,10,1,1]
e_coeff = 0.7
alpha = 5
grad_d_param = 1
perturbation_vector = [100,1,0.01,1,1,0.01]
direction_matrix= np.array([200, 5,0.01,1,1,0.02])
categorical_flag = [1,0,0,1,1,0]
num_cores = 5
default_param_list = parameter_list.copy()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def loss_function(x_1, x_2, x_3, x_4, x_5, x_6):
  if(x_5 > 0.5):
    paired_mult = 1
  else:
    paired_mult = 0.5
  genome_mult = x_6
  coinflip = random.random()
  if(coinflip < 0.5):
    sigmoid_factor = sigmoid(0.0005*x_1 +0.03*x_2-6*x_3+1*x_4)
  else:
    sigmoid_factor = sigmoid(0.0005*x_1 +0.02*x_2-6*x_3+1*x_4)
  total_score = paired_mult*genome_mult*sigmoid_factor
  return (1-total_score)
def loss_function_vec(X):
  return loss_function(X[0], X[1], X[2], X[3], X[4], X[5])
def loss_function_matrix(X):
  return np.apply_along_axis(loss_function_vec, 1, X)
def budget_function(x_1, x_2, x_3, x_4, x_5, x_6):
  read_len_budget = 30*x_1
  coverage_budget = 3*x_2
  error_budget = 500*(1- (1/(1+(math.sqrt((1/(x_3+eps))-1)))))
  cell_budget = 300
  cell_budget_scaler = x_4+1
  combined_terms = 2*x_4*x_2 + x_1*x_2
  scaling_factor = 0.03
  if(x_5 == 1):
    paired_modifier = 1.3
  else:
    paired_modifier = 1
  wgs_modifier = x_6
  total_cost = scaling_factor*wgs_modifier * paired_modifier * cell_budget_scaler*(read_len_budget + coverage_budget + error_budget + combined_terms)
  return total_cost

def train_cost_function_from_data(budget, n_samples, lowerbounds, upperbounds, n_latins, e_coeff, alpha, direction_matrix, categorical_flag, lamb,mesh_size, grad_d_param, perturbation_vector, cost_function = False):
  design_space = DesignSpace ([
    IntegerVariable (lowerbounds[0], upperbounds[0]), #Read Length
    FloatVariable (lowerbounds[1], upperbounds[1]), #Coverage
    FloatVariable (lowerbounds[2], upperbounds[2]), #error rate
    IntegerVariable (lowerbounds[3], upperbounds[3]), #number of single cells
    CategoricalVariable ([lowerbounds[4], upperbounds[4]]), #paired or unpaired
    FloatVariable (lowerbounds[5], upperbounds[5]), #fraction of genome produced
  ])
  sm_cost = MixedIntegerKrigingModel(
      surrogate=KRG(
          design_space=design_space,
          theta0=[1e-2],
          corr="matern32",
          n_start=20,
          categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
      ),
  )
  n_dor = int(n_samples*0.5)
  for i in range(n_latins):
    sampling = MixedIntegerSamplingMethod (LHS , design_space, criterion ="ese", random_state = np.random.randint(0,10000))
    Xt = sampling (n_doe)
    #GET COST FUNCITON VALUES QUICK
    Ct = cost_function_matrix(Xt)
  sm_cost.set_training_values(Xt,Ct)
  sm_cost.train()
  return sm_cost

def cost_function_krig_valued(Xt, surrogate): 
  return surrogate.predict_values(Xt)

def cost_function(x_1, x_2, x_3, x_4, x_5, x_6, max_budget= cost_max):
  return budget_function(x_1,x_2,x_3,x_4,x_5,x_6)/max_budget
def cost_function_vec(X):
  return cost_function(X[0], X[1], X[2], X[3], X[4], X[5])
def cost_function_matrix(X):
  return np.apply_along_axis(cost_function_vec, 1, X)

#ADJUST BUDGET and fix cost function into the algorithm
def probabilistic_round(x):
    return int(math.floor(x + random.random()))

def getGradientPoints(x_points, surrogate_model, gd_param, alpha, lowerbounds, upperbounds, perturbation_vector, categorical_flag):
  #check feasibility of new points
  #numerically estimate gradients
  all_gradients = []
  print(perturbation_vector)
  print(gd_param)
  perturb = gd_param*np.array(perturbation_vector)
  for i in range(len(perturb)): 
    if(categorical_flag[i] == 1):
      perturb[i] = probabilistic_round(perturb[i])
  for point in x_points: 
    gradient = []
    for j in range(len(point)):
      new_point1 = point.copy()
      new_point2 = point.copy()
      new_point1[j] = new_point1[j] + perturb[j]
      new_point2[j] = new_point2[j] - perturb[j]
      new_point1 = np.reshape(new_point1, (1,-1))
      new_point2 = np.reshape(new_point2, (1,-1))
      finite_diff = surrogate_model.predict_values(new_point1) - surrogate_model.predict_values(new_point2)
      grad_est_j = finite_diff[0]/(2*perturb[j])
      gradient.append(grad_est_j[0])
    all_gradients.append(gradient)
  all_gradients = np.array(all_gradients)
  #print(all_gradients)
  #stack alpha-shifted point
  gradient_points = []
  for index_point in range(x_points.shape[0]): 
    g_point = x_points[index_point] - alpha * all_gradients[index_point]
    for i in range(g_point.shape[0]):
      if(g_point[i] < lowerbounds[i]):
        g_point[i] = lowerbounds[i]
      if(g_point[i] > upperbounds[i]):
        g_point[i] = upperbounds[i]
    for i in range(g_point.shape[0]): 
      if(categorical_flag == 1): 
        g_point[i] = probabilistic_round(g_point[i])
    gradient_points.append(g_point)
  #check feasibility 
  gradient_points = np.array(gradient_points)
  return gradient_points

def checkCost(allX, cost_function_estimator):
  costs = cost_function_estimator(allX)
  cost_mask = (costs < 1)
  feasibleX = allX[cost_mask,:]
  return feasibleX

def fullMeshIteration(initial_points, number_mesh_points, current_mesh_width, direction_matrix, categorical_flag, lowerbounds, upperbounds):
  #check feasibility of new points 
  x_points = []
  num_rows = initial_points.shape[0]
  get_row = random.randint(0,num_rows-1)
  while(number_mesh_points > 0):
    initial_point = initial_points[get_row, :]
    sample_scheme = random.randint(0,2)
    continuous_perturbation_vector = []
    discrete_perturbation_vector = []
    for i in range(initial_point.shape[0]):
      if(categorical_flag[i] == 0):
        continuous_perturbation_vector.append(current_mesh_width*random.uniform(-2,2)*direction_matrix[i])
        discrete_perturbation_vector.append(0)
      else:
        continuous_perturbation_vector.append(0)
        discrete_perturbation_vector.append(current_mesh_width*random.randint(-2,2)*direction_matrix[i])
    cont_array = np.array(continuous_perturbation_vector)
    disc_array = np.array(discrete_perturbation_vector)
    #cont explorer
    initial_point = [float(numeric_string) for numeric_string in initial_point]
    initial_point = np.array(initial_point)
    if(sample_scheme == 0):
      new_point = initial_point + cont_array
    elif(sample_scheme == 1):
      new_point = initial_point + disc_array
    else:
      new_point = initial_point+disc_array+cont_array
    #feasibility check
    for i in range(new_point.shape[0]):
      if(new_point[i] < lowerbounds[i]):
        new_point[i] = lowerbounds[i]
      if(new_point[i] > upperbounds[i]):
        new_point[i] = upperbounds[i]
    for i in range(new_point.shape[0]):
      if(categorical_flag[i] == 1): 
        new_point[i] = probabilistic_round(new_point[i])
    x_points.append(new_point)
    number_mesh_points -= 1
  x_points = np.array(x_points)
  return x_points

def simulatePoints(X, lamb, cost_function, iteration_number, opt_store_directory):
  #parse X and feed to simulation function or program, and parallelize, reparse results
  #PARSE X
  simulation_list = []
  for i in range(len(X)):
    dataid = 'optimization_round'+str(iteration_number)+'_datapoint'+str(i)
    subparam_list = default_param_list.copy()
    subparam_list[0] = opt_store_directory
    subparam_list[1] = dataid
    read_len = X[i][0]
    frag_len = read_len*2
    coverage = X[i][1]
    error_rate = X[i][2]
    single_cells = X[i][3]
    paired = X[i][4]
    WES = 1-round(X[i][5])
    subparam_list[4] = read_len
    subparam_list[5] = error_rate
    subparam_list[6] =  frag_len
    subparam_list[8] = paired
    subparam_list[9] = WES
    subparam_list[10] = single_cells
    subparam_list[11] = coverage
    simulation_list.append(subparam_list)
  pool = Pool(NUM_CORES)
  pool.starmap(generateResults, simulation_list)
  #generate and parse the results of the simualtions
  
  points = loss_function_matrix(X)
  points = points.reshape(-1,1)
  costs = cost_function(X)
  costs = costs.reshape(-1,1)
  total_loss = points + lamb*costs
  return total_loss

def analyzeCurrentRound(allX, iterX, iteration_number, mesh_size, grad_d_param, e_coeff):
  #adjust gradient and mesh size
  #surrogate_predictions = sm_loss.predict_values(allX)
  #get mins of allX and iterX 
  allX = allX[allX[:, -1].argsort()]
  iterX = iterX[iterX[:,-1].argsort()]
  prevroundsmin = allX[0,-1]
  currentroundsmin = iterX[0,-1]
  print(prevroundsmin, currentroundsmin)
  fracdiff = (prevroundsmin-currentroundsmin)/prevroundsmin
  if(fracdiff < 0):
    e_coeff = e_coeff*0.9
    mesh_size = mesh_size* 0.97
    grad_param = grad_d_param*0.97
  else: 
    e_coeff = e_coeff*0.9
    mesh_size = mesh_size* 1.2
    grad_param = grad_d_param*1.2
  return mesh_size, grad_param, e_coeff

def getNewSetOfPointsFromSurrogate(surrogate_model, allX , n_samples, exploit_coeff, alpha, lowerbounds, upperbounds, mesh_size, direction_matrix, gd_param, categorical_flag, perturbation_vector, lamb, cost_function, iteration_number, opt_store_directory):
  #allX must be sorted by val for this to work
  #LCB Points
  design_space = DesignSpace ([
    IntegerVariable (lowerbounds[0], upperbounds[0]), #Read Length
    FloatVariable (lowerbounds[1], upperbounds[1]), #Coverage
    FloatVariable (lowerbounds[2], upperbounds[2]), #error rate
    IntegerVariable (lowerbounds[3], upperbounds[3]), #number of single cells
    CategoricalVariable ([lowerbounds[4], upperbounds[4]]), #paired or unpaired
    FloatVariable (lowerbounds[5], upperbounds[5]), #fraction of genome produced
  ])
  n_var = int(n_samples*exploit_coeff)
  n_var = max(1,n_var)
  n_other = n_samples - n_var
  n_other = max(1,n_other)
  variance_sample = MixedIntegerSamplingMethod (LHS , design_space, criterion ="ese", random_state =np.random.randint(0,10000))
  Xvar = variance_sample(300*n_samples) #make this giant
  #construct bounds around low points and sample there
  cutoff = int(0.2*allX.shape[0])
  miniarray = allX[:cutoff, :-1]
  miniarray = miniarray.astype(float)
  lowcutoff = []
  highcutoff = []
  for i in range(miniarray.shape[1]):
    minimal = np.min(miniarray[:, i])
    maximal = np.max(miniarray[:, i])
    if(minimal == maximal):
      lowcutoff.append(lowerbounds[i])
      highcutoff.append(upperbounds[i])
    else:
      lowcutoff.append(minimal)
      highcutoff.append(maximal)
  design_space2 = DesignSpace ([
    IntegerVariable (lowcutoff[0], highcutoff[0]), #Read Length
    FloatVariable (lowcutoff[1], highcutoff[1]), #Coverage
    FloatVariable (lowcutoff[2], highcutoff[2]), #error rate
    IntegerVariable (lowcutoff[3], highcutoff[3]), #number of single cells
    CategoricalVariable ([lowcutoff[4], highcutoff[4]]), #paired or unpaired
    FloatVariable (lowcutoff[5], highcutoff[5]), #fraction of genome produced
  ])
  variance_sample2 = MixedIntegerSamplingMethod (LHS , design_space2, criterion ="ese", random_state =np.random.randint(0,10000))
  Xvar2 = variance_sample2(300*n_samples)
  mergedX = np.vstack((Xvar, Xvar2))
  mergedX = checkCost(mergedX, cost_function)
  LCBS = surrogate_model.predict_values(mergedX) - alpha * surrogate_model.predict_variances(mergedX)
  LCBMat = np.hstack((mergedX, LCBS))
  LCBMat = LCBMat[LCBMat[:, -1].argsort()]
  LCBMat = LCBMat[:n_var,:]
  finalLCBMat = LCBMat[:, :-1]
  #Mesh Points
  mesh_threshold = 3
  initial_points = allX[0:mesh_threshold,:-1]
  number_mesh_points = n_other
  current_mesh_width = mesh_size
  MeshX = fullMeshIteration(initial_points, number_mesh_points, current_mesh_width, direction_matrix, categorical_flag, lowerbounds, upperbounds)
  MeshX = checkCost(MeshX, cost_function)
  checker = MeshX.shape[0]
  if(checker == 0): 
    while(checker == 0):
      current_mesh_width = current_mesh_width/2
      MeshX = fullMeshIteration(initial_points, number_mesh_points, current_mesh_width, direction_matrix, categorical_flag, lowerbounds, upperbounds)
      MeshX = checkCost(MeshX, cost_function)
      checker = MeshX.shape[0]
  #Gradient Points
  maximum_gradient_points = np.shape(allX)[0]
  print(maximum_gradient_points)
  taper_max = min(maximum_gradient_points, n_other)
  gradient_xpoints = allX[0:taper_max,:-1]
  GradientX = getGradientPoints(gradient_xpoints, surrogate_model, gd_param, alpha, lowerbounds, upperbounds, perturbation_vector, categorical_flag)
  GradientX = checkCost(GradientX, cost_function)
  checker = GradientX.shape[0]
  if(checker == 0):
    while(checker == 0):
      alpha = alpha/2
      GradientX = getGradientPoints(gradient_xpoints, surrogate_model, gd_param, alpha, lowerbounds, upperbounds, perturbation_vector, categorical_flag)
      GradientX = checkCost(GradientX, cost_function)
      checker = GradientX.shape[0]
  #merge matrices
  print(finalLCBMat.shape)
  print(MeshX.shape)
  print(GradientX.shape)
  newX = np.vstack((finalLCBMat, MeshX, GradientX))
  newY = simulatePoints(newX, lamb, cost_function, iteration_number, opt_store_directory)
  print(newY.min())
  return newX, newY

def updatePQ(allX, newX):
  #add newX to allX and sort
  allX = np.vstack((allX,newX))
  return allX[allX[:, -1].argsort()]

def fullOptimization(budget, n_samples, lowerbounds, upperbounds, n_latins, e_coeff, alpha, direction_matrix, categorical_flag, lamb,mesh_size, grad_d_param, perturbation_vector, cost_function, maximum_cost):
  n_doe = int(n_samples*0.5)
  n_doe = max(1, n_doe) #safety check
  design_space = DesignSpace ([
    IntegerVariable (lowerbounds[0], upperbounds[0]), #Read Length
    FloatVariable (lowerbounds[1], upperbounds[1]), #Coverage
    FloatVariable (lowerbounds[2], upperbounds[2]), #error rate
    IntegerVariable (lowerbounds[3], upperbounds[3]), #number of single cells
    CategoricalVariable ([lowerbounds[4], upperbounds[4]]), #paired or unpaired
    FloatVariable (lowerbounds[5], upperbounds[5]), #fraction of genome produced
  ])
  sm_loss = MixedIntegerKrigingModel(
    surrogate=KRG(
          design_space=design_space,
          theta0=[1e-2],
          corr="matern32",
          n_start=20,
          categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
      ),
  )
  fix_list = lowerbounds.copy()
  fix_list.append(-1000000000)
  allX = np.array(fix_list)
  target_size = int(1.5*n_samples)
  current_size = 0
  iteration_number = 0
  nullX = np.array(lowerbounds.copy())
  #CHECK THIS CODE, some numerics needed
  while(current_size < target_size):
    for i in range(n_latins):
      sampling = MixedIntegerSamplingMethod (LHS , design_space, criterion ="ese", random_state = np.random.randint(0,10000))
      Xi = sampling (n_doe)
      Xi = checkCost(Xi, cost_function)
      nullX = np.vstack((nullX, Xi))
      current_size += Xi.shape[0]
    Xt = nullX[1:,:]
    print('xt shape',Xt.shape)
    #generated from simulator
  Yt = simulatePoints(Xt, lamb, cost_function, iteration_number, opt_store_directory)
  sm_loss.set_training_values(Xt, Yt)
  sm_loss.train()
  subX = np.hstack((Xt,Yt))
  allX = updatePQ(allX, subX)

  allX = np.delete(allX, (0), axis=0)
  opt_value = allX[0,-1]
  print(opt_value)
  iterX = allX.copy()
  #SAVE allX to disk here
  np.savetxt(opt_store_directory+'allXinit.txt', allX, fmt = '%s')
  opt_value = math.inf
  while(budget > 0):
    #can swap here to iterX
    iteration_number += 1
    originalX = allX.copy()
    newX, newY = getNewSetOfPointsFromSurrogate(sm_loss, allX, n_samples, e_coeff, alpha, lowerbounds, upperbounds, mesh_size, direction_matrix, grad_d_param, categorical_flag, perturbation_vector, lamb, cost_function, iteration_number, opt_store_directory)
    sm_loss.set_training_values(newX, newY)
    sm_loss.train()
    iterX = np.hstack((newX,newY))
    print(iterX.shape)
    allX = updatePQ(allX, iterX)
    print(allX.shape)
    opt_value = allX[0,-1]
    print(opt_value)
    budget = budget - 1
    mesh_size, grad_d_param, e_coeff = analyzeCurrentRound(originalX, iterX, iteration_number, mesh_size, grad_d_param, e_coeff)
    #SAVE allX to DISK HERE
    np.savetxt(opt_store_directory+'allX{}.txt'.format(iteration_number), allX, fmt = '%s')
  return allX, sm_loss
#n_test = 100
#X_test = sampling(n_test)
#y_test = loss_function_matrix(Xt)
#surrogate_predictions = sm_loss.predict_values(Xt)
#point_derivatives = sm_loss.predict_derivatives(Xt,0)
#print(np.abs(np.subtract(surrogate_predictions, y_test)).mean())
if __name__ == '__main__':
  ts = time.time()
  base_directory = '/Users/arjunsrivatsa/Desktop/DesignOpt/test_results'
  try:
    opt_store_directory = base_directory+'/'+sys.argv[1]+'/'
  except: 
    opt_store_directory = base_directory+'/'+ str(random.randint(0,100000))+'/'
  #create directory to store run 
  default_param_list[0] = opt_store_directory
  makedir(opt_store_directory)
  allDesigns, sm_loss = fullOptimization(budget, n_samples, lowerbounds, upperbounds, n_latins, e_coeff, alpha, direction_matrix, categorical_flag, lamb, mesh_size, grad_d_param, perturbation_vector, cost_function_matrix, cost_max)
  te = time.time()
  print('time elapsed', te-ts)