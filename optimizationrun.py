#to ignore warning messages
import warnings
warnings.filterwarnings("ignore")

import os
import unittest
import numpy as np
from sys import argv
import math
import random
import matplotlib.pyplot as plt
import time

from smt.applications import EGO
from smt.applications.ego import Evaluator
from smt.utils.sm_test_case import SMTestCase
from smt.problems import Branin, Rosenbrock, HierarchicalGoldstein
from smt.sampling_methods import FullFactorial
from multiprocessing import Pool
from smt.sampling_methods import LHS
import itertools
from typing_extensions import NewType

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

eps = 1e-9
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
def cost_function(x_1, x_2, x_3, x_4, x_5, x_6, max_budget= 100000):
  return budget_function(x_1,x_2,x_3,x_4,x_5,x_6)/max_budget
def cost_function_vec(X):
  return cost_function(X[0], X[1], X[2], X[3], X[4], X[5])
def cost_function_matrix(X):
  return np.apply_along_axis(cost_function_vec, 1, X)

def getGradientPoints(x_points, gd_param, alpha):
  random_vector = 0
  tuning_param = 0
  gradient_estimate = 0
  return gradient_x_points, gradient_y_points

def winnowLHSbyBudget(LHSsample, cost_function_estimator):
  return LHSsample

def fullMeshIteration(initial_point, current_mesh_width, direction_matrix, categorical_flag):
  return x_points, y_points

def simulatePoints(X):
  #parse X and feed to simulation function or program
  points = loss_function_matrix(X)
  points = points.reshape(-1,1)
  return points
def analyzeCurrentRound(allX, oldX, iteration_number):
  if(iteration_number == 0):
    x = 0
  else:
    x = 1
    #adjust gradient and mesh size
  #check simulation budget and stoppage criterion

  #surrogate_predictions = sm_loss.predict_values(allX)
  opt_diff = 0
  mesh_size = 0
  grad_param =0 
  return mesh_size, grad_param, opt_diff
def getNewSetOfPointsFromSurrogate(surrogate_model, allX , n_samples, exploit_coeff, alpha, lowerbounds, upperbounds, mesh_size, direction_matrix, grad_d_param):
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
  n_other = n_samples - n_var
  variance_sample = MixedIntegerSamplingMethod (LHS , design_space, criterion ="ese", random_state =np.random.randint(0,10000))
  Xvar = variance_sample(400*n_samples) #make this giant!!
  LCBS = surrogate_model.predict_values(Xvar) - alpha * surrogate_model.predict_variances(Xvar)
  LCBMat = np.hstack((Xvar, LCBS))
  LCBMat = LCBMat[LCBMat[:, -1].argsort()]
  LCBMat = LCBMat[:n_var,:]
  newX = LCBMat[:, :-1] 
  newY = simulatePoints(newX)
  print(newY.min())
  #Mesh Points
  #Gradient Points
  return newX, newY

def updatePQ(allX, newX):
  #add newX to allX and sort
  allX = np.vstack((allX,newX))
  return allX[allX[:, -1].argsort()]

def fullOptimization(budget, n_samples, lowerbounds, upperbounds, n_latins, e_coeff, alpha, direction_matrix, categorical_flag, lamb,mesh_size, grad_d_param, cost_function = False):
  n_doe = n_samples
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
  sm_cost = MixedIntegerKrigingModel(
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
  for i in range(n_latins):
    sampling = MixedIntegerSamplingMethod (LHS , design_space, criterion ="ese", random_state = np.random.randint(0,10000))
    Xt = sampling (n_doe)
    #generated from simulator
    Yt = simulatePoints(Xt)
    Ct = cost_function_matrix(Xt)
    OverallValue = Yt + lamb*Ct
    sm_loss.set_training_values(Xt, Yt)
    sm_loss.train()
    sm_cost.set_training_values(Xt,Ct)
    sm_cost.train()
    subX = np.hstack((Xt,Yt))
    allX = updatePQ(allX, subX)
  allX = np.delete(allX, (0), axis=0)
  opt_value = math.inf
  current_mesh_size = mesh_size
  iteration_number = 0 
  while(budget > 0):
    oldX = allX.copy()
    newX, newY = getNewSetOfPointsFromSurrogate(sm_loss, allX , n_samples, e_coeff, alpha, lowerbounds, upperbounds, current_mesh_size, direction_matrix, grad_d_param)
    sm_loss.set_training_values(newX, newY)
    sm_loss.train()
    mergedX = np.hstack((newX,newY))
    allX = updatePQ(allX, mergedX)
    print(allX.shape)
    opt_value = allX[0,-1]
    print(opt_value)
    budget = budget - 1
    iteration_number += 1
    current_mesh_size, current_grad_d_param, opt_diff = analyzeCurrentRound(allX, oldX, iteration_number)
  return allX
#n_test = 100
#X_test = sampling(n_test)
#y_test = loss_function_matrix(Xt)
#surrogate_predictions = sm_loss.predict_values(Xt)
#point_derivatives = sm_loss.predict_derivatives(Xt,0)
#print(np.abs(np.subtract(surrogate_predictions, y_test)).mean())
n_samples = 60
budget = 5
n_latins = 3
lamb = 0
mesh_size = 5
lowerbounds = [1,1,0,0,"unpaired",0]
upperbounds = [5000,100,0.99,10,"paired",1]
e_coeff = 0.6
alpha = 1
grad_d_param = 1
direction_matrix= np.eye(3)
categorical_flag = [1,0,0,1,1,0]
allDesigns = fullOptimization(budget, n_samples, lowerbounds, upperbounds, n_latins, e_coeff, alpha, direction_matrix, categorical_flag, lamb, mesh_size, grad_d_param, cost_function = False)
print(allDesigns[:20,:])
