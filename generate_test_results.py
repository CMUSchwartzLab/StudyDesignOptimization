from liquid_biopsyopt import *
'''
path_to_train_matrix = '/projects/schwartzlabscratch/DesignOpt/test_results/adjustpaired3/allX3.txt'
liquid_bio_source_dir = '/projects/schwartzlabscratch/LiquidBioData/test_data'
ground_truth_dir = '/projects/schwartzlabscratch/LiquidBiopsyData/test_data/results'
output_dir = '/projects/schwartzlabscratch/DesignOpt/final_liquid_biopsy_results'

def loadTestPoints(X_final): 
  X_withscore = np.loadtxt(X_final)
  X = X_withscore[:-1]
  return X

X = loadTestPoints(path_to_train_matrix)
lamb = 0
points = doLiquidBiopsySimulationPipeline(X, lamb,0, output_dir, liquid_bio_source_dir, ground_truth_dir)
points = np.array(points)
#points = loss_function_matrix(X)
points = points.reshape(-1,1)
costs = cost_function(X)
costs = costs.reshape(-1,1)
total_loss = points + lamb*costs
total_loss = np.array(total_loss)
scores = np.reshape(total_loss, (len(total_loss), 1))
final_X = np.hstack((X, scores))
np.savetxt(final_X, output_dir +'/final_array.txt')
'''

sample_dir = '/projects/schwartzlabscratch/DesignOpt/test_results/adjustpaired3/0'
ground_truth_dir = '/projects/schwartzlabscratch/LiquidBiopsyData/training_data/results'

print(getscoresfromcalls(sample_dir, ground_truth_dir))
