from biogeme import *
from headers import *
from nested import *
from loglikelihood import *
from statistics import *

b_bicycle = -2.70
b_car = -0.40
b_pt1 = -6.03
b_pt2 = -3.24
b_pt3 = 1.67
cost_bicycle = -3.06
cost_car = -3.34
cost_pt1 = -1.21
cost_pt2 = -1.72
cost_pt3 = -3.44
cost_walk = -2.61

estimate_params = 1

l_active = Beta('l_active', 0.83, 0, 1, 1)
l_car = Beta('l_car', 1, 0, 10, 1)
l_pt = Beta('l_pt', 2.06, 1, 10, 1)

b_bicycle = Beta('b_bicycle', b_bicycle, -100, 100, estimate_params)
b_car = Beta('b_car', b_car, -100, 100, estimate_params)
b_pt1 = Beta('b_pt1', b_pt1, -100, 100, estimate_params)
b_pt2 = Beta('b_pt2', b_pt2, -100, 100, estimate_params)
b_pt3 = Beta('b_pt3', b_pt3, -100, 100, estimate_params)

cost_bicycle = Beta('cost_bicycle', cost_bicycle, -100, 100, estimate_params)
cost_car = Beta('cost_car', cost_car, -100, 100, estimate_params)
cost_pt1 = Beta('cost_pt1', cost_pt1, -100, 100, estimate_params)
cost_pt2 = Beta('cost_pt2', cost_pt2, -100, 100, estimate_params)
cost_pt3 = Beta('cost_pt3', cost_pt3, -100, 100, estimate_params)
cost_walk = Beta('cost_walk', cost_walk, -100, 100, estimate_params)

# TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED', TRAIN_TT / 100.0)
Bicycle_Cost_Outward = DefineVariable('Bicycle_Cost_Outward', Bicycle_Cost_Outward / 1000)
Bicycle_Cost_Return = DefineVariable('Bicycle_Cost_Return', Bicycle_Cost_Return / 1000)
Car_Cost_Outward = DefineVariable('Car_Cost_Outward', Car_Cost_Outward / 1000)
Car_Cost_Return = DefineVariable('Car_Cost_Return', Car_Cost_Return / 1000)
WAWE_Cost_Outward = DefineVariable('WAWE_Cost_Outward', WAWE_Cost_Outward / 1000)
WAWE_Cost_Return = DefineVariable('WAWE_Cost_Return', WAWE_Cost_Return / 1000)
PAWE_Cost_Outward = DefineVariable('PAWE_Cost_Outward', PAWE_Cost_Outward / 1000)
WAPE_Cost_Return = DefineVariable('WAPE_Cost_Return', WAPE_Cost_Return / 1000)
KAWE_Cost_Outward = DefineVariable('KAWE_Cost_Outward', KAWE_Cost_Outward / 1000)
WAKE_Cost_Return = DefineVariable('WAKE_Cost_Return', WAKE_Cost_Return / 1000)
Walk_Cost_Outward = DefineVariable('Walk_Cost_Outward', Walk_Cost_Outward / 1000)
Walk_Cost_Return = DefineVariable('Walk_Cost_Return', Walk_Cost_Return / 1000)
choice = DefineVariable('choice', choice)

# V1 = ASC_TRAIN + B_TIME * TRAIN_TT_SCALED + B_COST * TRAIN_COST_SCALED
v_bicycle = b_bicycle + cost_bicycle*Bicycle_Cost_Outward + cost_bicycle*Bicycle_Cost_Return
v_car = b_car + cost_car*Car_Cost_Outward + cost_car*Car_Cost_Return
v_pt1 = b_pt1 + cost_pt1*WAWE_Cost_Outward + cost_pt1*WAWE_Cost_Return
v_pt2 = b_pt2 + cost_pt2*PAWE_Cost_Outward + cost_pt2*WAPE_Cost_Return
v_pt3 = b_pt3 + cost_pt3*KAWE_Cost_Outward + cost_pt3*WAKE_Cost_Return
v_walk = cost_walk*Walk_Cost_Outward + cost_walk*Walk_Cost_Return

V = {1: v_bicycle,
     2: v_car,
     3: v_pt1,
     4: v_pt2,
     5: v_pt3,
     6: v_walk}

active = l_active, [1, 6]
car = l_car, [2]
pt = l_pt, [3, 4, 5]
nests = active, car, pt

av = {1: 1,
      2: 1,
      3: 1,
      4: 1,
      5: 1,
      6: 1}

prob = nested(V, av, nests, choice)

rowIterator('obsIter')
BIOGEME_OBJECT.ESTIMATE = Sum(log(prob), 'obsIter')
# BIOGEME_OBJECT.WEIGHTS = weights

# dl.data = dl.data[dl.get('Bicycle_av') != 0]
# dl.data = dl.data[dl.get('Car_av') != 0]
# dl.data = dl.data[dl.get('PT_Walk_Access_av') != 0]
# dl.data = dl.data[dl.get('PT_Park_Access_av') != 0]
# dl.data = dl.data[dl.get('PT_Kiss_Access_av') != 0]
# dl.data = dl.data[dl.get('Walk_av') != 0]
exclude = ((Bicycle_av == 0) +
           (Car_av == 0) +
           (PT_Walk_Access_av == 0) +
           (PT_Park_Access_av == 0) +
           (PT_Kiss_Access_av == 0) +
           (Walk_av == 0)
           ) > 0

BIOGEME_OBJECT.EXCLUDE = exclude

# Statistics
nullLoglikelihood(av, 'obsIter')
choiceSet = [1, 2, 3, 4, 5, 6]
cteLoglikelihood(choiceSet, choice, 'obsIter')
availabilityStatistics(av, 'obsIter')

BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "CFSQP"
BIOGEME_OBJECT.PARAMETERS['checkDerivatives'] = "1"
BIOGEME_OBJECT.PARAMETERS['numberOfThreads'] = "4"
BIOGEME_OBJECT.PARAMETERS['moreRobustToNumericalIssues'] = "0"
