import numpy as np
import csv

biogeme_dataset_file_name = 'HWW_Queensland.dat'
nlogit_dataset_file_name = 'nlogit_mode_choice_data_HWW.csv'
# data = np.genfromtxt(data_set_file_name, delimiter="\t", names=True)  # skip_header=1)
data = np.genfromtxt(biogeme_dataset_file_name, delimiter="\t", skip_header=1)
headers = np.array(open(biogeme_dataset_file_name, 'r').readline().rstrip().split('\t'))

av_columns = [-6, -5, -4, -3, -2, -1]
availability = data[:, av_columns]
print(headers[av_columns])

columns_dict = {}
for i, name in enumerate(headers): columns_dict[name] = i

output_headers = ('cset,choice,alt_id,label,av,experiment_id,weight,trip_id,o_zone,d_zone,p_zone,' +
                  'a_zone,period,purpose,direction,num_cars_0,num_cars_23,' +
                  'aof,acore,ancore,afr,cost_outward,cost_return').split(',')

cost_variables = [
    ['Bicycle_Cost_Outward', 'Bicycle_Cost_Return'],
    ['Car_Cost_Outward', 'Car_Cost_Return'],
    ['PAWE_Cost_Outward', 'WAPE_Cost_Return'],
    ['KAWE_Cost_Outward', 'WAKE_Cost_Return'],
    ['WAWE_Cost_Outward', 'WAWE_Cost_Return'],
    ['Walk_Cost_Outward', 'Walk_Cost_Return']
]

float_columns = {6, 21, 22}
str_columnls = {3}
def typeriser(i, x):
    if i in float_columns:
        return float(x)
    elif i in str_columnls:
        return x
    else:
        return int(x)

def is_chosen(alt, choice):
    if alt == choice:
        return 1
    else:
        return 0

alt_labels = {
    1: 'Bicycle',
    2: 'Car',
    3: 'KissRide',
    4: 'ParkRide',
    5: 'WalkRide',
    6: 'Walk',
}

output_matrix = np.zeros((np.sum(availability), len(output_headers)))
row_count = -1
for i in range(len(data)):
    for alt in [1, 2, 3, 4, 5, 6]:
        if availability[i, alt - 1] == 1:
            row_count += 1

            cset = np.sum(availability[i])
            choice = is_chosen(alt, data[i, columns_dict['choice']])
            alt_id = alt
            label = 0  # alt_labels[alt]
            av = 1
            experiment_id = data[i, columns_dict['experiment_id']]
            weight = data[i, columns_dict['weight']]
            trip_id = 0
            o_zone = 0
            d_zone = 0
            p_zone = 0
            a_zone = 0
            period = 0
            purpose = 0
            direction = 0
            num_cars_0 = data[i, columns_dict['CarOwnershipConstant0']]
            num_cars_23 = data[i, columns_dict['CarOwnershipConstant23']]
            aof = data[i, columns_dict['a_OuterFrame']]
            acore = data[i, columns_dict['a_CBDCore']]
            ancore = data[i, columns_dict['a_CBDNonCore']]
            afr = data[i, columns_dict['a_CBDFrame']]
            cost_outwards = data[i, columns_dict[cost_variables[alt - 1][0]]]
            cost_return = data[i, columns_dict[cost_variables[alt - 1][1]]]

            output_matrix[row_count] = np.array([
                cset, choice, alt_id, label, av, experiment_id, weight, trip_id,
                o_zone, d_zone, p_zone, a_zone, period, purpose, direction,
                num_cars_0, num_cars_23, aof, acore, ancore, afr, cost_outwards, cost_return,
            ])
        else:
            pass

with open(nlogit_dataset_file_name, 'w', newline='\n') as f:
    csv_writer = csv.writer(f)  # , delimiter=' ')
    # csv_writer.writerow(output_headers)
    for row in output_matrix:
        csv_writer.writerow([typeriser(i, x) for i, x in enumerate(row)])
