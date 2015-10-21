require_relative 'estimate'

# set up some parameters
# car cost, walk cost, pt cost, bike cost
# car...num_cars 0,1
# car...cbd_core, cbd_non_core

# ECs - PT

parameters = [
    # Param.new('CarCost',-0.31),
    # Param.new('WalkCost',-0.23),
    # Param.new('PtCost', -0.22),
    # Param.new('BikeCost',-0.28),
    # Param.new('PTWA_Const', 0.8),
    # Param.new('PTPA_Const', -0.55),
    # Param.new('PTKA_Const', -1.8),
    # Param.new('Car_Const', 0.83),
    # Param.new('Bike_Const', -2.4),
    # Param.new('Car_0CarHH', -3.5),
    # Param.new('Car_1CarHH', -1.2),
    # Param.new('Car_CBDCore', -1.95),
    # Param.new('Car_CBDNonCore', -1.1),

    Param.new('CarCost',-0.39),
    Param.new('WalkCost',-0.26),
    Param.new('PtCost', -0.25),
    Param.new('BikeCost',-0.37),
    Param.new('PTWA_Const', 1.0),
    Param.new('PTPA_Const', -0.45),
    Param.new('PTKA_Const', -1.7),
    Param.new('Car_Const', 0.75),
    Param.new('Bike_Const', -2.4),
    Param.new('Car_0CarHH', -3.5),
    Param.new('Car_1CarHH', -1.3),
    Param.new('Car_CBDCore', -2.1),
    Param.new('Car_CBDNonCore', -1.1),

    EvRandomParam.new('EC_PT', -0.53),

    EvRandomParam.new('EC_Car', -0.1),
    EvRandomParam.new('EC_NonMech', 0.33),
    EvRandomParam.new('EC_Bike', -0.61),
    EvRandomParam.new('RP_PtCost', -0.28),
    EvRandomParam.new('RP_WalkCost', -0.11),
    EvRandomParam.new('RP_CarCost', -0.14),
    EvRandomParam.new('RP_BikeCost', -0.3),
    EvRandomParam.new('EC_PTKA', -0.2),
]

cost_param = {
    'PT_Walk_Access' => 'PtCost',
    'PT_Park_Access' => 'PtCost',
    'PT_Kiss_Access' => 'PtCost',
    'Walk'           => 'WalkCost',
    'Bicycle'        => 'BikeCost',
    'Car'            => 'CarCost'
}

rand_cost_params = {
    'PT_Walk_Access' => 'RP_PtCost',
    'PT_Park_Access' => 'RP_PtCost',
    'PT_Kiss_Access' => 'RP_PtCost',
    'Walk'           => 'RP_WalkCost',
    'Bicycle'        => 'RP_BikeCost',
    'Car'            => 'RP_CarCost'
}

ascs = {
    'PT_Walk_Access' => 'PTWA_Const',
    'PT_Park_Access' => 'PTPA_Const',
    'PT_Kiss_Access' => 'PTKA_Const',
    'Bicycle'        => 'Bike_Const',
    'Car'            => 'Car_Const'
}

ec_alts = { 'EC_PT' => ['PT_Walk_Access','PT_Park_Access','PT_Kiss_Access'],
            'EC_PTKA' => ['PT_Kiss_Access'],
            'EC_Car' => ['Car','PT_Park_Access','PT_Kiss_Access'],
            'EC_NonMech' => ['Walk','Bicycle'],
            'EC_Bike'    => ['Bicycle']}

observations = nil

File.open('/Users/timveitch/Documents/VLC/RandomParamsLogitEstimation/nlogit_mode_choice_data_HWW_headers.csv','r') { |file|
  header = file.readline.strip.split(',')
  RowType = Struct.new(*(header.map(&:to_sym)))
  rows = file.readlines.map { |row| RowType.new(*(row.strip.split(','))) } #59243 #9011 # [0..59243]

  observations = rows.group_by { |row| row.experiment_id }.map { |exp_id, alts|
    alternatives = alts.map { |alt|
      feature_hash = {}

      feature_hash[cost_param.fetch(alt.label)] = alt.Cost.to_f / 100.0

      if rand_cost_params.has_key?(alt.label)
        feature_hash[rand_cost_params.fetch(alt.label)] = alt.Cost.to_f / 500.0
      end

      ec_alts.each { |ec_param,alts_for_ec|
        feature_hash[ec_param] = 1.0 if alts_for_ec.include?(alt.label)
      }

      feature_hash[ascs[alt.label]] = 1.0 if ascs.has_key?(alt.label)

      if alt.label == 'Car'
        feature_hash['Car_0CarHH'] = alt.num_cars_0.to_f
        feature_hash['Car_1CarHH'] = alt.num_cars_1.to_f
        feature_hash['Car_CBDCore'] = alt.acore.to_f
        feature_hash['Car_CBDNonCore'] = alt.ancore.to_f
      end
      Alternative.new(feature_hash, alt.choice.to_i)
    }
    Observation.new(alternatives)
  }
}

RandomParamsLogitEstimator.estimate(observations, parameters, 1000,0.025, 1.0)

parameters.each { |param|
  puts "Param #{param.name} => #{param.value}"
}
# read some data to create some observations, each with alternatives, that have features

