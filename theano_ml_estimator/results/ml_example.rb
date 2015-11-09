class Param
  attr_reader :name
  attr_accessor :value

  def initialize(name,initial_value = 0.0)
    @name = name
    @value = initial_value
  end

  def is_random?
    false
  end
end

class EvRandomParam < Param
  def is_random?
    true
  end

  def draw_randomly
    #r = rand()
    #r = e^(-e^(-vx))
    #e^(-vx) = -ln(r)
    #x = -ln(-ln(r)) / v
    #-Math.log(-Math.log(rand())) / @value
    -Math.log(-Math.log(rand()))
  end

  #def prob_density_grad(x)
    #exp_vx = Math.exp(-@value * x)
    #@value * x * exp_vx * Math.exp(-exp_vx) * (exp_vx - 1)
  #end
end

class Alternative
  attr_reader :chosen
  def initialize(features,chosen)
    @features = features
    @chosen = chosen
  end

  def feature_value(feature_name)
    @features.fetch(feature_name,0.0)
  end

  def utility(params_by_name)
    u = 0.0
    @features.each { |feature_name, feature_value|
      u += params_by_name.fetch(feature_name, 0.0) * feature_value
    }
    u
  end
end

class Observation
  attr_accessor :alternatives

  def initialize(alternatives)
    @alternatives = alternatives
  end
end

class RandomParamsLogitEstimator
  def self.estimate(observations, params, num_draws, learning_rate, dampening_power)
    summed_steps = params.map { |_| 1.0 }
    log_likelihood = 0.0
    total_log_likelihood = 0.0

    last_1000 = []
    sum_last_1000 = 0.0
    srand(42)
    observations.shuffle.each_with_index { |observation,obs_index|


      numerator   = params.map { |_| 0.0 }
      denominator = numerator.dup
      sum_likelihood = 0.0

      1.upto(num_draws) {
        simulated_params = params.map { |param|
          if param.is_random?
            param.draw_randomly
          else
            1.0
          end
        }
        params_by_name = Hash[*(simulated_params.zip(params).map { |val,param| [param.name, param.value * val] }.flatten)]
        utilities = observation.alternatives.map { |alt|
          alt.utility(params_by_name)
        }
        #p utilities

        exp_utilities = utilities.map { |u| Math.exp(u) }
        sum_exp = exp_utilities.inject(0.0) { |sum,e| sum + e }
        probabilities = exp_utilities.map { |u| u / sum_exp }
        #p probabilities

        errors = observation.alternatives.zip(probabilities).map { |alt,prob|
          alt.chosen - prob
        }
        #p errors
        chosen_probability = observation.alternatives.zip(probabilities).select { |alt,prob|
          alt.chosen.to_i == 1
        }.first.last
        #p chosen_probability
        sum_likelihood += chosen_probability


        params.zip(simulated_params).each_with_index  { |(param,simulated_param),index|
          #if param.is_random?
            #numerator[index]   += chosen_probability * simulated_param * alt.feature_value(param.name) * error
            #denominator[index] += chosen_probability
          #else
          observation.alternatives.zip(errors).each { |alt, error|
            numerator[index]   += chosen_probability * simulated_param * alt.feature_value(param.name) * error
            denominator[index] += chosen_probability
          }
          #end

        }

      }

      log_likelihood = Math.log(sum_likelihood / num_draws)
      total_log_likelihood += log_likelihood

      if last_1000.size == 1000
        sum_last_1000 -= last_1000.first
        last_1000 = last_1000.drop(1)
      end
      last_1000 << log_likelihood
      sum_last_1000 += log_likelihood
      if obs_index % 1000 == 0
        puts "Mean (last #{last_1000.size}) after #{obs_index} log likelihood: #{sum_last_1000 / last_1000.size}"
        puts "Mean so far: #{total_log_likelihood / (obs_index+1)}"
        p params.map(&:value)
        #p summed_steps
      end
      #numerator.map! { |n| n / num_draws }
      #denominator.map! { |d| d / num_draws }
      #p numerator
      #p denominator
      gradient = numerator.zip(denominator).map { |n,d| n / d }
      step = gradient.zip(summed_steps).map { |grad,s|
        grad * learning_rate / (s ** dampening_power)
      }
      summed_steps = summed_steps.zip(step).map { |ss,s| ss + s.abs }
      params.zip(step).each { |param,s|
        param.value += s
      }
      #p params.map(&:value)
    }
  end
end