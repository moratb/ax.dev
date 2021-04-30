import sys
from ax import *
from ax.core.objective import ScalarizedObjective
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.generation_strategy import GenerationStep
sys.path.append('../../src')
sys.path.append('../src')
sys.path.append('./src')
from lib import *


def parse_search_space(search_space_dict):
    parameters_list = []
    parameter_types = {"INT": ParameterType.INT, "FLOAT": ParameterType.FLOAT, "STRING": ParameterType.STRING}
    for i, j in search_space_dict['parameters'].items():
        if j['type'] == 'range':
            tmp_parameter = RangeParameter(name=i,
                                           lower=j['lower_bound'],
                                           upper=j['upper_bound'],
                                           parameter_type=parameter_types[j['dtype']])
            parameters_list += [tmp_parameter]
        elif j['type'] == 'choice':
            tmp_parameter = ChoiceParameter(name=i,
                                            values=j['values'],
                                            parameter_type=parameter_types[j['dtype']])
            parameters_list += [tmp_parameter]

    search_space = SearchSpace(parameters=parameters_list)
    return search_space


def create_new_experiment(input_data, runner, metric, saver_loader):
    ## parse search space
    search_space = parse_search_space(input_data['search_space'])
    
    ## define experiment
    experiment = Experiment(name=input_data['test_name'],
                            search_space=search_space,
                            description=input_data['test_description'])
    
    ## set control_group
    if input_data['control_group']:
        experiment.status_quo = Arm(name="control", 
                                    parameters=input_data['control_group'])
    else:
        pass
    
    ## create objectives
    metrics = []
    weights = []
    for i, j in input_data['metrics_weights'].items():
        metrics += [metric(name=i, lower_is_better=False)]
        weights += [j]
    
    main_objective = ScalarizedObjective(metrics=metrics, 
                                         weights=weights,
                                         minimize=False)
    
    optimization_config = OptimizationConfig(objective=main_objective)
    experiment.optimization_config = optimization_config
    
    ## create generator strategy
    if input_data['arms_to_generate'] == -1:
        generation_step0_model = Models.FACTORIAL
    else:
        generation_step0_model = Models.SOBOL

    if input_data['test_description']['module'] == 'bayesian_optimization':
        if 'choice' in [j['type'] for i, j in input_data['search_space']['parameters'].items()]:
            return 'choice param not implemented for bayesian opt'
        else:
            generation_step1_model = Models.BOTORCH
    elif input_data['test_description']['module'] == 'bandit':
        generation_step1_model = Models.THOMPSON
    
    generation_strategy = GenerationStrategy(
        steps=[
            GenerationStep(model=generation_step0_model,
                           num_trials=1),
            GenerationStep(model=generation_step1_model,
                           num_trials=-1,
                           model_kwargs={'min_weight': 0.01}),
        ]
    )
    
    ## generate primary arms
    generation_strategy.gen(experiment=experiment,
                            search_space=search_space,
                            n=input_data['arms_to_generate'])
    
    
    ## Runners can also be manually added to a trial to override the experiment default.
    experiment.runner = runner()
    
    ## create first trial with starting arms
    if input_data['control_group']:
        optimize_for_power = True
    else:
        optimize_for_power = False
    
    experiment.new_batch_trial(generator_run=generation_strategy.last_generator_run,
                               optimize_for_power=optimize_for_power)
    
    ## save experiment
    saver_loader.save_full_experiment(experiment, generation_strategy)
    
    ## return information
    exp_json = object_to_json(experiment)
    experiment_metadata = {
        'experiment_name': exp_json['name'],
        'experiment_description': exp_json['description'],
        'search_space': exp_json['search_space'],
        'trial0_arms': {object_to_json(arm)['name']: {'parameters': object_to_json(arm)['parameters'],
                                                      'weight': weight}
                        for arm, weight in experiment.trials[0].normalized_arm_weights().items()},
        'optimization_config': exp_json['optimization_config'],
        'control_group': exp_json['status_quo'],
        'runner': exp_json['runner'],
        'time_created': exp_json['time_created']['value']
    }
    return experiment_metadata


if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    saver_loader_main = SaverLoader_DB(conn_params['experiment_db'], input_data['test_name'])
    saver_loader_main.connector.create_tables()
    create_new_experiment(input_data,
                          runner=DummyRunner,
                          metric=oraculum_config['metrics_dict'][input_data['experiment_type']],
                          saver_loader=saver_loader_main)