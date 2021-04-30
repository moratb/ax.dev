import sys
sys.path.append('../../src')
sys.path.append('../src')
sys.path.append('./src')
from lib import *


def evaluator(input_data, experiment, generation_strategy):
    def fetch_fresh_data(experiment):
        last_trial = max(experiment.trials.keys())
        if experiment.description['module'] == 'bayesian_optimization':
            result_data = experiment.fetch_data()
        elif experiment.description['module'] == 'bandit':
            result_data = experiment.trials[last_trial].fetch_data()
        experiment.attach_data(result_data)
        return result_data

    if input_data['query_type'] == 'fetch_data':
        if 'offset_date' in input_data.keys():
            experiment.description['offset_date'] = dt.datetime.strptime(input_data['offset_date'], '%Y-%m-%d')
        result_data = fetch_fresh_data(experiment)
        res = result_data.df
        return res.to_html()

    elif input_data['query_type'] == 'generate_new_arms':
        if verify_n(experiment):
            result_data = fetch_fresh_data(experiment)
            try:
                generation_strategy.gen(experiment=experiment,
                                        data=result_data,
                                        n=input_data['arms_to_generate'])
                group_gen = (n for n in range(len(generation_strategy.last_generator_run.arm_weights.items()) + 2))
                new_arms = {object_to_json(arm)['name'] or
                            str(max(list(experiment.trials.keys()))+1) + '_' + str(next(group_gen)):
                            {'parameters': object_to_json(arm)['parameters'], 'weight': weight} for
                            arm, weight in generation_strategy.last_generator_run.arm_weights.items()}

                expected_improvement = str(generation_strategy.last_generator_run.gen_metadata)
                return pd.DataFrame(new_arms).T.reset_index().to_html() + '/n ' + expected_improvement
            ## in case a metric is missing
            except ValueError as e:
                return str(e)
        else:
            res_str = 'Not enough users for one of the metrics!'
            return res_str


if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    cur_exp_decorator = decorator_factory(conn_params['experiment_db'], input_data['test_name'], mode='full')
    decorated_evaluator = cur_exp_decorator(evaluator)
    decorated_evaluator(input_data=input_data)