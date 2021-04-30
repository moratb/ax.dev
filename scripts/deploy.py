import sys
sys.path.append('../../src')
sys.path.append('../src')
sys.path.append('./src')
from lib import *


def deployer(input_data, experiment, generation_strategy):
    if input_data['query_type'] == 'new_experiment':
        running_trial = experiment.trials_by_status[0][0].run()
        return running_trial.run_metadata

    elif input_data['query_type'] == 'next_trial':
        try:
            ## complete RUNNING TRIAL
            experiment.trials_by_status[4][0].complete()
        except IndexError:
            pass

        if experiment.status_quo:
            optimize_for_power = True
        else:
            optimize_for_power = False
        experiment.new_batch_trial(generator_run=generation_strategy.last_generator_run,
                               optimize_for_power=optimize_for_power)
        ## run CANDIDATE TRIAL
        running_trial = experiment.trials_by_status[0][0].run()
        return running_trial.run_metadata

    elif input_data['query_type'] == 'next_manual_trial':
        try:
            ## complete RUNNING TRIAL
            experiment.trials_by_status[4][0].complete()
        except IndexError:
            pass

        ## run CANDIDATE TRIAL
        try:
            running_trial = experiment.trials_by_status[0][0].run()
            return running_trial.run_metadata
        except IndexError:
            pass


    elif input_data['query_type'] == 'end_experiment':
        try:
            experiment.trials_by_status[4][0].complete()
            return str(experiment.trials_by_status)
        except IndexError as e:
            res_str = ' no trial to complete. the experiment likely has ended'
            return str(e)+res_str


if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    cur_exp_decorator = decorator_factory(conn_params['experiment_db'], input_data['test_name'], mode='full')
    decorated_deployer = cur_exp_decorator(deployer)
    decorated_deployer(input_data=input_data)