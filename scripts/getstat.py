import sys
import itertools
from plotly.offline import plot
from scipy import stats
sys.path.append('../../src')
sys.path.append('../src')
sys.path.append('./src')
from ax.service.utils.best_point import _get_best_row_for_scalarized_objective
from ax.plot.bandit_rollout import plot_bandit_rollout
from ax.plot.marginal_effects import plot_marginal_effects
from ax.plot.scatter import plot_fitted
from ax.plot.slice import plot_slice
from ax.plot.contour import plot_contour
from lib import *


def make_stat(input_data, experiment, generation_strategy):
    def prepare_data_for_plots():
        ## preparing data
        res = pd.DataFrame()
        for trial in experiment.data_by_trial:
            last_entry = list(experiment.data_by_trial[trial].keys())[-1]
            res = res.append(experiment.data_by_trial[trial][last_entry].df)
        res = res.sort_values(by=['trial_index', 'arm_name', 'metric_name']
                              ).drop_duplicates(subset=['arm_name', 'metric_name'], keep='last')
        generation_strategy._set_or_update_model(data=Data(df=res))

    def prepare_data_for_stat_table():
        res = pd.DataFrame()
        for trial in experiment.data_by_trial:
            last_entry = list(experiment.data_by_trial[trial].keys())[-1]
            res = res.append(experiment.data_by_trial[trial][last_entry].df)
        arm_dict = pd.DataFrame(experiment.arms_by_name.items())
        arm_dict[1] = arm_dict[1].apply(lambda x: x.parameters)
        arm_dict.columns = ['arm_name', 'arm_params']
        arm_table = pd.merge(res, arm_dict, on='arm_name', how='left')
        arm_table['arm_params'] = arm_table['arm_params'].apply(json.dumps)
        return arm_table

    def get_mean_diff(x):
        group_vals = np.array(x)
        diff = np.subtract.outer(group_vals, group_vals)
        diff_clean = diff[~np.eye(diff.shape[0], dtype=bool)].reshape(len(group_vals), len(group_vals) - 1)
        return diff_clean.mean(axis=1)

    def get_winner_confidence(arm_table, control_index, best_index, metric):
        if 'ARPU' in metric:
            mu1, sem1, n1 = arm_table.loc[best_index, ['mean', 'sem', 'n']].values.T
            mu2, sem2, n2 = arm_table.loc[control_index, ['mean', 'sem', 'n']].values.T
            std1 = sem1 * np.sqrt(n1)
            std2 = sem2 * np.sqrt(n2)
            satSE = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
            Z = (mu1 - mu2) / satSE
            pval = stats.t.sf(np.abs(Z), n1 + n2 - 2) * 2

        else:
            p1, n1 = arm_table.loc[best_index][['mean', 'n']].values.T
            p2, n2 = arm_table.loc[control_index][['mean', 'n']].values.T
            SE = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
            Z = (p1 - p2) / SE
            pval = stats.norm.sf(Z) * 2

        confidence = 1 - pval
        return Z, pval, confidence

    if input_data['query_type'] == 'experiment_info':
        exp_json = object_to_json(experiment)
        end_time = ''
        if not [trial for stage_type in [0, 1, 4] for trial in experiment.trials_by_status[stage_type]]:
            try:
                end_time = experiment.trials_by_status[3][-1].time_completed.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        experiment_info = {
            'experiment_name': exp_json['name'],
            'experiment_description': exp_json['description'],
            'status': {trial: data['status'] for trial, data in exp_json['trials'].items()},
            'tеst_started': exp_json['time_created']['value'],
            'test_ended': end_time,
            'ends_approx': '????'}
        return experiment_info

    # TODO: show dates on x axis for rollout plot
    elif input_data['query_type'] == 'rollout_plot':
        plot_config = plot_bandit_rollout(experiment).data
        plot_config['layout']['showlegend'] = True
        rollout_plot = plot(plot_config, show_link=False, output_type='div')
        return rollout_plot

    elif input_data['query_type'] == 'marginal_effects_plots':
        prepare_data_for_plots()
        me_plots = {}
        params_list = list(experiment.parameters.keys())
        for metric in experiment.metrics.keys():
            try:
                if experiment.description['module'] == 'bayesian_optimization':
                    if len(params_list) == 1:
                        plot_config = plot_slice(model=generation_strategy.model,
                                                 param_name=params_list[0],
                                                 metric_name=metric).data
                        slice_plot = plot(plot_config, show_link=False, output_type='div')
                        me_plots[metric] = slice_plot
                    elif len(params_list) >= 2:
                        for i, j in itertools.combinations(params_list, 2):
                            plot_config = plot_contour(model=generation_strategy.model,
                                                       param_x=i,
                                                       param_y=j,
                                                       metric_name=metric).data
                            counter_plot = plot(plot_config, show_link=False, output_type='div')
                            me_plots[metric+'_'+i+'_'+j] = counter_plot

                elif experiment.description['module'] == 'bandit':
                    plot_config = plot_marginal_effects(model=generation_strategy.model, metric=metric).data
                    marginal_effects_plot = plot(plot_config, show_link=False, output_type='div')
                    me_plots[metric] = marginal_effects_plot
            except(ValueError, KeyError) as e:
                me_plots[metric] = str(e)
        return me_plots

    elif input_data['query_type'] == 'predicted_outcomes_plots':
        prepare_data_for_plots()
        me_plots = {}
        for metric in experiment.metrics.keys():
            try:
                plot_config = plot_fitted(model=generation_strategy.model, metric=metric, rel=False).data
                predicted_outcomes_plot = plot(plot_config, show_link=False, output_type='div')
                me_plots[metric] = predicted_outcomes_plot
            except(ValueError, KeyError) as e:
                me_plots[metric] = str(e)
        return me_plots

    elif input_data['query_type'] == 'full_result_table':
        try:
            arm_table = prepare_data_for_stat_table()
            arm_table = pd.pivot_table(arm_table,
                                       index=['trial_index', 'arm_name', 'arm_params'],
                                       columns='metric_name',
                                       values=['mean', 'sem', 'n'])
            arm_table.columns = ['_'.join(i) for i in arm_table.columns]
            arm_table = arm_table.reset_index()
            return arm_table.to_html()
        except:
            arm_table = pd.DataFrame()
            return arm_table.to_html()

    elif input_data['query_type'] == 'last_trial_statistics':
        try:
            arm_table = prepare_data_for_stat_table()
            metric_weight = {metric.name: weight for metric, weight in
                             experiment.optimization_config.objective.metric_weights}
            arm_table['weight'] = arm_table['metric_name'].replace(metric_weight)
            arm_table = arm_table.sort_values(by='trial_index', ascending=True).drop_duplicates(
                subset=['arm_name', 'metric_name'], keep='last')
            arm_table['diffs'] = arm_table.groupby('metric_name')['mean'].transform(get_mean_diff)
            arm_table['share'] = arm_table['diffs'] / (arm_table['mean'] - arm_table['diffs'])
            arm_table['type'] = np.where(arm_table['weight'] > 0, 'main', 'other')

            def z_scaler(x):
                return (x - np.mean(x)) / np.std(x)
            arm_table_scaled = arm_table.copy()
            arm_table_scaled['mean'] = arm_table_scaled.groupby('metric_name')['mean'].transform(z_scaler)
            best_row = _get_best_row_for_scalarized_objective(arm_table_scaled,
                                                              experiment.optimization_config.objective)
            arm_table['best_arm'] = np.where(arm_table['arm_name'] == best_row['arm_name'], True, False)

            arm_table[['share_control', 'confidence']] = np.nan, np.nan
            if 'control' in arm_table['arm_name'].unique():
                arm_table = pd.merge(arm_table,
                                     arm_table[arm_table['arm_name'] == 'control'][['metric_name', 'mean']],
                                     on=['metric_name'], how='left', suffixes=['', '_control'])
                arm_table['diffs2'] = arm_table['mean'] - arm_table['mean_control']
                arm_table['share_control'] = arm_table['diffs2'] / (arm_table['mean'] - arm_table['diffs2'])
                for metric in arm_table['metric_name'].unique():
                    arm_table_cut = arm_table[arm_table['metric_name'].str.contains(metric)]
                    control_index = arm_table_cut[arm_table_cut['arm_name'] == 'control'].index[0]
                    best_index = arm_table_cut[arm_table_cut['best_arm']].index[0]
                    confidence = get_winner_confidence(arm_table=arm_table,
                                                       control_index=control_index,
                                                       best_index=best_index,
                                                       metric=metric)[2]
                    arm_table.loc[best_index, 'confidence'] = confidence

            arm_table = arm_table[['arm_name', 'best_arm', 'confidence',
                                   'arm_params', 'type', 'metric_name',
                                   'mean', 'n', 'share', 'share_control']]
            full_list = ['arm_name', 'best_arm', 'arm_params', 'type', 'metric_name']
            arm_table = arm_table.groupby(full_list)[['mean', 'n', 'share', 'share_control', 'confidence']].apply(
                lambda x: x.to_dict('records')[0]).reset_index(name='data')
            for i in range(2):
                agg = full_list.pop()
                arm_table = arm_table.groupby(full_list)[[agg, 'data']].apply(
                    lambda x: x.set_index(agg)['data'].to_dict()).reset_index(name='data')
            arm_table['data'] = arm_table.apply(lambda row: {**row['data'],
                                                             "arm_params": json.loads(row['arm_params']),
                                                             "best_arm": row['best_arm']}, axis=1)
            arm_table = arm_table.set_index('arm_name')['data'].to_json()
            return arm_table
        except:
            return '{}'


if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    if input_data['query_type']=='get_all_tests':
        all_experiments = ExperimentDBConnector(conn_params['experiment_db']).get_all_experiments()
        print(json.dumps({'experiments': all_experiments}))
    else:
        cur_exp_decorator = decorator_factory(conn_params['experiment_db'], input_data['test_name'], mode='load_only')
        decorated_make_stat = cur_exp_decorator(make_stat)
        decorated_make_stat(input_data=input_data)