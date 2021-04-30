import re
import numpy as np
from ax.core.runner import Runner
from ax.core.metric import Metric
from ax.core.data import Data
from ax.utils.stats.statstools import agresti_coull_sem
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from connectors import *
import logging
import warnings
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

def decorator_factory(conn_params, test_name, mode='full'):
    def decorator(func):
        def wrapper(**kwargs):
            saver_loader_main = SaverLoader_DB(conn_params, test_name)
            experiment, generation_strategy = saver_loader_main.load_full_experiment()
            func_result = func(experiment=experiment,  generation_strategy=generation_strategy, **kwargs)
            if mode == 'full':
                saver_loader_main.save_full_experiment(experiment, generation_strategy)
            return func_result
        return wrapper
    return decorator


class DummyRunner(Runner):
    def run(self, trial):
        index = str(trial.index)
        exp_name = trial.experiment.name
        settings_output = {exp_name+'_trial'+index+'_arm'+arm.name:{'parameters':arm.parameters, 'weight':weight} for
                           arm, weight in trial.normalized_arm_weights().items()}
        return settings_output
register_runner(DummyRunner)


def get_tags(trial, metric_class_name, mode):
    if metric_class_name == 'MultiMetric':
        if trial.experiment.description['module'] == 'bayesian_optimization':
            trial_tags = [trial.experiment.name + '_trial' + str(trial.index) + '_arm' + arm.name
                          for arm in trial.arms]
        elif trial.experiment.description['module'] == 'bandit':
            trial_tags = [trial.experiment.name + '_trial' + str(i) + '_arm' + arm.name
                          for arm in trial.arms for i in range(trial.index + 1)]

    elif metric_class_name == 'MultiDummyMetric':
        trial_tags = [arm.name for arm in trial.arms]

    return trial_tags if mode == 'full' else list({re.sub(r'.*arm', '', i) for i in trial_tags})


def verify_n(experiment):
    ## verifying we have enough users
    last_trial = list(experiment.data_by_trial)[-1]
    last_data = list(experiment.data_by_trial[last_trial])[-1]
    df_to_check = experiment.data_by_trial[last_trial][last_data].df

    metrics_to_check = []
    for i in experiment.optimization_config.objective.metric_weights:
        if i[1] > 0:
            metrics_to_check += [i[0].name]

    df_to_check = df_to_check[df_to_check['metric_name'].isin(metrics_to_check)]
    min_users_dict = oraculum_config['min_users_dict']
    min_users_dict_detail = {i: min_users_dict[i.split('_')[-2]] for i in metrics_to_check}
    enough_users = all(df_to_check['n'] > df_to_check['metric_name'].replace(min_users_dict_detail))
    return enough_users


def create_metric_result_df(trial, metric_class_name, metric, day):
    return pd.DataFrame({'tags': get_tags(trial,
                                          metric_class_name=metric_class_name,
                                          mode='arms'),
                         'metric_name': metric + '_' + str(day),
                         'mean': np.nan,
                         'sem': np.nan,
                         'n': 0})


def ret_stat(df, days, trial, metric_class_name):
    metric = 'Retention'
    result_df = pd.DataFrame()

    for day in days:
        day = int(day)

        metric_result_df = create_metric_result_df(trial, metric_class_name, metric, day).set_index('tags')

        regs = df[df['lifetime_days'] >= day].groupby('tags')['client_id'].nunique().reset_index(name='n')
        rets = df[df['report_day'] == day].groupby('tags')['client_id'].nunique().reset_index(name='returned')
        ret_stat = pd.merge(regs, rets, how='left', on='tags')
        ret_stat['sem'] = agresti_coull_sem(ret_stat['returned'], ret_stat['n'])
        ret_stat['mean'] = ret_stat['returned'] / ret_stat['n']
        ret_stat['metric_name'] = metric + '_' + str(day)

        metric_result_df.update(ret_stat[['tags', 'metric_name', 'mean', 'sem', 'n']].set_index('tags'))
        result_df = result_df.append(metric_result_df.reset_index())
    return result_df


def conv_stat(df, days, trial, metric_class_name):
    metric = 'Conversion'
    result_df = pd.DataFrame()
    for day in days:

        ## creating empty metric_result_df
        metric_result_df = create_metric_result_df(trial, metric_class_name, metric, day).set_index('tags')

        if day=='all':
            regs = df.groupby('tags')['client_id'].nunique().reset_index(name = 'n')
            convs = df[df['payments']>0].groupby('tags')['client_id'].nunique().reset_index(name = 'converted')
            conv_stat = pd.merge(regs, convs, how='left', on='tags' )
        else:
            day = int(day)
            regs = df[df['lifetime_days']>=day].groupby('tags')['client_id'].nunique().reset_index(name = 'n')
            convs = df[(df['lifetime_days']>=day) &
                            (df['report_day']<=day) &
                            (df['payments']>0)].groupby('tags')['client_id'].nunique().reset_index(name = 'converted')
            conv_stat = pd.merge(regs, convs, how='left', on='tags' )
        conv_stat['sem'] = agresti_coull_sem(conv_stat['converted'], conv_stat['n'])
        conv_stat['mean'] = conv_stat['converted']/conv_stat['n']
        conv_stat['metric_name'] = metric + '_' + str(day)

        metric_result_df.update(conv_stat[['tags','metric_name','mean','sem','n']].set_index('tags'))
        result_df = result_df.append(metric_result_df.reset_index())
    return result_df


def ARPU_stat(df, days, trial, metric_class_name):
    metric = 'ARPU'
    result_df = pd.DataFrame()
    for day in days:

        metric_result_df = create_metric_result_df(trial, metric_class_name, metric, day).set_index('tags')
        ## this is a fix for pandas bug for empty df agg
        if df.empty: 
            df = df.astype(np.int64)

        if day == 'all':
            payments_stat = df.groupby(['tags','client_id'])['inapp_gross'].sum().reset_index()
        else:
            day = int(day)
            payments_stat = df[(df['lifetime_days'] >= day) &
                               ((df['report_day'] <= day) |
                                (df['report_day'].isna()))].groupby(['tags', 'client_id']
                                                                    )['inapp_gross'].sum().reset_index()

        ## creating final_table
        payments_stat_agg  = payments_stat.groupby('tags').agg({'inapp_gross': 'mean',
                                                                'client_id': 'nunique'}).reset_index()
        payments_stat_agg.columns = ['tags', 'mean', 'n']

        ## bootstraping for SEM
        payments_stat_agg['sem'] = np.nan
        for i in payments_stat['tags'].unique():
            tag_subset = payments_stat[payments_stat['tags']==i]['inapp_gross']
            bootstraped_means = [np.mean(np.random.choice(tag_subset,
                                                      len(tag_subset),
                                                      replace=True)) for i in range(10000)]
            payments_stat_agg.loc[payments_stat_agg['tags']==i, 'sem'] = np.std(bootstraped_means)
        payments_stat_agg['metric_name'] = metric + '_' + str(day)
        payments_stat_agg = payments_stat_agg.replace({0: np.nan})

        metric_result_df.update(payments_stat_agg[['tags', 'metric_name', 'mean', 'sem', 'n']].set_index('tags'))
        result_df = result_df.append(metric_result_df.reset_index())
    return result_df


def get_all_stat(df, trial, metrics_list, func_dict, metric_class_name):
    result_df = pd.DataFrame()
    unique_mset = sorted(set([i.split('_')[-2] for i in metrics_list]))
    grouped_mdict = {x:[y.split('_')[-1] for y in sorted(metrics_list) if x in y] for x in unique_mset}
    for key, value in grouped_mdict.items():
        result_df = result_df.append(func_dict[key](df, value, trial, metric_class_name))
    result_df = result_df.rename(columns={'tags': 'arm_name'})
    result_df['trial_index'] = trial.index
    result_df['mean'] = result_df['mean'].fillna(0)
    result_df['sem'] = result_df.groupby('metric_name')['sem'].transform(lambda x: x.fillna(0 if np.isnan(x.mean()) else x.mean()))
    return result_df


def dummy_get_stat(trial_num, arm_weight, arm_name, arm_params, mode):
    current_n = 1000
    if mode == 'retention':
        current_arm_succesess = np.random.binomial(current_n, 0.4)
        mean = np.abs(current_arm_succesess) / current_n
        sem = agresti_coull_sem(np.abs(current_arm_succesess), current_n)

    elif mode == 'conversion':
        current_arm_succesess = np.random.binomial(current_n, 0.05)
        mean = np.abs(current_arm_succesess) / current_n
        sem = agresti_coull_sem(np.abs(current_arm_succesess), current_n)

    elif mode == 'arpu':
        beta_vals = np.random.beta(0.01, 1, 1000) * 200
        mean = beta_vals.mean()
        sem = beta_vals.std() / np.sqrt(current_n)

    return current_n, mean, sem


class DummyMetric(Metric):
    def fetch_trial_data(self, trial):
        current_data = []
        for arm, arm_weight in trial.normalized_arm_weights().items():
            arm_name = arm.name
            arm_params = arm.parameters
            trial_num = trial.index

            n, mean, sem = dummy_get_stat(trial_num=trial_num,
                                          arm_weight=arm_weight,
                                          arm_name=arm_name,
                                          arm_params=arm_params,
                                          mode=self.name)

            current_data.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "mean": mean,
                    "sem": sem,
                    "trial_index": trial_num,
                    "n": n,
                }
            )

        return Data(df=pd.DataFrame.from_records(current_data))
register_metric(DummyMetric)


class MultiDummyMetric(Metric):
    @classmethod
    def fetch_trial_data_multi(cls, trial, metrics):
        exp_project = trial.experiment.description['project']
        db_project = oraculum_config['project_dict'][exp_project]['project']
        platforms = trial.experiment.description['platform']
        if platforms == ['all']:
            platforms = oraculum_config['project_dict'][exp_project]['platforms']
        if len(platforms) == 1:
            platforms += ['tuple_dummy']

        fake_trial_kpi_query = f"""
        """

        rds_kpi = rds_sql_query('rds_db', fake_trial_kpi_query)
        rds_kpi[['inapp_gross', 'payments']] = rds_kpi[['inapp_gross', 'payments']].fillna(0)
        rds_kpi = rds_kpi[rds_kpi['report_day'] >= 0]

        ## assigning fake tags
        tags_weights = {
            trial.experiment.name +
            '_trail' + str(trial.index) + '_arm' +
            arm.name: weight for arm, weight in trial.normalized_arm_weights().items()
        }
        fake_tags = np.random.choice(list(tags_weights.keys()), rds_kpi['client_id'].nunique(),
                                     p=list(tags_weights.values()))
        clients_ids = rds_kpi[['client_id']].drop_duplicates()
        clients_ids['tags'] = fake_tags
        rds_kpi = pd.merge(rds_kpi, clients_ids, on='client_id')

        ## changing tags to arm names
        rds_kpi['tags'] = rds_kpi['tags'].str.replace(r'.*arm', '')

        ## preparing other info
        metrics_list = [i.name for i in metrics]
        func_dict = {'ARPU': ARPU_stat, 'Conversion': conv_stat, 'Retention': ret_stat}
        result_df_current_trial = get_all_stat(rds_kpi, trial, metrics_list, func_dict, cls.__name__)

        return Data(df=result_df_current_trial)
register_metric(MultiDummyMetric)


class MultiMetric(Metric):
    @classmethod
    def fetch_trial_data_multi(cls, trial, metrics):
        tags_list = get_tags(trial, metric_class_name=cls.__name__, mode='full')
        exp_project = trial.experiment.description['project']
        db_project = oraculum_config['project_dict'][exp_project]['project']
        platforms = trial.experiment.description['platform']
        if platforms == ['all']:
            platforms = oraculum_config['project_dict'][exp_project]['platforms']
        if len(platforms) == 1:
            platforms += ['tuple_dummy']
        if 'offset_date' in trial.experiment.description.keys():
            starting_date = trial.experiment.description['offset_date'].strftime('%Y-%m-%d')
        else:
            starting_date = trial.experiment.time_created.strftime('%Y-%m-%d')

        trial_kpi_query = f"""
        """
        rds_kpi = rds_sql_query('rds_db', trial_kpi_query)
        rds_kpi[['inapp_gross', 'payments']] = rds_kpi[['inapp_gross', 'payments']].fillna(0)
        rds_kpi = rds_kpi[(rds_kpi['report_day'] >= 0) | (rds_kpi['report_day'].isna())]

        ## changing tags to arm names
        rds_kpi['tags'] = rds_kpi['tags'].str.replace(r'.*arm', '')

        ## preparing other info
        metrics_list = [i.name for i in metrics]
        func_dict = {'ARPU': ARPU_stat, 'Conversion': conv_stat, 'Retention': ret_stat}
        result_df_current_trial = get_all_stat(rds_kpi, trial, metrics_list, func_dict, cls.__name__)

        return Data(df=result_df_current_trial)
register_metric(MultiMetric)


oraculum_config['metrics_dict'] = {'fake': MultiDummyMetric, 'real': MultiMetric}