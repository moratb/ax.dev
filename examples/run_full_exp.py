import os

get_tests = """
{

    "query_type" : "get_all_tests"
}
"""

input_data_new ="""
{
    "experiment_type": "",
    "project": "",
    "platform": "",
    "user_type": "",
    "traffic_type": "",
    "test_name": "",
    "test_description" : 
    {
        "project": "",
        "platform": ["", ""],
        "user_type": "",
        "traffic_type": "",
        "module": "",
        "main_description":""
    },
    "arms_to_generate" : 10,
    "search_space": 
    {
    "parameters":{
        "abc": {"type":"range", "dtype":"INT", "lower_bound":3, "upper_bound":100},
        "abc2": {"type":"range", "dtype":"FLOAT", "lower_bound":0.03, "upper_bound":0.99}
        }
    },
    "control_group":{"abc":2, "abc2":0.5},
    "metrics_weights":
    {
        "Retention_1":1.0,
        "Retention_3":0.0,
        "Conversion_1":0.0,
        "Conversion_all":1.0,
        "ARPU_all":0
    },
    "evaluation_options":
    {
        "mode":"manual"
    }
}
"""



input_data_deploy1 = """
{
    "test_name" : "bayes_opt_test1",
    "query_type" : "new_experiment"
}
"""

input_data_eval1 = """
{
    "test_name" : "bayes_opt_test1",
    "query_type" : "fetch_data"
}
"""

input_data_eval2 = """
{
    "test_name" : "bayes_opt_test1",
    "query_type" : "generate_new_arms",
    "arms_to_generate": 5
}
"""

input_data_deploy2 = """
{
    "test_name" : "bayes_opt_test1",
    "query_type" : "next_trial"
}
"""

input_data_plots1 = """
{
    "test_name" : "bayes_opt_test1",
    "query_type" : "predicted_outcomes_plots"
}
"""

input_data_plots2 = """
{
    "test_name" : "bayes_opt_test1",
    "query_type" : "marginal_effects_plots"
}
"""



print('all_tests_getter')
os.system(f"""python ../scripts/getstat.py '{get_tests}'""")

print('new_exp111')
os.system(f"""python ../scripts/newexp.py '{input_data_new}'""")

print('deploying trail1')
os.system(f"""python ../scripts/deploy.py '{input_data_deploy1}'""")

print('eval trail1')
os.system(f"""python ../scripts/eval.py '{input_data_eval1}'""")

print('gen_new_arms')
os.system(f"""python ../scripts/eval.py '{input_data_eval2}'""")

print('deploy2')
os.system(f"""python ../scripts/deploy.py '{input_data_deploy2}'""")

print('eval trail2')
os.system(f"""python ../scripts/eval.py '{input_data_eval1}'""")

print('gen_new_arms2')
os.system(f"""python ../scripts/eval.py '{input_data_eval2}'""")


print('get predicted_outcomes_plots')
os.system(f"""python ../scripts/getstat.py '{input_data_plots1}'""")

print('get marginal effects plots')
os.system(f"""python ../scripts/getstat.py '{input_data_plots2}'""")
