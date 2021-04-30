import psycopg2
import psycopg2.extras
import json
import datetime as dt
import pandas as pd
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.decoder import object_from_json
from config import *


class ExperimentDBConnector:
    def __init__(self, conn_params):
        self.conn_params = conn_params
        self.table_dict = {'Experiment': 'experiments',
                           'GenerationStrategy': 'generation_strategies'}
        
    def _get_connection(self):
        conn = psycopg2.connect(**self.conn_params)
        return conn
    
    def create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor() 
        
        for table in self.table_dict.values():
            query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                experiment_name VARCHAR(255) NOT NULL,
                object_json json NOT NULL,
                ins_ts TIMESTAMP not NULL, 
                PRIMARY KEY (experiment_name)
            )
            """
            cursor.execute(query)
            
        conn.commit()
        conn.close()
        return None
        
    def insert_object(self, object_dict, experiment_name):
        conn = self._get_connection()
        cursor = conn.cursor() 
        table_name = self.table_dict[object_dict['__type']]
        object_json = json.dumps(object_dict).replace('\'', '\''*2)
        
        query = f"""
        INSERT INTO {table_name} (experiment_name, object_json, ins_ts)
        VALUES('{experiment_name}', '{object_json}', '{dt.datetime.now()}')
        ON CONFLICT (experiment_name) DO UPDATE
            SET object_json = excluded.object_json,
                ins_ts = excluded.ins_ts;
        """
        
        cursor.execute(query)
        conn.commit()
        conn.close()
        return None

    def extract_object(self, object_ax_type, experiment_name):
        conn = self._get_connection()
        cursor = conn.cursor() 
        table_name = self.table_dict[object_ax_type]
        
        query = f"""
        SELECT object_json
        FROM {table_name}
        WHERE experiment_name = '{experiment_name}'
        """

        cursor.execute(query)
        object_dict = cursor.fetchall()[0][0]
        conn.close()
        return object_dict

    def get_all_experiments(self):
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        tables = list(self.table_dict.values())

        query = f"""
        SELECT 
        experiment_name, 
        {tables[0]}.object_json as exp_json,
        {tables[1]}.object_json as gs_json
        FROM {tables[0]} left join {tables[1]}
        USING (experiment_name)
        """

        cursor.execute(query)
        object_dict = cursor.fetchall()
        object_dict = [dict(record) for record in object_dict]
        conn.close()
        return object_dict

        
class SaverLoader_DB:
    def __init__(self, conn_params, test_name):
        self.test_name = test_name
        self.connector = ExperimentDBConnector(conn_params)

    def save_object(self, object_ax):
        object_dict = object_to_json(object_ax)
        self.connector.insert_object(object_dict, self.test_name)
        return None
    
    def load_object(self, object_ax_type):
        object_dict = self.connector.extract_object(object_ax_type, self.test_name)
        if object_ax_type == 'GenerationStrategy':
            object_dict['had_initialized_model'] = False
        object_ax = object_from_json(object_dict)
        return object_ax
    
    def save_full_experiment(self, experiment, generation_strategy):
        self.save_object(experiment)
        self.save_object(generation_strategy)
        return None
        
    def load_full_experiment(self):
        experiment = self.load_object('Experiment')
        generation_strategy = self.load_object('GenerationStrategy')
        
        ## loading actualy stores generation_strategy.experiment and experiment 
        ## into different memory parts. we need to connect them again
        generation_strategy.experiment = experiment
        return experiment, generation_strategy


class SaverLoader_json:
    def __init__(self, filepath, test_name):
        self.fp_main = filepath + test_name
        self.fp_suffix_dict = {'Experiment': '_exp.json',
                               'GenerationStrategy': '_gs.json'}

    def save_object(self, object_ax):
        object_dict = object_to_json(object_ax)
        filepath = self.fp_main + self.fp_suffix_dict[object_dict['__type']]
        with open(filepath, "w+") as file:
            file.write(json.dumps(object_dict))
        return None
    
    def load_object(self, object_ax_type):
        filepath = self.fp_main + self.fp_suffix_dict[object_ax_type]
        with open(filepath, "r") as file:
            object_dict = json.loads(file.read())
            if object_ax_type == 'GenerationStrategy':
                object_dict['had_initialized_model'] = False
        object_ax = object_from_json(object_dict)
        return object_ax
    
    def save_full_experiment(self, experiment, generation_strategy):
        self.save_object(experiment)
        self.save_object(generation_strategy)
        return None
        
    def load_full_experiment(self):
        experiment = self.load_object('Experiment')
        generation_strategy = self.load_object('GenerationStrategy')
        
        ## loading actualy stores generation_strategy.experiment and experiment 
        ## into different memory parts. we need to connect them again
        generation_strategy.experiment = experiment
        return experiment, generation_strategy


def rds_sql_query(db, query):
    con = psycopg2.connect(**conn_params[db])
    df = pd.read_sql(query, con=con)
    con.close()
    return df