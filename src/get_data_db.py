from configparser import ConfigParser
import psycopg2
import os
from pathlib import Path
import pandas as pd

def get_data_db(query,col_names,save_file):
    """ Wrapper function to pull data from Civis, using direct database connection

        Keyword Args:  
        query: str
            Name of sql query file (ex: query.sql)

        col_names: [str]
            List of column names, in the order they are stored. 
            Can copy these in DBeaver (results pane > advanced copy > copy column names)      
        
        save_file: bool
            If True, save returned raw data to ../data/raw/ before returning dataframe. 
            If False, only returns dataframe.
    """
    ################################################
    # HANDLING CONFIG FILE
    ################################################
    print(os.getcwd())
    
    CONFIG_FILE = Path("config.cfg")
    if not CONFIG_FILE.exists(): # Check if there is a config file in the expected location.
        print("ERROR: Missing config file. Please add!")
        return
    else: # read config file for Civis API key and save value
        print("Reading config file.")
        config = ConfigParser()
        config.read('config.cfg')
        HOST = config.get('database', 'host')
        DATABASE = config.get('database', 'database')
        USER = config.get('database', 'user')
        PASSWORD = config.get('database', 'password')

    print("Reading config file completed!")

    ################################################
    # READING SQL QUERY
    ################################################
    SQL_QUERY = Path(f'../data/raw/{query}')
    if not SQL_QUERY.exists(): # Check if there is a SQL file in the expected location.
        print("ERROR: Missing SQL file. Please add!")
        return
    else:
        print("Reading SQL file.")
        f = open(SQL_QUERY, 'r')
        sql = f.read()
        f.close()
    
    ################################################
    # QUERYING DATABASE
    ################################################
    
    # Establish database connection
    conn = psycopg2.connect(
        host=HOST,
        database=DATABASE,
        user=USER,
        password=PASSWORD)
    
    # Execute sql
    cur=conn.cursor()
    cur.execute(sql)

    # Collect returned records
    records = cur.fetchall()
    cur.close()

    # Load records to pandas dataframe
    column_names=col_names
    data = pd.DataFrame(records,columns=column_names)
    print("Query completed")

    if save_file:
        file_name = query.strip(".sql")
        data.to_csv(f'../data/raw/{file_name}_raw.csv',index=False)
    
    return data