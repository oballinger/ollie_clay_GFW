import time
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse
from tqdm import tqdm
from joblib import Parallel, delayed
import ollie_clay_utils as utils
from ollie_clay_utils import Thumbnail, apply_model, apply_model_batched
import pandas as pd
from google.cloud import bigquery
import torch
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import argparse

def download_thumbnails(date):
    if not os.path.exists("data/tmp"):
        os.makedirs("data/tmp")

    if not os.path.exists("data/tmp/"+date):
        os.makedirs("data/tmp/"+date)

    # Download thumbnails
    cmd = "gsutil -q -m cp -r gs://gfw-sentinel2-thumbnails-us-central1/sentinel2_world_v20230811/{}/*.png data/tmp/{}".format(date, date)
    print(cmd)
    os.system(cmd)

    print(f"{date} Thumbnails downloaded successfully")

def upload_to_bq(df, dataset_id, table_base_name, date_suffix):
    """
    Upload a dataframe to a BigQuery date-sharded table.

    Parameters:
        df (pandas.DataFrame): DataFrame to upload.
        dataset_id (str): BigQuery dataset ID.
        table_base_name (str): Base name of the table (without the date suffix).
        date_suffix (str): Date suffix in 'YYYYMMDD' format.
    """
    client = bigquery.Client(project='world-fishing-827')
    dataset_ref = client.dataset(dataset_id)
    
    # Construct the full table ID with the date suffix
    table_id = f"{table_base_name}{date_suffix}"
    table_ref = dataset_ref.table(table_id)
    
    job_config = bigquery.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    job_config.schema = [
        bigquery.SchemaField("detect_id", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
    ]

    # Upload the DataFrame to the specified table
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete

    print(f"Loaded {len(df)} rows into {dataset_id}.{table_id}")


def check_bq_table_exists(dataset_id, table_id):
    """
    Check if a BigQuery table exists.

    Parameters:
        dataset_id (str): BigQuery dataset ID.
        table_id (str): BigQuery table ID.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    client = bigquery.Client(project='world-fishing-827')
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    try:
        client.get_table(table_ref)
        return True
    except:
        return False
    
def check_processed_dates(dataset_id):
    client = bigquery.Client(project='world-fishing-827')
    dataset_ref = client.dataset(dataset_id)
    tables = client.list_tables(dataset_ref)
    dates = [table.table_id.split('_')[-1] for table in tables if table.table_id.startswith('sentinel2_world_v20230811_embeddings_')]
    return dates    

def get_embeddings_from_png(file, model, device=torch.device("cpu")):
    try:
        thumbnail = Thumbnail.load_s2_png(file)
        detect_id = thumbnail.detect_id
        embedding = apply_model(model, device, thumbnail)
        row = pd.DataFrame({'detect_id': detect_id, 'embedding': [embedding.flatten()]})
        return row
    except Exception as e:
        print(f"Error processing detection {file}: {e}")
        return None

def batch_get_embeddings_from_pngs_gpu(batch, model, device=torch.device("cuda")):
    """
    Get embeddings for a batch of PNG files using a model on GPU.

    Parameters:
    batch (list of str): List of file paths to PNG files.
    model (torch.nn.Module): The model to generate embeddings.
    device (torch.device): The device for computation (default: CUDA).

    Returns:
    pd.DataFrame: A DataFrame containing detect_ids and their corresponding embeddings.
    """


    thumbnails, detect_ids=[],[]
    for file in batch:
        thumbnail=Thumbnail.load_s2_png(file)
        detect_id=thumbnail.detect_id
        thumbnails.append(thumbnail)
        detect_ids.append(detect_id)
        
    # Apply the model to the batch of thumbnails
    
    embeddings = apply_model_batched(model, device, thumbnails)
    rows = []
    for i in range(len(embeddings)):
        rows.append(pd.DataFrame({'detect_id': detect_ids[i], 'embedding': [embeddings[i].flatten()]}))
    df=pd.concat(rows, ignore_index=True)
    return df



def run_s2_pipeline(date, batch_size=1000):
    files = glob(f'data/tmp/{date}/*RGB*.png')
    df=pd.DataFrame()
    device=torch.device("cuda")
    model, _ = utils.load_model(device)
    model.batch_first=True
    indices = list(range(len(files)))
    batches=[files[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    rows=[]
    for batch in tqdm(batches, desc=f'Extracting Embeddings for {date}'):
        rows.append(batch_get_embeddings_from_pngs_gpu(batch, model, device))
    #rows = Parallel(n_jobs=n_jobs)(delayed(batch_get_embeddings_from_pngs)(batch, model, device) for batch in batches)
    df = pd.concat(rows, ignore_index=True)
    
    upload_to_bq(df, 'scratch_ollie', 'sentinel2_world_v20230811_embeddings_', date)
    cmd = f"rm -r data/tmp/{date}"
    os.system(cmd)
    print(f"Finished processing {date}")

def run_year(year):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    delta = end - start
    dates = [start + timedelta(days=i) for i in range(delta.days + 1)]
    dates = [d.strftime('%Y%m%d') for d in dates]
    for date in tqdm(dates, desc=f"Processing {year}"): 
        skip=check_bq_table_exists('scratch_ollie', f'sentinel2_world_v20230811_embeddings_{date}')
        if skip:
            print(f"{date} already processed")
        else:
            try:
                #download_thumbnails(date)
                run_s2_pipeline(date, batch_size=100)
            except Exception as e:
                print(f"Error processing {date}: {e}")


def run_year_threaded(year):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    delta = end - start
    dates = [start + timedelta(days=i) for i in range(delta.days + 1)]
    dates = [d.strftime('%Y%m%d') for d in dates]
    processed=check_processed_dates('scratch_ollie')
    dates=[date for date in dates if date not in processed]
    semaphore = threading.Semaphore(2)

    def process_date(date):
        try:
            run_s2_pipeline(date, batch_size=1000)
        except Exception as e:
            print(f"Error processing {date}: {e}")

    def download_next_date(next_date):
        try:
            while len(os.listdir('data/tmp')) >= 2:
                time.sleep(5)  # Wait for 5 seconds before checking again
            download_thumbnails(next_date)
        except Exception as e:
            print(f"Error downloading thumbnails for {next_date}: {e}")

    # Check if data/tmp is empty
    if not os.listdir('data/tmp'):
        # If empty, start downloading the first date
        first_date = dates[0]
        download_thumbnails(first_date)
        process_date(first_date)
        dates = dates[1:]  # Remove the first date from the list

    for i in range(len(dates)):
        date = dates[i]
        next_date = dates[i + 1] if i + 1 < len(dates) else None
        if next_date:
            threading.Thread(target=download_next_date, args=(next_date,)).start()
        process_date(date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Sentinel-2 data for a given year.")
    parser.add_argument("--year", type=int, required=True, help="Year to process (e.g., 2024)")
    args = parser.parse_args()
    run_year_threaded(args.year)
    #run_year(args.year)