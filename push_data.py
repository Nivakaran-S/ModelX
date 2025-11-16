import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

import certifi
import pandas as pd
import numpy as np
from pymongo import MongoClient
from src.exception.exception import DementiaException
from src.logging.logger import logging

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DementiaDataExtract:
    def __init__(self):
        pass

    def csv_to_json_converter(self, file_path):
        try:
            logging.info(f"Reading CSV file: {file_path}")
            data = pd.read_csv(file_path, low_memory=False)

            data = data.replace({np.nan: None})  # Convert NaN to None for MongoDB

            records = data.to_dict(orient="records")
            logging.info(f"Converted {len(records)} records from CSV")

            return records
        except Exception as e:
            raise DementiaException(e, sys)

    def insert_data_mongodb(self, records, database, collection, batch_size=500):
        try:
            logging.info(f"Connecting to MongoDB...")

            client = MongoClient(
                MONGO_DB_URL,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=60000,
                connectTimeoutMS=60000,
                socketTimeoutMS=600000,
            )

            db = client[database]
            col = db[collection]

            total_records = len(records)
            logging.info(f"Total records to insert: {total_records}")

            for i in range(0, total_records, batch_size):
                batch = records[i:i + batch_size]
                col.insert_many(batch, ordered=False)
                logging.info(f"Inserted batch {i//batch_size + 1} ({len(batch)} records)")

            logging.info(f"Successfully inserted all {total_records} records")
            return total_records

        except Exception as e:
            raise DementiaException(e, sys)


if __name__ == "__main__":
    FILE_PATH = "./data/DementiaData.csv"
    DATABASE = "Adagard"
    COLLECTION = "DementiaData"

    dementiaobj = DementiaDataExtract()

    records = dementiaobj.csv_to_json_converter(file_path=FILE_PATH)
    inserted = dementiaobj.insert_data_mongodb(records, DATABASE, COLLECTION)

    print(f"INSERT COMPLETE: {inserted} records pushed to MongoDB")
