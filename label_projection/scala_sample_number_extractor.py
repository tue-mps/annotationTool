import pandas as pd
import os
from DBReader.DBReader import SyncReader

# Data location   
RAW_SEQUENCES_DIR = "/home/danil/data/RADIal/raw"
LABELS_CSV = "/home/danil/data/RADIal/Ready_to_use/RADIal/labels_clean.csv"
OUTPUT_SCALA_SAMPLE_NUMBERS_FILE = "/home/danil/data/RADIal/scala_sample_numbers/data.txt"

labels = pd.read_csv(LABELS_CSV, usecols=['numSample', 'dataset', 'index'])
datasets = labels['dataset'].unique()
print("Found", len(datasets), "datasets")

# Create resulting relation sample ID <> scala sample ID
result = pd.DataFrame(columns=['sample_id', 'scala_sample_id'])

for dataset in datasets:
    db: SyncReader
    try:
        db = SyncReader(os.path.join(RAW_SEQUENCES_DIR, dataset), tolerance=20000)
    except FileNotFoundError:
        print("Dataset", dataset, "is not found")
        continue
    data = labels[labels["dataset"] == dataset][['numSample', 'index']]
    for index, row in data.iterrows():
        scala_sample_number = db.GetSensorData(row['index'])['scala']['sample_number']
        result.loc[len(result)] = [row['numSample'], scala_sample_number]
     
result.to_csv(OUTPUT_SCALA_SAMPLE_NUMBERS_FILE, index=False, header=False)


  
    





    
