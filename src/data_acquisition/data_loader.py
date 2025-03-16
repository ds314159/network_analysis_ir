import json
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import os
from typing import Dict, List, Any, Optional, Union

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_from_json(self, file_path):
        """Charge les données depuis un fichier JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def load_from_csv(self, file_path, delimiter='\t'):
        """Charge les données depuis un fichier CSV"""
        return pd.read_csv(file_path, delimiter=delimiter)