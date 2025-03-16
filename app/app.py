from flask import Flask, render_template, request, jsonify
import networkx as nx
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Imports des modules du projet
from src.data_acquisition.data_loader import DataLoader
from src.graph_analysis.graph_builder import GraphBuilder
from src.graph_analysis.graph_viz import GraphVisualizer
from src.search_engine.indexer import Indexer
from src.search_engine.query_processor import QueryProcessor
from src.search_engine.ranking import RankingSystem
from src.clustering.text_clusterer import TextClusterer
from src.clustering.label_generator import LabelGenerator
from src.classification.classifier import Classifier