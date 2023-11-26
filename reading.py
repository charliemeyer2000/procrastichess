#!/Users/charlie/miniconda3/bin/python3
import csv
import torch
import numpy as np
import pandas as pd

print("hello")

PGN_FORMAT_IDX = 15

df = pd.read_csv("data/club_games_data.csv")



white_username = df['white_username']
black_username = df['black_username']
white_id = df['white_id']
black_id = df['black_id']
white_rating = df['white_rating']
black_rating = df['black_rating']
white_result = df['white_result']
black_result = df['black_result']
time_class = df['time_class']
time_control = df['time_control']
rules = df['rules']
rated = df['rated']
fen = df['fen']
pgn = df['pgn']

print(pgn[0])