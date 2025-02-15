from load_data import load_vine_data
import pandas as pd
from understanding_data import understanding_data
from data_integrity_report import integrity_report
from check_duplicates import check_duplicates
from remove_duplicates import remove_duplicates
from summary_statistics import summary_statistics
from histograms import draw_histogram
from heatmap import draw_heatmap
from boxplots import draw_boxplots
from remove_outliers import remove_outliers

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

red_wine, white_wine = load_vine_data()

understanding_data(red_wine, white_wine)
integrity_report(red_wine, white_wine)
check_duplicates(red_wine, white_wine)
red_wine_cleaned, white_wine_cleaned = remove_duplicates(red_wine, white_wine)
# Repeat integrity report after cleaning data
integrity_report(red_wine_cleaned, white_wine_cleaned, skip_data_types=True)
summary_statistics(red_wine_cleaned, white_wine_cleaned)
draw_histogram(red_wine_cleaned, white_wine_cleaned)
draw_heatmap(red_wine_cleaned, white_wine_cleaned)
draw_boxplots(red_wine_cleaned, white_wine_cleaned)
red_wine_cleaned, white_wine_cleaned = remove_outliers(red_wine_cleaned, white_wine_cleaned)
draw_boxplots(red_wine_cleaned, white_wine_cleaned) # draw boxplots after remove outliers