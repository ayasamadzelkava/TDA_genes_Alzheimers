# config.py 

# paths and variables

from pathlib import Path

project_root_dir = Path.cwd().parent

data_dir = project_root_dir / 'data'
raw_data_dir = data_dir / 'raw'

gene_expression_data_path = raw_data_dir / 'GSE44772_series_matrix.txt'