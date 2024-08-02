1import pandas as pd
import re
import csv
import numpy as np
from umap.umap_ import UMAP
import sklearn
from sklearn.manifold import Isomap, MDS
#import kmapper as km
import statsmodels.api as sm
#from rpy2.robjects import numpy2ri
from scipy.stats import zscore
import matplotlib
import matplotlib.pyplot as plt
#import community as community_louvain
#import networkx as nx
#numpy2ri.activate()

# PREPROCESSING

def extract_meta_data_GSE33000(data):
    '''
    Extracting metadata for each patient [index, patient ID, age, sex, disease]
    :param data:
    :return:
    '''
    # Define mapping for gender and disease status
    gender_mapping = {"female": 0, "male": 1}
    disease_mapping = {"non-demented": 0, "Alzheimer's disease": 1, "Huntington's disease": 2}

    # Initialize lists to hold the sample IDs, ages, genders, and disease statuses
    sample_ids = []
    ages = []
    genders = []
    disease_statuses = []

    for sublist in data:
        if sublist and sublist[0] == '!Sample_characteristics_ch2':
            attribute_type = sublist[1].split(":")[0]

            if attribute_type == 'age':
                ages = [int(re.search('age: (\d+)', item).group(1)) if 'age: ' in item else -1 for item in sublist[1:]]
            elif attribute_type == 'gender':
                genders = [gender_mapping.get(re.search('gender: (\w+)', item).group(1), -1) if 'gender: ' in item else -1 for item in sublist[1:]]
            elif attribute_type == 'disease status':
                disease_statuses = [disease_mapping.get(re.search('disease status: (.*)', item).group(1), -1) if 'disease status: ' in item else -1 for item in sublist[1:]]
        elif sublist and sublist[0] == '!Series_sample_id':
            sample_ids = sublist[1].split(' ')

    # Get the number of samples
    num_samples = len(sample_ids)

    # If any of the lists are shorter than num_samples, extend them with -1 or '' as appropriate
    ages.extend([-1] * (num_samples - len(ages)))
    genders.extend([-1] * (num_samples - len(genders)))
    disease_statuses.extend([-1] * (num_samples - len(disease_statuses)))

    # Combine the sample IDs, ages, genders, and disease statuses into the processed data
    processed_data = [[i, sample_ids[i], ages[i], genders[i], disease_statuses[i]] for i in range(num_samples)]

    return processed_data

def adjust_for_covariates_GSE33000(expression_df, metadata):
    """
    Adjust the gene expression data for age and sex using linear regression.

    Parameters:
    - expression_df: DataFrame containing gene expression data where rows are genes and columns are samples.
    - metadata: dataframe with metadata

    Returns:
    - DataFrame with adjusted gene expression data.
    """
    # Convert metadata into a DataFrame for easier manipulation
    meta_df = metadata

    # Ensure gender is coded numerically for regression
    gender_mapping = {"female": 0, "male": 1}
    meta_df["gender"] = meta_df["gender"].map(gender_mapping)

    adjusted_data = expression_df.copy()

    for sample in expression_df.columns:
        sample_meta = meta_df[meta_df["participant_id"] == sample]

        if not sample_meta.empty:
            age = sample_meta["age"].values[0]
            gender = sample_meta["gender"].values[0]

            # Use age and gender as covariates
            X = pd.DataFrame({"age": [age], "gender": [gender]})
            X = sm.add_constant(X)  # Add a constant to the model (intercept)

            y = expression_df[sample]

            # Check for NaN or inf values in X and y
            if not (X.isnull().values.any() or np.isinf(X.values).any() or y.isnull().values.any() or np.isinf(
                    y.values).any()):
                # Fit linear regression model
                model = sm.OLS(y, X).fit()

                # Get the residuals, which represent the adjusted data
                adjusted_data[sample] = model.resid

    return adjusted_data

def extract_test_groups_GSE33000(data):
    disease_mapping = {"non-demented": 0, "Alzheimer's disease": 1, "Huntington's disease": 2}
    # Initialize the result dictionary
    result = {disease: [] for disease in disease_mapping.keys()}

    # Initialize variables to keep track of the current group
    current_disease = None
    start_index = None

    # Iterate over the data
    for i, row in enumerate(data):
        # Check if the current row's disease is different from the current disease
        if row[-1] != current_disease:
            # If the current disease is not None, save the start and end indices for this disease
            if current_disease is not None:
                try:
                    end_index = i - 1
                    result[list(disease_mapping.keys())[list(disease_mapping.values()).index(current_disease)]].append((start_index, end_index))
                except ValueError:
                    print(f'Unexpected value for current_disease: {current_disease}')
                    break

            # Update the current disease and start index
            current_disease = row[-1]
            start_index = i

    # Add the last group to the result
    if current_disease in disease_mapping.values():
        result[list(disease_mapping.keys())[list(disease_mapping.values()).index(current_disease)]].append((start_index, i))

    return result

def extract_most_variable_genes(expression_matrix):
    """
    Extract the top one-third of the most variable genes based on IQR.

    Parameters:
    - expression_matrix: DataFrame with genes as rows and samples as columns.

    Returns:
    - DataFrame containing the top one-third most variable genes.
    """

    # Calculate IQR for each gene
    iqr_values = expression_matrix.apply(lambda x: x.quantile(0.75) - x.quantile(0.25), axis=1)

    # Rank genes based on IQR
    ranked_genes = iqr_values.sort_values(ascending=False)

    # Select the top one-third
    top_genes = ranked_genes.head(len(ranked_genes) // 3).index

    return expression_matrix.loc[top_genes]

#07/24/24
def extract_most_variable_genes_between_conditions(healthy_df, disease_df, top_fraction=1/3):
    """
    Extracts the most variable genes between healthy and disease conditions using IQR.
    
    Parameters:
    healthy_df (pd.DataFrame): DataFrame with healthy samples where rows are genes and columns are samples (patients).
    disease_df (pd.DataFrame): DataFrame with disease samples where rows are genes and columns are samples (patients).
    top_fraction (float): Fraction of most variable genes to return.
    
    Returns:
    pd.DataFrame: DataFrame containing the top fraction of most variable genes for healthy samples.
    pd.DataFrame: DataFrame containing the top fraction of most variable genes for disease samples.
    """
    # Calculate the IQR for each gene in both groups
    iqr_healthy = healthy_df.apply(lambda x: x.quantile(0.75) - x.quantile(0.25), axis=1)
    iqr_disease = disease_df.apply(lambda x: x.quantile(0.75) - x.quantile(0.25), axis=1)
    
    # Calculate the absolute difference in IQR between healthy and disease
    iqr_diff = (iqr_healthy - iqr_disease).abs()
    
    # Determine the number of top variable genes to select
    top_n = int(len(iqr_diff) * top_fraction)
    
    # Get the indices of the top N most variable genes
    top_genes_indices = iqr_diff.nlargest(top_n).index
    
    # Return the DataFrames containing only the top variable genes
    return healthy_df.loc[top_genes_indices], disease_df.loc[top_genes_indices]

def extract_metadata_GSE44772(data):
    '''
    Extracting specific metadata for each sample in the GSE44772 dataset.
    :param data: List of lists, each representing a line from the file.
    :return: A list of lists, each containing specific metadata for a sample.
    '''

    # Initialize lists to hold the sample IDs and metadata
    sample_ids = []
    pH = []
    Age = []
    RIN = []
    Gender = []
    Batch = []
    PMI = []
    Tissue = []
    Disease = []
    Preservation_Method = []
    Brain_Region = []

    metadata_labels = ['Index', 'Sample ID', 'pH', 'Age', 'RIN', 'Gender', 'Batch', 'PMI', 'Tissue', 'Disease', 'Preservation Method', 'Brain Region']


    for sublist in data:
        if sublist and sublist[0] == '!Sample_characteristics_ch2':
            # Extracting metadata from each entry
            for entry in sublist[1:]:
                if 'ph:' in entry:
                    pH.append(entry.split(':')[1].strip())
                elif 'age:' in entry:
                    Age.append(entry.split(':')[1].strip())
                elif 'rin:' in entry:
                    RIN.append(entry.split(':')[1].strip())
                elif 'gender:' in entry:
                    Gender.append(entry.split(':')[1].strip())
                elif 'batch:' in entry:
                    Batch.append(entry.split(':')[1].strip())
                elif 'pmi:' in entry:
                    PMI.append(entry.split(':')[1].strip())
                elif 'tissue:' in entry:
                    Tissue.append(entry.split(':')[1].strip())
                elif 'disease:' in entry:
                    Disease.append(entry.split(':')[1].strip())
                elif 'pres:' in entry:
                    Preservation_Method.append(entry.split(':')[1].strip())
        elif sublist and sublist[0] == '!Sample_description':
            # Extracting brain region from the description
            Brain_Region = sublist[1:]
            # if 'DLPFC' in description:
            #     Brain_Region.append('DLPFC')
            # elif 'visual cortex' in description:
            #     Brain_Region.append('visual cortex')
            # elif 'cerebellum' in description:
            #     Brain_Region.append('cerebellum')
            # else:
            #     Brain_Region.append('Unknown')  # or use '' if you prefer an empty string for unknown

        elif sublist and sublist[0] == '!Series_sample_id':
            sample_ids = sublist[1].split(' ')

    # Combine the sample IDs and metadata into the processed data
    processed_data = []
    for i in range(len(sample_ids)):
        sample_data = [
            i,
            sample_ids[i],
            pH[i] if i < len(pH) else '',
            Age[i] if i < len(Age) else '',
            RIN[i] if i < len(RIN) else '',
            Gender[i] if i < len(Gender) else '',
            Batch[i] if i < len(Batch) else '',
            PMI[i] if i < len(PMI) else '',
            Tissue[i] if i < len(Tissue) else '',
            Disease[i] if i < len(Disease) else '',
            Preservation_Method[i] if i < len(Preservation_Method) else '',
            Brain_Region[i] if i < len(Brain_Region) else 'Unknown'
        ]
        processed_data.append(sample_data)
    meta_df = pd.DataFrame(processed_data, columns=metadata_labels)


    return meta_df

def adjust_for_covariates_GSE44772(df_GSE44772_interpolated, meta_data_GSE44772_df):
    """
    Adjust the gene expression data for covariates using linear regression.
    Parameters:
    - df_GSE44772_interpolated: DataFrame containing gene expression data (rows: genes, columns: samples).
    - meta_data_GSE44772: List of lists containing metadata for each sample.
    Returns:
    - DataFrame with adjusted gene expression data.
    """
    meta_columns = meta_data_GSE44772_df.columns.tolist()
    meta_df = meta_data_GSE44772_df

    # Convert numeric metadata and map categorical variables
    for col in ['pH', 'Age', 'RIN', 'Batch', 'PMI']:
        meta_df[col] = pd.to_numeric(meta_df[col], errors='coerce')
    meta_df['Gender'] = meta_df['Gender'].map({'M': 1, 'F': 0})
    meta_df['Disease'] = meta_df['Disease'].map({'A': 1, 'N': 0})
    meta_df['Brain Region'] = meta_df['Brain Region'].map({'cerebellum': 1, 'DLPFC': 0, 'visual cortex':2})
    meta_df['Preservation Method'] = meta_df['Preservation Method'].map({'LNV': 1, 'Dry-ice': 0})

    adjusted_data = df_GSE44772_interpolated.copy()

    for sample in df_GSE44772_interpolated.columns:
        sample_meta = meta_df[meta_df['Sample ID'] == sample]
        if not sample_meta.empty:
            # Selecting covariates and replicating for each gene
            covariates = sample_meta[['Age', 'Gender', 'Brain Region']].iloc[0]
            covariates_df = pd.DataFrame([covariates.values] * len(df_GSE44772_interpolated), columns=covariates.index)
            X = sm.add_constant(covariates_df)  # Add a constant (intercept)

            y = df_GSE44772_interpolated[sample]

            # Check for NaN or inf in X and y
            if not (X.isnull().values.any() or np.isinf(X.values).any() or y.isnull().values.any() or np.isinf(
                    y.values).any()):
                # Fit linear regression model
                model = sm.OLS(y, X).fit()
                adjusted_data[sample] = model.resid

    return adjusted_data

def adjust_for_covariates_new(data_df, metadata_df, columns_to_exclude=None):
    
    # load data
    data = data_df.copy()
    metadata = metadata_df.copy()
    adjusted_data = pd.DataFrame(index=data.index, columns=data.columns)
    
    if columns_to_exclude is None:
        columns_to_exclude=[]
    set_to_exclude = set(['participant_id', 'disease_status', 'Sample ID', 'Index', 'Disease', 'index'] + columns_to_exclude)

    covariates_list = []
    for col in metadata.columns:
        if col in set_to_exclude:
            continue
        covariates_list.append(col)
    print('correction for: ', covariates_list)
    
    X = metadata[covariates_list]
    X = sm.add_constant(X)
    for gene_index in data.index:
        Y = data.iloc[gene_index].values
        #model = sm.RLM(Y, X, M=sm.robust.norms.HuberT()).fit()
        model = sm.OLS(Y, X).fit()
        adjusted_data.iloc[gene_index] = model.resid  
        
    return adjusted_data

def convert_meta_to_numeric(metadata):
    """
    Converts columns in the metadata DataFrame to numeric where possible. If a column contains
    strings, it converts them to categorical integers and provides mappings for these categories.
    Excludes any conversion for the 'Sample ID' column.

    Parameters:
        metadata (pd.DataFrame): DataFrame containing metadata with mixed data types.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Updated DataFrame with numeric and categorical integer conversions.
            - dict: Dictionary with mappings from strings to integers for categorical columns.
    """
    metadata = metadata.copy()  # Work on a copy to avoid modifying the original DataFrame.
    category_mappings = {}  # To store mappings for each categorical column.

    for col in metadata.columns:
        if col == 'Sample ID':
            continue  # Skip the 'Sample ID' column.

        # Try converting column to numeric, coercing errors to NaN.
        converted_column = pd.to_numeric(metadata[col], errors='coerce')

        if converted_column.isna().all():  # If conversion fails (all values are NaN), treat as categorical.
            uniques = metadata[col].dropna().unique()  # Get unique non-NA values
            codes, unique_values = pd.factorize(uniques)  # Factorize the unique values
            metadata[col] = metadata[col].map({uv: code for uv, code in zip(unique_values, codes)})  # Map all values in the column
            category_mappings[col] = dict(zip(unique_values, codes))
        else:
            # Successfully converted to numeric, fill NaNs (if any) with the column median.
            metadata[col] = converted_column.fillna(converted_column.median())

    return metadata, category_mappings


# GO TERMS

def extract_go_terms(file_path):
    """
    Extracts GO terms from the provided file and strictly removes gene IDs that don't have descriptors.
    """
    # Dictionaries to store GO annotations for each ontology
    go_bp = {}
    go_mf = {}
    go_cc = {}

    # Open and read the file
    with open(file_path, 'r') as file:
        # Skip lines until we reach the actual data table headers
        for line in file:
            if line.startswith('!platform_table_begin'):
                break

        # Now, the next line should be the headers
        headers = file.readline().strip().split('\t')

        for line in file:
            data = line.strip().split('\t')
            gene_id = data[0]

            # Create a dictionary for the current row using zip (handles uneven lengths)
            row_dict = dict(zip(headers, data))

            # Extract GO annotations based on known headers and structure
            bp_terms = row_dict.get('GO:Process', '').split('///')
            mf_terms = row_dict.get('GO:Function', '').split('///')
            cc_terms = row_dict.get('GO:Component', '').split('///')

            # Clean up terms
            bp_terms = [term for term in bp_terms if term]
            mf_terms = [term for term in mf_terms if term]
            cc_terms = [term for term in cc_terms if term]

            # Assign cleaned terms to the dictionaries
            if bp_terms:
                go_bp[gene_id] = bp_terms
            if mf_terms:
                go_mf[gene_id] = mf_terms
            if cc_terms:
                go_cc[gene_id] = cc_terms

    return go_bp, go_mf, go_cc


def category_to_geneid_dict(gene_ids, go_annotations):
    """
    Categorize a list of genes based on their ontology descriptors.

    Parameters:
    - gene_ids: List of gene IDs to categorize.
    - go_annotations: Dictionary of gene IDs to their associated GO terms.

    Returns:
    A dictionary where each key is a GO term and the value is a list of genes associated with that term.
    """
    categorized_genes = {}

    for gene_id in gene_ids:
        # Convert gene ID to string for matching
        str_gene_id = str(gene_id)
        # Check if the gene has GO annotations
        if str_gene_id in go_annotations:
            for go_term in go_annotations[str_gene_id]:
                if go_term not in categorized_genes:
                    categorized_genes[go_term] = []
                categorized_genes[go_term].append(str_gene_id)

    return categorized_genes


def genes_in_categories_df(gene_ids, all_genes_dict):
    """
    gene_ids: list of data (ex: GSE33000) gene ids
    all_genes_dict: (dict) from annotations file. keys - genes ids, values - categories that gene belongs to

    Returns: a pandas DataFrame where each row corresponds to a gene from all_genes and each column
    corresponds to a category (biological process).
    Each cell in the DataFrame indicates whether the gene (specified by the row) belongs to the category
    (specified by the column). 1 indicates the gene belongs to that category, and 0 indicates it does not.
    """

    this_category_to_geneid_dict = category_to_geneid_dict(gene_ids, all_genes_dict)

    all_genes_str = [str(gene) for gene in gene_ids]
    categories = sorted(this_category_to_geneid_dict.keys())

    # Initialize an empty dictionary to hold our data
    data = {category: [] for category in categories}

    # Fill the dictionary with 1s and 0s, indicating gene presence in categories
    for gene in all_genes_str:
        for category in categories:
            genes_in_category_str = [str(g) for g in this_category_to_geneid_dict[category]]
            data[category].append(1 if gene in genes_in_category_str else 0)

    # Create the DataFrame from the dictionary, using the gene names as the index
    df = pd.DataFrame(data, index=all_genes_str)

    return df



#   COLOR

def get_cluster_labels_and_colorscale_GSE33000(graph, data):
    # DBSCAN clusters
    cluster_labels = -np.ones(data.shape[0])
    for cluster_id, indices in enumerate(graph['nodes'].values()):
        for index in indices:
            cluster_labels[index] = int(cluster_id)

    # Prepare for visualization
    n_bins = len(np.unique(cluster_labels))
    cmap = plt.cm.gist_rainbow.reversed()
    colorscale = [[i/n_bins, matplotlib.colors.to_hex(cmap(i/n_bins))] for i in range(n_bins)]

    return cluster_labels, colorscale



# CLUSTERING

def get_graph_from_Adj_matrix(A, metadata_df, keep_nans=False):
    """
    Function takes in adjescency matrix and generates a graph using metadata
    
    Parameters:
    A(array) - 
    metadata(df) - metadata used to look up gene symbols. for a_ij in A if a =1, i-sourse, j-target 
    keep_nans(bool) - a fag to indicate if to keep dna sequences that are not associated with any genes. 
    
    Returns:
    df with columns: 'Source', 'Target', 'Weight'
    """
    F_graph_df = pd.DataFrame(columns=['Source', 'Target', 'Weight'])

    for row in range(len(A)):
        for col in range(len(A)):
            if A[row,col] == 1:
                source = metadata_df.iloc[row, 3]      #3rd col is Gene symbol
                target = metadata_df.iloc[col, 3]
                
                if keep_nans==True and (source=='NaN' or target=='Nan'):
                    if source == 'Nan':
                        source = metadata_df.iloc[row, 1]      #1st col is Gene ID
                    if target == 'NaN':
                        target = metadata_df.iloc[col, 1]
                    F_graph_df.loc[len(F_graph_df)] = [source, target, 1]
                
                else:
                    if source != 'Nan' and target != "NaN" :
                        F_graph_df.loc[len(F_graph_df)] = [source, target, 1]
                    
    return F_graph_df 




# OLD SHIT


'''
the following chunk of code is for when i wa trying to cluster genes how they did it in the alzh paper using louvain clustering. i opted out 
to do directly from tda

# Step 2: Convert the correlation matrix to an adjacency matrix using a power function
# You will need to choose 'b' such that the network approximates a scale-free topology
b = 8.5  # This is just an example value, you should determine this based on your data
adjacency_matrix = np.abs(healthy_ar_cov) ** b

# Step 3: Check for scale-free topology


# Step 4: Calculate the Topological Overlap Matrix
def calculate_tom(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    tom = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u = np.sum(adjacency_matrix[i, :] * adjacency_matrix[j, :])
            v = np.sum(adjacency_matrix[i, :] + adjacency_matrix[j, :]) - adjacency_matrix[i, j]
            tom[i, j] = u / v if v > 0 else 0
    # Make TOM symmetric
    tom = (tom + tom.T) / 2
    return tom

tom = calculate_tom(adjacency_matrix)

np.save("calculated_data/healthy/healthy_hv_tom.npy", tom)
np.savetxt("calculated_data/healthy/healthy_hv_tom.txt", tom)
tom = np.load("calculated_data/healthy/healthy_hv_tom.npy")

np.fill_diagonal(tom, 1)
# Step 5: Module detection using hierarchical clustering
# Convert the TOM into a distance matrix
dist_matrix = 1 - tom
np.fill_diagonal(dist_matrix, 0)

# dimentionality reduction
reducer = UMAP(n_components=100, random_state=1)  # Reducing to 1000 dimensions
embedding = reducer.fit_transform(dist_matrix)  # your TOM matrix

from sklearn.neighbors import kneighbors_graph
knn_graph = kneighbors_graph(embedding, n_neighbors=20, mode='distance', include_self=True)

# Perform  clustering
def convert_matrix_to_graph(matrix):
    """Convert a given matrix to a graph."""
    G = nx.Graph()
    for i in range(matrix.shape[0]):
        for j in range(i, matrix.shape[1]):  # Consider upper triangle of matrix
            if abs(matrix[i, j]) > 0.01:  # Assuming only positive weights are meaningful
                G.add_edge(i, j, weight=matrix[i, j])
    return G

G = nx.from_scipy_sparse_array(knn_graph)
partition = community_louvain.best_partition(G, weight='weight')

# visualization
from collections import Counter

cluster_counts = Counter(partition.values())
sorted_cluster_counts = dict(sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True))

clusters = list(sorted_cluster_counts.keys())
sizes = list(sorted_cluster_counts.values())
plt.figure(figsize=(12, 6))
plt.bar(clusters, sizes, color='skyblue')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Points')
plt.title('Distribution of Cluster Sizes')
plt.show()




# Step 6: Assign modules to clusters
cluster_labels = np.array([partition.get(i, -1) for i in range(len(healthy_cov))])
np.save("calculated_data/healthy/healthy_hv_clusters_UMAP_Louvian_cov.npy", cluster_labels)
labeled_data = np.column_stack((healthy_cov, cluster_labels))
'''