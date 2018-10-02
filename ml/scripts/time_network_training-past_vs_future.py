import os
import sys

import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from glmnet import LogitNet
from itertools import product

sys.path.append('../../hetnet-ml/src')
import graph_tools as gt
from extractor import MatrixFormattedGraph
from processing import DegreeTransform, DWPCTransform

## Set arguments to run the script
parser = argparse.ArgumentParser(description='Run Machine Learning on Time-Based SemmedDB Network')
parser.add_argument('network_year', help="The year of the network for the analysis to be run on", type=int)
parser.add_argument('-p', '--min_pmids', help="Number of pmids needed for an edge to be considered valid", type=int, default=1)
parser.add_argument('-g', '--gs_treat', help='Use the original TREATS edge rather than that from the gold standard', action='store_true')
parser.add_argument('-a', '--alpha', help="Set the alpha value for the ElasticNet Regression", type=float, default=0.1)
parser.add_argument('-w', '--weight', help="Set the weight factor for DWPC calculations", type=float, default=0.4)
parser.add_argument('-m', '--multiplier', help="Multiplier of number positives for number of negative examples, to be selected for training", type=int, default=10)
parser.add_argument('-d', '--difference', help='Maximum year difference between network and indication approvals for training positives', type=int, default=0)
parser.add_argument('-s', '--scoring', help='Scoring metric to use for ElasticNet regression', type=str, default='roc_auc')
parser.add_argument('-e', '--seed', help='Seed for random state values', type=int, default=0)
args = parser.parse_args()

## Define variables that will govern the network analysis
base_dir = '../data/time_networks-6_metanode/'
network_year = str(args.network_year)
max_indication_network_diff = args.difference
alpha = args.alpha
scoring = args.scoring
negative_multiplier = args.multiplier
min_pmids = args.min_pmids
gs_treat = args.gs_treat
w = args.weight
seed = args.seed

test_params = os.path.join('alpha_{}'.format(alpha), '{}x_pos-neg'.format(negative_multiplier), '{}_year_diff'.format(max_indication_network_diff))

if scoring != 'roc_auc':
    test_params = os.path.join(test_params, '{}-scoring'.format(scoring))

if min_pmids > 1:
    test_params = os.path.join(test_params, '{}_pmids'.format(min_pmids))

if gs_treat:
    test_params = os.path.join(test_params, 'gs_treats')

if w != 0.4:
    test_params = os.path.join(test_params, 'dwpc_w_{}'.format(w))

if seed != 0:
    test_params = os.path.join(test_params, 'seed_{}'.format(seed))

load_dir = os.path.join(base_dir, network_year)
out_dir = os.path.join(load_dir, test_params)
n_jobs = 32
get_all_probas = True
if scoring.lower() == 'none':
    scoring = None

# Make sure the save directory exists, if not, make it
try:
    os.stat(out_dir)
except:
    os.makedirs(out_dir)


nodes = gt.remove_colons(pd.read_csv(os.path.join(load_dir, 'nodes.csv')))
edges = gt.remove_colons(pd.read_csv(os.path.join(load_dir, 'edges.csv')))

# Filter PMIDs if Applicable
if min_pmids > 1:
    print('Filtering edges to at least {} PMIDs per edge'.format(min_pmids))

    edges = edges.query('n_pmids >= @min_pmids').reset_index(drop=True)
    node_ids = list(set(edges['start_id'].unique()).union(set(edges['end_id'].unique())))
    nodes = nodes.query('id in @node_ids').reset_index(drop=True)

comp_ids = set(nodes.query('label == "Chemicals & Drugs"')['id'])
dis_ids = set(nodes.query('label == "Disorders"')['id'])

# Get Indications ready
indications = pd.read_csv(os.path.join(load_dir, 'indications.csv'))
gs_edges = (indications.rename(columns={'compound_semmed_id':'start_id', 'disease_semmed_id': 'end_id'})
              [['start_id', 'end_id', 'approval_year', 'year_diff', 'year_cat']])

# Just look at compounds and diseases in the gold standard
compounds = gs_edges['start_id'].unique().tolist()
diseases = gs_edges['end_id'].unique().tolist()

# Ensure all the compounds and diseases actually are of the correct node type and in the network
node_kind = nodes.set_index('id')['label'].to_dict()
compounds = [c for c in compounds if c in comp_ids]
diseases = [d for d in diseases if d in dis_ids]

if not gs_treat:
    print('Using the original TREATS edge from semmedDB')
else:
    print('Removing SemmedDB TREATS edges and repalcing with those from Gold Standard')

    def drop_edges_from_list(df, drop_list):
        idx = df.query('type in @drop_list').index
        df.drop(idx, inplace=True)

    # Filter out any compounds and diseases wrongly classified or lost due to PMID filtering
    gs_edges = gs_edges.query('start_id in @compounds and end_id in @diseases')
    # Remove the TREATs edge form edges
    drop_edges_from_list(edges, ['TREATS_CDtDO'])
    gs_edges['type'] = 'TREATS_CDtDO'

    column_order = edges.columns
    edges = pd.concat([edges, gs_edges], sort=False)[column_order].reset_index(drop=True)

# Save the Gold Standard used and network if there are any changes
if min_pmids > 1:
    gs_edges.to_csv(os.path.join(out_dir, 'gold_standard.csv'))

if min_pmids > 1 or gs_treat:
    pmids = edges['pmids'].dropna().apply(eval)
    pmids = pmids.apply(lambda p: 'http://www.ncbi.nlm.nih.gov/pubmed/' + ','.join(map(str, p)))
    edges['pmids'] = pmids
    (gt.add_colons(nodes, id_name='cui', col_types={'name': 'STRING', 'id_source': 'STRING'})
     .to_csv(os.path.join(out_dir, 'nodes_neo4j.csv'), index=False))
    (gt.add_colons(edges, col_types={'n_pmids': 'INT', 'name': 'STRING', 'pmids': 'STRING'})
     .to_csv(os.path.join(out_dir, 'edges_neo4j.csv'), index=False))

print('{:,} Nodes'.format(len(nodes)))
print('{:,} Edges'.format(len(edges)))

print('{} Compounds * {} Diseases = {} CD Pairs'.format(len(compounds), len(diseases), len(compounds)*len(diseases)))

num_pos = len(gs_edges)
num_neg = negative_multiplier*num_pos

# Make a DataFrame for all compounds and diseases
# Include relevent compund info and treatment status (ML Label)
cd_df = pd.DataFrame(list(product(compounds, diseases)), columns=['cd_id', 'do_id'])
id_to_name = nodes.set_index('id')['name'].to_dict()
cd_df['cd_name'] = cd_df['cd_id'].apply(lambda i: id_to_name[i])
cd_df['do_name'] = cd_df['do_id'].apply(lambda i: id_to_name[i])

merged = pd.merge(cd_df, gs_edges, how='left', left_on=['cd_id', 'do_id'], right_on=['start_id', 'end_id'])
merged['status'] = (~merged['start_id'].isnull()).astype(int)
cd_df = merged.loc[:, ['cd_id', 'cd_name', 'do_id', 'do_name', 'status', 'approval_year', 'year_diff', 'year_cat']]

# Train on those before or at the cutoff difference year
train_pos = cd_df.query('status == 1 and year_diff <= @max_indication_network_diff').index
test_pos = cd_df.query('status == 1 and year_diff > @max_indication_network_diff').index

neg_index = cd_df.query('status == 0').sample(n=num_neg, random_state=int(network_year)*(seed+1)).index

# Use 4/5 of the negatives for training and 1/5 for testing
cutoff = len(neg_index) * 4 // 5
train_neg = neg_index[:cutoff]
test_neg = neg_index[cutoff:]

train = train_neg.union(train_pos)
test = test_neg.union(test_pos)

# And keep labels of which were used for training vs testing
cd_df.loc[train, 'dataset'] = 'train'
cd_df.loc[test, 'dataset'] = 'test'
cd_df['dataset'] = cd_df['dataset'].fillna('unused')

print('Training Positives: {}'.format(cd_df.query('dataset == "train"')['status'].sum()))
print('Testing Positives: {}'.format(cd_df.query('dataset == "test"')['status'].sum()))

# Convert graph to Matrices for ML feature extraction
mg = MatrixFormattedGraph(nodes, edges, 'Chemicals & Drugs', 'Disorders', w=w)
# Extract prior
prior = mg.extract_prior_estimate('CDtDO', start_nodes=compounds, end_nodes=diseases)
prior = prior.rename(columns={'chemicals & drugs_id': 'cd_id', 'disorders_id':'do_id'})
# Extract degree Features
degrees = mg.extract_degrees(start_nodes=compounds, end_nodes=diseases)
degrees = degrees.rename(columns={'chemicals & drugs_id': 'cd_id', 'disorders_id':'do_id'})
degrees.columns = ['degree_'+c if '_id' not in c else c for c in degrees.columns]
# Generate blacklisted features and drop
blacklist = mg.generate_blacklist('CDtDO')
degrees.drop([b for b in blacklist if b.startswith('degree_')], axis=1, inplace=True)
# Extract Metapath Features (DWPC)
mp_blacklist = [b.split('_')[-1] for b in blacklist]
mps = [mp for mp in mg.metapaths.keys() if mp not in mp_blacklist]
dwpc = mg.extract_dwpc(metapaths=mps, start_nodes=compounds, end_nodes=diseases, n_jobs=n_jobs)
dwpc = dwpc.rename(columns={'chemicals & drugs_id': 'cd_id', 'disorders_id':'do_id'})
dwpc.columns = ['dwpc_'+c if '_id' not in c else c for c in dwpc.columns]

# Merge extracted features into 1 DataFrame
print('Merging Features...')
feature_df = pd.merge(cd_df, prior, on=['cd_id', 'do_id'], how='left')
feature_df = pd.merge(feature_df, degrees, on=['cd_id', 'do_id'], how='left')
feature_df = pd.merge(feature_df, dwpc, on=['cd_id', 'do_id'], how='left')

features = [f for f in feature_df.columns if f.startswith('degree_') or f.startswith('dwpc_')]
degree_features = [f for f in features if f.startswith('degree_')]
dwpc_features = [f for f in features if f.startswith('dwpc_')]

# Transform Features
dt = DegreeTransform()
dwpct = DWPCTransform()

X_train = feature_df.loc[train, features].copy()
y_train = feature_df.loc[train, 'status'].copy()

print('Transforming Degree Features')
X_train.loc[:, degree_features] = dt.fit_transform(X_train.loc[:, degree_features])
print('Tranforming DWPC Features')
X_train.loc[:, dwpc_features] = dwpct.fit_transform(X_train.loc[:, dwpc_features])

# Train our ML Classifer (ElasticNet Logistic Regressor)
print('Training Classifier...')
classifier = LogitNet(alpha=alpha, n_jobs=n_jobs, min_lambda_ratio=1e-8, n_lambda=150, standardize=True,
              random_state=(int(network_year)+1)*(seed+1), scoring=scoring)

classifier.fit(X_train, y_train)

# Just get the results on the test set
print('Obtainig Test Set results')
X_test = feature_df.loc[test, features].copy()
X_test.loc[:, degree_features] = dt.transform(X_test.loc[:, degree_features])
X_test.loc[:, dwpc_features] = dwpct.transform(X_test.loc[:, dwpc_features])
test_probas = classifier.predict_proba(X_test)[:, 1]
cd_df.loc[test, 'test_probas'] = test_probas

def glmnet_coefs(glmnet_obj, X, f_names):
    """Helper Function to quickly return the model coefs and correspoding fetaure names"""
    l = glmnet_obj.lambda_best_[0]

    coef = glmnet_obj.coef_[0]
    coef = np.insert(coef, 0, glmnet_obj.intercept_)

    names = np.insert(f_names, 0, 'intercept')

    z_intercept = coef[0] + sum(coef[1:] * X.mean(axis=0))
    z_coef = coef[1:] * X.values.std(axis=0)
    z_coef = np.insert(z_coef, 0, z_intercept)

    return pd.DataFrame([names, coef, z_coef]).T.rename(columns={0:'feature', 1:'coef', 2:'zcoef'})

coefs = glmnet_coefs(classifier, X_train, features)

print('Positivie Coefficients: {}\nNegative Coefficitents: {}'.format(len(coefs.query('coef > 0')), len(coefs.query('coef < 0'))))

coefs.to_csv(os.path.join(out_dir, 'model_coefficients.csv'), index=False)

if get_all_probas:
    # Get probs for all pairs
    print('Beginning extraction of all probabilities')
    print('Transforming all features...')
    feature_df.loc[:, degree_features] = dt.transform(feature_df.loc[:, degree_features])
    feature_df.loc[:, dwpc_features] = dwpct.transform(feature_df.loc[:, dwpc_features])

    print('Calculating Probabilities')
    all_probas = classifier.predict_proba(feature_df.loc[:, features])[:, 1]
    cd_df['all_probas'] = all_probas


def add_percentile_column(in_df, group_col, new_col, cdst_col='prediction'):

    grpd = in_df.groupby(group_col)
    predict_dfs = []

    for grp, df1 in grpd:
        df = df1.copy()

        total = df.shape[0]

        df.sort_values(cdst_col, inplace=True)
        order = np.array(df.reset_index(drop=True).index)

        percentile = (order+1) / total
        df[new_col] = percentile

        predict_dfs.append(df)

    return pd.concat(predict_dfs, sort=False)

if get_all_probas:
    cd_df = add_percentile_column(cd_df, group_col='cd_id', new_col='cd_percentile', cdst_col='all_probas')
    cd_df = add_percentile_column(cd_df, group_col='do_id', new_col='do_percentile', cdst_col='all_probas')

cd_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

