import subprocess
import pandas as pd

filename = '../data/time_networks-6_metanode/1985/edges.csv'

df = pd.read_csv(filename)

# Run most populous edges first...
edge_types = df[':TYPE'].value_counts().index

for et in edge_types:
    subprocess.call(['./scripts/run_e_replace.sh', et])
