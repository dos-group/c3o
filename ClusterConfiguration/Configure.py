import collections

import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from RuntimePrediction.Predict import Predictor


### CONFIG ###
scaleout_range = 4, 14, 2 # Min, max, step
confidence = 0.95

Model = collections.namedtuple('Model', ['name', 'predictor', 'kwargs'])
Job = collections.namedtuple('Job', ['name', 'X', 'y'])

sort_df = pd.read_csv('data/sort.tsv', sep='\t')
grep_df = pd.read_csv('data/grep.tsv', sep='\t')
sgd_df = pd.read_csv('data/sgd.tsv', sep='\t')
kmeans_df = pd.read_csv('data/kmeans.tsv', sep='\t')
pagerank_df = pd.read_csv('data/pagerank.tsv', sep='\t')


def get_training_data(df, features, filters):
    # Get medians
    g = df.groupby(by=['instance_count','machine_type']+features)
    df = pd.DataFrame(g.median().to_records())
    # Apply filters
    # e.g. only for one machine type each, the full c3o-experiments were conducted
    # No full cartesian product!
    for k, s, v in filters:
        if s == '==': df = df[df[k] == v]
        if s == '>' : df = df[df[k] >  v]
    X = df[['instance_count'] + features]
    y = (df[['gross_runtime']]).squeeze()
    return X, y


def get_machine_type(job_name):
    mtypes = {'Sort': 'c4.2xlarge',
              'Grep': 'm4.2xlarge',
              'SGDLR': 'r4.2xlarge',
              'K-Means': 'r4.2xlarge',
              'Page Rank': 'r4.2xlarge'}
    return mtypes[job_name]


def get_jobs():
    td = get_training_data
    jobs = [
        Job('Sort',
            *(td(sort_df,
                 ['data_size_MB'],
                 [('machine_type', '==', 'c4.2xlarge'),
                  ('line_length', '==', 100)] )) ),
        Job('Grep',
            *(td(grep_df,
                 ['data_size_MB', 'p_occurrence'],
                 [('machine_type', '==', 'm4.2xlarge')] )) ),
        Job('SGDLR',
            *(td(sgd_df,
                 ['observations', 'features', 'iterations'],
                 [('machine_type', '==', 'r4.2xlarge'),
                  ('instance_count', '>', 2)] )) ),
        Job('K-Means',
            *(td(kmeans_df,
                 ['observations', 'features', 'k'],
                 [('machine_type', '==', 'r4.2xlarge'),
                  ('instance_count', '>', 2)] )) ),
        Job('Page Rank',
            *(td(pagerank_df,
                 ['links', 'pages', 'convergence_criterion'],
                 [('machine_type', '==', 'r4.2xlarge')] )) ),
    ]
    return jobs

def get_configuration(job_name, max_runtime, *args):

    jobs = {job.name: job for job in get_jobs()}

    ### Verify the input ###
    if not job_name in jobs:
        print(f"Job '{job_name}' has no runtime data available."); exit(1)

    print(f"Configuring cluster to execute a {job_name} job in {max_runtime}s"+\
          f" with a confidence of {confidence}")

    job = jobs[job_name]
    keys = ', '.join(k for k in job.X.keys()[1:])
    if not  len(job.X.keys()) == len(args)+1:
        print(f"Job '{job.name}' requires {len(job.X.keys())-1} context args:"+\
              f" {keys} but {'none' if len(args)==0 else len(args)} were given")
        exit(1)

    values = '\n'.join(f"    {k}: {v}" for k,v in zip(job.X.keys()[1:], args))
    print(f"Execution context for {job.name}:\n{values}")

    ### Estimate the accuracy of the runtime predictor ###

    rtpred = Predictor()
    X_tr, X_te, y_tr, y_te = train_test_split(job.X, job.y, test_size=0.1)
    rtpred.fit(X_tr, y_tr)
    y_hat = rtpred.predict(X_te)
    errors = (y_hat - y_te).to_numpy()
    mu, sigma = errors.mean(), errors.std()
    print(f"Estimated mean runtime prediction error: {mu:.2f}s, " +\
          f"standard deviation: {sigma:.2f}s")

    x = scipy.special.erfinv(2 * confidence - 1) * np.sqrt(2)
    tolerance = mu + x * sigma
    print(f"Required tolerance to reach the deadline in {confidence*100}%"+\
          f" of cases: {tolerance:.2f}s")

    ### Determine which scale-out is predicted to be sufficient ###

    rtpred.fit(job.X, job.y)
    possibilities = np.array(list((so, *args) for so in range(*scaleout_range)))
    y_hat = rtpred.predict(possibilities)

    chosen_machinetype = get_machine_type(job.name)
    chosen_scaleout, resulting_runtime = None, None
    for so, rt in zip(range(*scaleout_range), y_hat):
        if rt+tolerance <= max_runtime:
            chosen_scaleout = so; resulting_runtime = rt; break

    if not chosen_scaleout:
        print("Max scaleout insufficient to confidently reach the deadline")
    else:
        print(f"Estimated optimal cluster configuration: "+\
              f"{chosen_scaleout} x {chosen_machinetype} "+\
              f"with estimated runtime: {resulting_runtime:.2f}s\n")


