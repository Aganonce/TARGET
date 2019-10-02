#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'James Flamino'
__copyright__ = 'Copyright 2019, SCNARC @ RPI'
__credits__ = ['James Flamino']
__license__ = 'GPL'
__version__ = '1.0.0'
__maintainer__ = 'James Flamino'
__email__ = 'flamij@rpi.edu'
__status__ = 'Production'

import os
import time
import pickle
import numpy as np
import pandas as pd
from dask import delayed
import tools.utils as utils
from datetime import datetime
from tools.generate_resf import generate_resf as _generate_resf
from tools.detect_outliers import outlier_update as _outlier_update
from tools.detect_outliers import detect_outliers as _detect_outliers

class TARGET:
    '''
    Text-Agnostic Response-Generated Event Tracking.

    Parameters
    ----------
    
    stream : bool (default: False)
        Initialize the stream mode.
    
    verbose : bool (default: False)
        Verbosity mode.
    
    workers : int (default: 0)
        Number of Dask workers to use for parallel computation of response features. Value must be greater
        than 1 to enable MapReduce functionality.

    nodes_per_thread : int (default: 0)
        Number of root_ids that will be assigned per worker during parallel computation of the response features.
        Value must be greater than 1 to enable MapReduce functionality.

    outlier_method : str (defaut: IsolationForest):
        The algorithm used to detect outliers from response features. There are two currently available options:
        IsolationForest and HDBSCAN. Both require only a threshold, but thresholding values will change
        depending on the outlier method. When stream=True the selected method will be responsible for
        the active detection of outliers. When stream=False the selected method will be responsible for
        generating the outlier_scores_ attribute.

    thresholds : dict (default: empty dict)
        Pass in only when stream=True. Outlier threshold dictionary where key is the platform
        and value is some float that represents the outlier threshold determined for that platform.
    
    watch_list : list (default: empty list)
        Pass in only when stream=True. List that contains root_ids that will be targeted by outlier
        detection during stream_update() calls.

    *args : None
        Placeholder. Use keyword arguments instead.

    **kwargs : dict, optional
        Extra arguments to fine-tune the algorithm used to detect outliers (type indicated by outlier_method).
        Some possible arguments for IsolationForest:
        max_samples : int (default: 'auto') The number of samples to draw from X to train each base estimator.
        contamination : int (default: 'auto') The amount of contamination of the data set, i.e. the proportion of 
        outliers in the data set.
        random_state : int (default: 1234) The random seed.
        Some possible arguments for HDBSCAN:
        min_cluster_size : int (default: 2) The minimum number of allowed clusters during cluster detection.
    
    Attributes
    ----------
    
    resf_ : dict, {platform : [n_root_ids, 6]}
        Dictionary where the key is the platform and the value is a response feature matrix.

    nodes_maps : dict, {platform : [n_root_ids,]}
        Mapping of each root_id to their associated data in response feature matrix arranged by platform.

    outlier_scores : dict {platform : [n_root_ids,]}
        Mapping of each root_id index to an associated outlier score as determined by the defined outlier detection
        algorithm, arranged by platform.

    Notes
    -----

    Attributes are only accessible when stream=False.

    Function stream_initialize() is required prior to stream_update() to establish context for the
    outlier detection algorithm.

    resF stands for response features
    '''
    def __init__(self, stream=False, verbose=False, workers=0, nodes_per_thread=0, outlier_method='IsolationForest', thresholds={}, watch_list=[], *args, **kwargs):
        self.stream = stream
        self.verbose = verbose
        
        self.workers = workers
        self.nodes_per_thread = nodes_per_thread
        if workers > 1 and nodes_per_thread > 1:
            utils.open_dask(n_workers = workers)
        
        self.outlier_method = outlier_method
        self.outlier_kwargs = kwargs
        if stream:
            self.training_resf = {}
            self.training_node_maps = {}
            
            self.test_df = {}
            self.test_node_maps = {}

            self.thresholds = thresholds
            self.watch_list = watch_list
            self.watch_list_triggered = np.zeros(len(watch_list))

            self.outliers = {}

            self.log = []


    def load_pkl(self, ofile, evolving=False):
        '''
        Load saved response features.

        Parameters
        ----------
        
        ofile : str 
           String path to where pickled response features are stored.
        
        evolving : bool (default: False)
            Indicator for if the pickled response features being passed in were evolving versus static as generated
            by train_csv().
        
        '''
        if self.stream:
            print('Cannot use load_pkl() when stream=True')
            return None
        else:
            return _load_resf(ofile, evolving=evolving)

    def train_csv(self, infile, save=False, evolving=False, time_steps=60, time_grain=3600):
        '''
        Compute response features 

        Parameters
        ----------
        
        infile : str or dataframe
           Either string path to data, or Pandas dataframe containing loaded data. If the
           filename for this data exists within the /cache/ folder in the working directory,
           TARGET will load in data from there instead. If infile is a dataframe, it will check
           /cache/ for cache.pkl
        
        save : bool (default: False)
            Pickle results to /cache/ in the working directory. Saved results will be named
            same as infile filename. If filename is a dataframe, results are saved simply as 
            /chache/cache.pkl
        
        evolving : bool (default: False)
            Evaluate aggregate response features as a function of time. Generate response features
            for all responses equal to or less than some defined time step for some number of time
            steps

        time_steps : int (default : 60)
            The number of time steps to evaluate time-evolving response features.

        time_grain : int (default : 3600)
            The size of each time step (in seconds)
        
        '''
        if self.stream:
            print('Cannot use train_csv() when stream=True')
            return None
        else:
            if _check_path(infile):
                if self.verbose:
                    print('Cached resF found at ' + _format_infile(infile) + ', loadin place')
                
                self.resf_, self.node_maps_ = self.load_pkl(_format_infile(infile), evolving=evolving)
                outliers, self.outlier_scores_ = _detect_outliers(self.resf_, outlier_method=self.outlier_method, **self.outlier_kwargs)

                return self
            else:
                resf = _assemble_response_features(infile, self.verbose, self.workers, self.nodes_per_thread, evolving=evolving, time_steps=time_steps, time_grain=time_grain)

                if save:
                    if isinstance(infile, pd.DataFrame):
                        _save_resf('cache', resf, self.verbose)
                    else:
                        _save_resf(infile, resf, self.verbose)

                self.resf_, self.node_maps_ = _compress_resf(resf, evolving)
                outliers, self.outlier_scores_ = _detect_outliers(self.resf_, outlier_method=self.outlier_method, **self.outlier_kwargs)

                if self.workers > 1 and self.nodes_per_thread > 1:
                    utils.close_dask()

                return self

    def stream_initialize(self, infile):
        '''
        Initialize training data for streaming. Given data, training resFs are generated and used as context
        for nodes added during stream_update() when detecting outliers using the outlier detection algorithm.

        Parameters
        ----------
        
        infile : str or dataframe
           Either string path to data, or Pandas dataframe containing loaded data. If the
           filename for this data exists within the /cache/ folder in the working directory,
           TARGET will load in data from there instead. If infile is a dataframe, it will check
           /cache/ for cache.pkl. Users can also point a path directly to a pkl generated by
           TARGET.
        
        '''
        if self.stream:
            if isinstance(infile, pd.DataFrame):
                resf = _assemble_response_features(infile, self.verbose, self.workers, self.nodes_per_thread, evolving=False, time_steps=60, time_grain=3600)
                _save_resf('cache', resf, self.verbose)
                self.training_resf, self.training_node_maps = _compress_resf(resf, False)
            else:         
                if '.csv' in infile:
                    resf = _assemble_response_features(infile, self.verbose, self.workers, self.nodes_per_thread, evolving=False, time_steps=60, time_grain=3600)
                    _save_resf(infile, resf, self.verbose)
                    self.training_resf, self.training_node_maps = _compress_resf(resf, False)
                elif '.pkl' in infile:
                    if self.verbose:
                        print('Loading resF from pickle')
                    self.training_resf, self.training_node_maps = _load_resf(infile, evolving=False)

            if self.workers > 1 and self.nodes_per_thread > 1:
                utils.close_dask()
        else:
            print('Cannot use initialize_base() when stream=False')

    def stream_update(self, id, user_id, created_at, parent_id, root_id, community):
        '''
        Add node into the stream.

        Parameters
        ----------
        
        id : str
            The node ID of the node being generated.

        user_id : str
            The user ID for the user that generated the node.

        created_at : int
            The unix timestamp marking the creation of the node.

        parent_id : str
            The node ID for the parent element that the current node is responding to. If the
            current node is an initial node, root_id and parent_id will be the same.

        root_id : str
            The node ID for the root element that contains all parent elements and child elements.
            The root ID represents the post. If user_id, parent_id, and root_id are all the same
            then the current node is the post itself.
        
        Notes
        -----

        All nodes added are tracked in the log; however, only elements with a root_node tracked in
        the watch_list are run through outlier detection per update.
        
        '''
        row = {'id': id, 'user_id': user_id, 'created_at': created_at, 'parent_id': parent_id, 'root_id': root_id, 'community': community}
        if self.stream:
            if id == root_id:
                if community not in self.test_df:
                    self.test_df[community] = pd.DataFrame(columns=['id_h', 'user_id', 'created_at', 'parent_id', 'root_id', 'community'])
                    self.test_df[community].loc[0] = list(row.values())
                    self.test_node_maps[community] = [id]
                else:
                    if id not in self.test_node_maps[community]:
                        pos = len(self.test_df[community].index) + 1
                        self.test_df[community].loc[pos] = list(row.values())
                        self.test_node_maps[community].append(id)
            else:
                if community not in self.test_df:
                    pass
                else:
                    pos = len(self.test_df[community].index) + 1
                    self.test_df[community].loc[pos] = list(row.values())
            if id != root_id and root_id in self.watch_list and community in self.thresholds:
                if len(self.test_df[community][self.test_df[community]['root_id'] == root_id]) > 100:
                    resf = _get_response_features(self.test_df[community], root_id, False, 60, 3600)
                    resf = np.array(list(resf.values()))
                    if root_id not in self.training_node_maps[community]:
                        self.training_resf[community] = np.append(self.training_resf[community], np.array([resf]), axis=0)
                        self.training_node_maps[community].append(root_id)
                    else:
                        pos = self.training_node_maps[community].index(root_id)
                        self.training_resf[community][pos,:] = resf

                    self.outliers[community] = _outlier_update(self.training_resf[community], threshold=self.thresholds[community], outlier_method=self.outlier_method, **self.outlier_kwargs)
                    root_index = self.training_node_maps[community].index(root_id)
                    if root_index in self.outliers[community]:
                        row['outlier'] = True
                        if self.verbose and not self.watch_list_triggered[self.watch_list.index(root_id)]:
                            print('ALERT: ' + root_id + ' becoming outlier at ' + datetime.utcfromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S'))
                            self.watch_list_triggered[self.watch_list.index(root_id)] = 1
            if 'outlier' not in row:
                row['outlier'] = False
            self.log.append(row)
        else:
            print('Cannot use update() when stream=False')

    def get_log(self):
        '''

        Return a log of all events added with stream_update(), indicating at what point in time the root_ids in the watch
        list become outliers (if any).


        Output
        ------

        log : list of dict
            The log is a list of dictionaries containing the primary format, including an 'outlier' key where the value 
            is a boolean indicating whether or not the current root_id has been identified as an outlier. 

        '''
        if self.stream:
            return self.log
        else:
            print('Cannot use get_log() when stream=False')

# Reformat infile path to cache path
def _format_infile(infile):
    infile_list = infile.split('/')
    return './cache/' + infile_list[-1].replace('.csv', '') + '.pkl'

# Check if infile path has cached pkl
def _check_path(infile):
    if isinstance(infile, pd.DataFrame):
        return os.path.isfile(_format_infile('cache'))
    else:
        return os.path.isfile(_format_infile(infile)) 

# Save resF to cache path
def _save_resf(infile, resf, verbose):
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
    ofile = _format_infile(infile)
    with open(ofile, 'wb') as d:
        pickle.dump(resf, d, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print('resF saved to ' + ofile)

# Reformat general resF output to dict of resF numpy matrices where rows are mapped to unique root_ids by a dict of node_maps
def _compress_resf(resf, evolving):
    communities = list(resf.keys())
    container = {}
    node_map = {}
    if evolving:
        for community in communities:
            community_resf = resf[community]
            node_map[community] = []
            time_range = list(community_resf[list(community_resf.keys())[0]].keys())

            A = {}
            row, col = len(community_resf), len(community_resf[list(community_resf.keys())[0]][0])
            for step in time_range:
                X = np.zeros((row, col))
                count = 0
                for key, val in community_resf.items():
                    if key not in node_map[community]:
                        X[count, :] = np.array(list(val[step].values()))
                        node_map[community].append(key)
                    else:
                        pos = node_map[community].index(key)
                        X[pos, :] = np.array(list(val[step].values()))
                    count += 1
                A[step] = X
            container[community] = A
    else:
        for community in communities:
            community_resf = resf[community]
            node_map[community] = []
            
            row, col = len(community_resf), len(community_resf[list(community_resf.keys())[0]])
            X = np.zeros((row, col))
            count = 0
            for key, val in community_resf.items():
                node_map[community].append(key)
                X[count,:] = np.array(list(val.values()))
                count += 1
            container[community] = X
    return container, node_map

# Load resF pkl from cache path
def _load_resf(ofile, evolving=False):
    resf = pickle.load(open(ofile, 'rb'))
    return _compress_resf(resf, evolving)

# Load cascade data and generate response features (resF) for each unique root_id
def _assemble_response_features(infile, verbose, workers, nodes_per_thread, evolving=False, time_steps=60, time_grain=3600):
    if verbose:
        print('Generating resF')
    
    resf = {}

    if isinstance(infile, pd.DataFrame):
        df = infile
    else:
        df = pd.read_csv(infile)
    
    df['created_at'] = df['created_at'].apply(lambda x: time.mktime(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timetuple()))
    df = df.sort_values(by=['created_at'])

    overall_posts = df[df['root_id'] == df['id_h']]
    
    communities = df.drop_duplicates('community')
    
    for community in communities['community']:
        resf[community] = {}
        posts = overall_posts[overall_posts['community'] == community]

        resf[community] = _task_response_features(df, posts['root_id'].values, verbose, evolving, time_steps, time_grain, workers, nodes_per_thread)
    return resf

# Assign the process of generating resF to either a proceedural process (basic for loop) or a parallel process (Dask MapReduce)
def _task_response_features(df, posts, verbose, evolving, time_steps, time_grain, workers, nodes_per_thread):
    resf = {}
    if workers < 2 or nodes_per_thread < 2:
        for post in posts:
            if verbose:
                print('Procedural scanning root_id: ' + post)
            resf[post] = _get_response_features(df, post, evolving, time_steps, time_grain)
    else:
        nodes = utils.partition_nodes(len(posts), nodes_per_thread)
        scatter_data_kwargs = {'root_ids': posts, 'df': df}
        map_function_kwargs = {'verbose': verbose, 'evolving': evolving, 'time_steps': time_steps, 'time_grain': time_grain}
        resf = utils.map_reduce(nodes, 
                                delayed(_worker_get_response_features), 
                                delayed(_reduce_dict),
                                scatter_data_kwargs=scatter_data_kwargs, 
                                map_function_kwargs=map_function_kwargs).compute()
    return resf

# Dask worker function for generating resF for a subset of root_ids 
def _worker_get_response_features(nodes, root_ids, df, verbose, evolving, time_steps, time_grain, *args, **kwargs):
    root_ids = root_ids[nodes]
    resf = {}
    for root_id in root_ids:
        if verbose:
            print('Parallel scanning root_id: ' + root_id)
        resf[root_id] = _get_response_features(df, root_id, evolving, time_steps, time_grain)
    return resf

# Dask worker function for consolidating resF across workers
def _reduce_dict(reduce_dict, *args, **kwargs):
    reduced_dict = reduce_dict[0]
    for row in reduce_dict[1:]:
        reduced_dict.update(row)
    return reduced_dict

# Calculate resF for a target root_id
def _get_response_features(df, root_id, evolving, time_steps, time_grain):
    data = df[df['root_id'] == root_id]
    res = {}
    if evolving:
        current_time = data['created_at'].values[0]
        for step in range(1, time_steps + 1):
            res[step - 1] = _generate_resf(data[data['created_at'] <= current_time + step * time_grain], root_id)
    else:
        res = _generate_resf(data, root_id)
    return res