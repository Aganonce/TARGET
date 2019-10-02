#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: James Flamino

import os
import dask
import numpy as np
import dask.distributed

# MapReduce function
def map_reduce(partition, map_function, reduce_function, scatter_data_kwargs={}, map_function_kwargs={}, reduce_function_kwargs={}):
    if scatter_data_kwargs:
        scatter_data_kwargs = scatter_kwargs(scatter_data_kwargs)    
    if map_function_kwargs:
        map_function_kwargs = scatter_kwargs(map_function_kwargs)
    if reduce_function_kwargs:
        reduce_function_kwargs = scatter_kwargs(reduce_function_kwargs)

    partition = [map_function(indices, **scatter_data_kwargs, **map_function_kwargs) for indices in partition]

    partition_mapping = partition_nodes(len(partition), 2)
    if len(np.array(partition_mapping).ravel()) % 2:
        partition_mapping[-2:] = [np.concatenate((partition_mapping[-2], partition_mapping[-1]), axis=None)]

    partition = [reduce_function(partition[indices[0]:indices[-1]+1], **reduce_function_kwargs) for indices in partition_mapping]

    while len(partition) > 1:
        partition_mapping = partition_nodes(len(partition), 2)
        partition = [reduce_function(partition[indices[0]:indices[-1]+1], **reduce_function_kwargs) for indices in partition_mapping]
    return partition[0]

# Scatter function
def scatter_kwargs(kwargs_to_scatter):
    return {key: value for key, value in zip(list(kwargs_to_scatter.keys()), dask.distributed.get_client().scatter(list(kwargs_to_scatter.values()), broadcast=True))}

# Partition function
def partition_nodes(node_count, nodes_per_partition, shuffle=False, track_pos=False):
    nodes = np.arange(0, node_count)
    if shuffle:
        np.random.shuffle(nodes)
    node_bins = [nodes[x: x+nodes_per_partition] for x in np.arange(0, len(nodes), nodes_per_partition)]
    if track_pos:
        node_bins = list(zip(range(len(node_bins)), node_bins))
    return node_bins

# Dask client controllers
def open_dask(n_workers=8, memory_limit='8GB', local_directory=os.getcwd()):
    config = {'n_workers': n_workers, 'memory_limit': memory_limit, 'local_directory': local_directory}
    try:
        dask.distributed.get_client()
    except ValueError:
        cluster = dask.distributed.LocalCluster(**config)
        client = dask.distributed.Client(cluster)

def close_dask():
    client = dask.distributed.get_client()
    client.close()
