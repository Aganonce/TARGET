#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from tools.generate_entropy import generate_entropy

# Find username associated with user_id
def reverse_lookup(lookup, val):
    for k, v in lookup.items():
        if v == val:
            return k
    return -1

def get_hour(date1, date2):
    return (date2 - date1) / 3600 # (date2 - date1) // 3600

# The time delta between when a branch starts (first response) and ends (last response)
def calc_longevity(data, source, target, lookup):
    target = reverse_lookup(lookup, target)
    target_data = data[data["id_h"] == target]['created_at'].values[0]
    source_data = data[data["id_h"] == source]['created_at'].values[0]
    return get_hour(source_data, target_data)

# The number of unique users a branch has (engages)
def calc_engagement(data, G, lookup):
    nodes = list(G)
    user_set = []
    for node in nodes:
        id_h = reverse_lookup(lookup, node)
        user_id = data[data["id_h"] == id_h]["user_id"].values[0]
        user_set.append(user_id)
    user_set = set(user_set)
    return len(user_set)

# Find the node farthest from the source node (max length of branch in comment tree)
def get_farthest_target(G, source):
    all_paths = nx.shortest_path(G, source=source)
    max_target = list(all_paths.keys())[-1]
    max_path_len = len(all_paths[max_target])
    return max_target, max_path_len

def avg(val):
    return sum(val) * 1.000 / len(val)

def generate_resf(data, root_id, get_depth = True, get_breadth = True, get_longevity = True, get_engagement = True):
    id_data = data.drop_duplicates("id_h")
    id_data = id_data[id_data["id_h"] != root_id]
    lookup = {}
    lookup[root_id] = 0
    count = 1
    for item in id_data["id_h"]:
        lookup[item] = count
        count += 1
    del id_data

    depth_data = []
    breadth_data = []
    longevity_data = []
    engagement_data = []

    depth = []
    for index, item in data.iterrows():
        parent = lookup[item["parent_id"]]
        if parent != 0:
            user = lookup[item["id_h"]]
            depth.append((parent, user))

    G = nx.Graph((x, y, {'weight': 1}) for (x, y), v in Counter(depth).items())
    del depth
    
    initial_nodes = list(set([lookup[x] for x in data[data["root_id"] == data["parent_id"]]["id_h"].values]))
    subgraph_list = list(G.subgraph(c) for c in nx.connected_components(G))
    for subG in subgraph_list:
        source = -1

        subG_nodes = list(subG)
        for node in initial_nodes:
            if node in subG_nodes:
                source = node
                initial_nodes.remove(node)
                break
        del subG_nodes

        source_id_h = reverse_lookup(lookup, source)
        target, max_depth = get_farthest_target(subG, source)
        if get_depth:
            depth_data.append(max_depth)
        if get_breadth:
            breadth = subG.size()
            breadth_data.append(breadth)
        if get_longevity:
            longevity_data.append(calc_longevity(data, source_id_h, target, lookup))
        if get_engagement:
            engagement_data.append(calc_engagement(data, subG, lookup))

    del G

    depth = 0
    if len(depth_data) > 0:
        depth = max(depth_data)
    breadth = 0
    if len(breadth_data) > 0:
        breadth = avg(breadth_data)
    longevity = 0
    if len(longevity_data) > 0:
        longevity = avg(longevity_data)
    engagement = 0
    if len(engagement_data) > 0:
        engagement = avg(engagement_data)

    e1, e2 = generate_entropy(data, root_id)
    return {"depth": depth, "breadth": breadth, "longevity": longevity, "engagement": engagement, "entropy_1": e1, "entropy_2": e2}
