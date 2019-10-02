#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# Get user_id from dataframe
def get_user_id(data, id):
    data = data[data["id_h"] == id]["user_id"].values[0]
    return data

# Find user index in activity list
def find_user(activity, user_id):
    count = 0
    for row in activity:
        if row[0] == user_id:
            return count
        count += 1
    return -1

# Generate edge given u and v have interacted in a single direction
def find_directed_edge(edges, source, target):
    count = 0
    for row in edges:
        if row[0] == source and row[1] == target:
            return count
        count += 1
    return -1

# Generate edge given user u and v have interacted in either direction
def find_undirected_edge(edges, source, target):
    count = 0
    for row in edges:
        if (row[0] == source and row[1] == target) or (row[0] == target and row[1] == source):
            return count
        count += 1
    return -1

# Generate a directed graph where nodes are users, and edges are interactions
# Weights are calculated by recurrence of interactions
def weighted_edge_list(unfiltered_result, filtered_result):
    edges = []
    activity = []
    for index, row in filtered_result.iterrows():
        if row["parent_id"] == row["root_id"]:
            index = find_user(activity, row["user_id"])
            
            if index > -1:
                activity[index][1] += 1
            else:
                activity.append([row["user_id"], 1])
        
        else:
            try:
                source = get_user_id(unfiltered_result, row["parent_id"])
            except:
                continue
            
            target = row["user_id"]
            
            edge_index = find_directed_edge(edges, source, target)
            source_index = find_user(activity, source)
            target_index = find_user(activity, target)
            
            if edge_index > -1:
                edges[edge_index][2] += 1
            else:
                edges.append([source, target, 1])
            
            if source_index > -1:
                activity[source_index][1] += 1
            else:
                activity.append([source, 1])
            
            if target_index > -1:
                activity[target_index][1] += 1
            else:
                activity.append([target, 1])
    
    for i in range(len(activity)):
        activity[i] = tuple(activity[i])
    for i in range(len(edges)):
        edges[i] = tuple(edges[i])

    edges = np.array(edges)
    activity = np.array(activity)

    return edges, activity

# Calculate first order entropy
def entropy_1(activity):
	e = 0
	tot = 0
	for i in activity:
		tot = tot + int(i[1])
	for i in activity:
		p = int(i[1])/tot
		e = float(e+p*np.log(p))
	return(-e * 1.000/np.log(len(activity)))

# Calculate second order entropy
def entropy_2(edges, N):
	e = 0
	tot = 0
	for i in edges:
		tot = tot + int(i[2])
	for i in edges:
		e = e + (int(i[2]) * 1.000/tot)*np.log(int(i[2]) * 1.000/tot)
	return(-e * 1.000/np.log(N*(N-1)))

# Given data, assemble first and second order entropy for all unique users
def generate_entropy(data, root_id):
    filtered_result = data[data["user_id"] != "[deleted]"]
    edges, activity = weighted_edge_list(filtered_result, filtered_result)
    e1 = entropy_1(activity)
    e2 = entropy_2(edges, len(activity))
    return e1, e2