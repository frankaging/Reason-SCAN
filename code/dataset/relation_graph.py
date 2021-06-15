from collections import namedtuple, OrderedDict
import itertools
import os
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
import random
from itertools import product
import copy
import re
import random

from utils import one_hot
from utils import generate_possible_object_names
from utils import numpy_array_to_image

from vocabulary import *
from object_vocabulary import *
from world import *
from grammer import *
from simulator import *

import networkx as nx
from networkx.algorithms import isomorphism
from networkx import DiGraph
from networkx import line_graph

class ReaSCANGraph(object):
    """
    SAME_ROW_REGEX = "$SAME_ROW"
    SAME_COL_REGEX = "$SAME_COLUMN"
    SAME_SHAPE_REGEX = "$SAME_SHAPE"
    SAME_COLOR_REGEX = "$SAME_COLOR"
    SAME_SIZE_REGEX = "$SAME_SIZE"
    IS_INSIDE_REGEX = "$IS_INSIDE"
    """

    def __init__(
        self, objects, object_patterns, vocabulary, 
        relations=None, positions=None, referred_object=None,
        debug=False
    ):
        self.vocabulary = vocabulary
        self.relations = OrderedDict({})
        self.G = nx.MultiDiGraph()
        self.G_full = nx.MultiDiGraph()
        if positions == None:
            # This means it is a abstract pattern, not a grounded shapeWorld graph.
            for obj_name, obj_str in objects.items():
                if referred_object != None and obj_name == referred_object:
                    self.G.add_node(obj_name, type="object", referent=True)
                    self.G_full.add_node(obj_name, type="object", referent=True)
                else:
                    self.G.add_node(obj_name, type="object", referent=False)
                    self.G_full.add_node(obj_name, type="object", referent=False)
                obj_str = obj_str.split(" ")
                if len(obj_str) == 1:
                    if obj_str[0] != "object":
                        self.relations[(obj_name, obj_str[0])] = ['$SHAPE']
                elif len(obj_str) == 2:
                    if obj_str[0] in vocabulary.get_size_adjectives():
                        # BUG: adding all descriptors to size.
                        self.relations[(obj_name, obj_str[0]+obj_str[1])] = ['$SIZE']
                    elif obj_str[0] in vocabulary.get_color_adjectives():
                        self.relations[(obj_name, obj_str[0])] = ['$COLOR']
                    if obj_str[1] != "object":
                        self.relations[(obj_name, obj_str[1])] = ['$SHAPE']
                elif len(obj_str) == 3:
                    if obj_str[0] in vocabulary.get_size_adjectives():
                        size = obj_str[0]
                    elif obj_str[0] in vocabulary.get_color_adjectives():
                        color = obj_str[0]
                    if obj_str[1] in vocabulary.get_size_adjectives():
                        size = obj_str[1]
                    elif obj_str[1] in vocabulary.get_color_adjectives():
                        color = obj_str[1]
                    # BUG: adding all descriptors to size.
                    self.relations[(obj_name,  obj_str[0]+obj_str[1]+obj_str[2])] = ['$SIZE']
                    self.relations[(obj_name,  color)] = ['$COLOR']
                    if obj_str[2] != "object":
                        self.relations[(obj_name, obj_str[2])] = ['$SHAPE']
            for edge, relation in relations.items():
                src_node, dst_node = edge[0], edge[1]
                if (src_node, dst_node) not in self.relations:
                    self.relations[(src_node, dst_node)] = [relation]
                else:
                    self.relations[(src_node, dst_node)].append(relation)
                if relation != "$IS_INSIDE":
                    # permutable
                    dst_node, src_node = edge[0], edge[1]
                    if (src_node, dst_node) not in self.relations:
                        self.relations[(src_node, dst_node)] = [relation]
                    else:
                        self.relations[(src_node, dst_node)].append(relation)
            for edge, relations in self.relations.items(): 
                src_node, dst_node = edge[0], edge[1]
                for relation in relations:
                    if relation in ["$SAME_ROW", "$SAME_COLUMN", 
                                    "$SAME_SHAPE", "$SAME_COLOR", 
                                    "$SAME_SIZE", "$IS_INSIDE"]:
                        self.G.add_edge(src_node, dst_node, type=relation, key=relation)
                    self.G_full.add_edge(src_node, dst_node, type=relation, key=relation)
            if debug:
                print(self.relations)
        else:
            self.objects = objects
            self.positions = positions
            self.referred_object = referred_object
            relations = self.parse_relations(objects, object_patterns, positions)
            self.relations = relations
            if debug:
                print(self.relations)
            # formulate the graph
            for obj_name, obj in objects.items():
                if referred_object != None and obj_name == referred_object:
                    self.G.add_node(obj_name, type="object", referent=True)
                    self.G_full.add_node(obj_name, type="object", referent=True)
                else:
                    self.G.add_node(obj_name, type="object", referent=False)
                    self.G_full.add_node(obj_name, type="object", referent=False)
            for edge, relations in relations.items():
                src_node, dst_node = edge[0], edge[1]
                for relation in relations:
                    if relation in ["$SAME_ROW", "$SAME_COLUMN", 
                                    "$SAME_SHAPE", "$SAME_COLOR", 
                                    "$SAME_SIZE", "$IS_INSIDE"]:
                        self.G.add_edge(src_node, dst_node, type=relation, key=relation)
                    self.G_full.add_edge(src_node, dst_node, type=relation, key=relation)

        self.relation_color_map = {
            "$SAME_ROW" : "r",
            "$SAME_COLUMN" : "g",
            "$SAME_SHAPE" : "b",
            "$SAME_COLOR" : "c",
            "$SAME_SIZE" : "m",
            "$IS_INSIDE" : "y"
        }
        
    def draw(self, G_to_plot=None, save_file=None):
        """
        This function only draws objects and relations, but not attributes.
        """
        if G_to_plot == None:
            G = self.G
        else:
            G = G_to_plot

        color_map = []
        for node in G.nodes(data=True):
            if "referent" in node[1]:
                if node[1]["referent"]:
                    color_map.append('black')
                else:
                    color_map.append('grey')
            else:
                color_map.append('grey')
        edge_labels=dict(
            [((u,v,),d['type']) for u,v,d in G.edges(data=True)]
        )
            
        import matplotlib.pyplot as plt
        from matplotlib.legend_handler import HandlerTuple
        
        y_off = 0.5
        pos = nx.spring_layout(G)
        
        nx.draw_networkx_nodes(
            G, pos, node_color=color_map
        )

        ax = plt.gca()
        arrow_legend = []
        arrow_label = []
        node_connection_map = {}
        annotation_group_map = {
            "$SAME_ROW" : [],
            "$SAME_COLUMN" : [],
            "$SAME_SHAPE" : [],
            "$SAME_COLOR" : [],
            "$SAME_SIZE" : [],
            "$IS_INSIDE" : []
        }
        for e in G.edges(data=True):
            if ((e[0], e[1]) in node_connection_map and e[2]["type"] in node_connection_map[(e[0], e[1])]) or \
                ((e[1], e[0]) in node_connection_map and e[2]["type"] in node_connection_map[(e[1], e[0])]):
                continue # We draw using bidirectional arrows already!
            if (e[0], e[1]) not in node_connection_map:
                node_connection_map[(e[0], e[1])] = [e[2]["type"]]
            else:
                node_connection_map[(e[0], e[1])].append(e[2]["type"])
            connection_count = len(node_connection_map[(e[0], e[1])])
            if e[2]["type"] == "$IS_INSIDE":
                arrowstyle="<|-"
            else:
                arrowstyle="<|-|>"
            an = ax.annotate(
                "",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(
                    lw=3,
                    arrowstyle=arrowstyle, color=self.relation_color_map[e[2]["type"]],
                    shrinkA=10, shrinkB=10,
                    patchA=None, patchB=None,
                    connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*connection_count)),
                ),
                label = "a"
            )
            annotation_group_map[e[2]["type"]].append(an.arrow_patch)

        groups = []
        legends = []
        for type, group in annotation_group_map.items():
            if len(group) >= 1:
                groups.append(tuple([group[0]]))
                legends.append(type)
        plt.legend(groups, legends, numpoints=1, 
                   handler_map={tuple: HandlerTuple(ndivide=None)}, 
                   loc='upper center', bbox_to_anchor=(0.48, 0.05),
                   ncol=3, fancybox=True, shadow=True, fontsize=11)
        plt.margins(0.2)
        plt.axis('off')
        if save_file != None:
            plt.savefig(save_file, dpi=1000)
            plt.close()
        else:
            plt.show()

    def parse_relations(self, objects, object_patterns, positions):
        relations = OrderedDict({})
        
        # Attributes
        for obj_id, obj_spec in objects.items():
            # relations[(obj_id, obj_spec.size)] = "$SIZE"
            relations[(obj_id, obj_spec.color)] = ["$COLOR"]
            relations[(obj_id, obj_spec.shape)] = ["$SHAPE"]

            # BUG: adding all descriptors to size.
            for to_obj_id, to_obj_spec in objects.items():
                if obj_id != to_obj_id:
                    # depending on the object pattern, we actually
                    # need to be certain on the attributes.
                    if to_obj_spec.shape == obj_spec.shape:
                        if obj_spec.size < to_obj_spec.size:
                            relations[(obj_id, "small"+obj_spec.shape)] = ["$SIZE"]
                        if obj_spec.size > to_obj_spec.size:
                            relations[(obj_id, "big"+obj_spec.shape)] = ["$SIZE"]
                    
                    # If there is color and shape map, we need to add
                    # another set of edges other is color specific size edges.
                    if to_obj_spec.shape == obj_spec.shape and to_obj_spec.color == obj_spec.color:
                        if obj_spec.size < to_obj_spec.size:
                            relations[(obj_id, "small"+obj_spec.color+obj_spec.shape)] = ["$SIZE"]
                        if obj_spec.size > to_obj_spec.size:
                            relations[(obj_id, "big"+obj_spec.color+obj_spec.shape)] = ["$SIZE"]
                    
                    # For non-same shape, it is also possible to have shape descriptor
                    if obj_spec.size < to_obj_spec.size:
                        relations[(obj_id, "smallobject")] = ["$SIZE"]
                    if obj_spec.size > to_obj_spec.size:
                        relations[(obj_id, "bigobject")] = ["$SIZE"]
                    
                    if to_obj_spec.color == obj_spec.color:
                        # For non-same shape, it is also possible to have shape descriptor
                        if obj_spec.size < to_obj_spec.size:
                            relations[(obj_id, "small"+obj_spec.color+"object")] = ["$SIZE"]
                        if obj_spec.size > to_obj_spec.size:
                            relations[(obj_id, "big"+obj_spec.color+"object")] = ["$SIZE"]
                    
        # Relations
        for obj_id_left, obj_spec_left in objects.items():
            for obj_id_right, obj_spec_right in objects.items():
                if obj_id_left != obj_id_right:
                    obj_pos_left = positions[obj_id_left]
                    obj_pos_right = positions[obj_id_right]
                    key = (obj_id_left, obj_id_right)
                    if obj_pos_left.row == obj_pos_right.row:
                        if key not in relations:
                            relations[key] = ["$SAME_ROW"]
                        else:
                            relations[key].append("$SAME_ROW")
                    if obj_pos_left.column == obj_pos_right.column:
                        if key not in relations:
                            relations[key] = ["$SAME_COLUMN"]
                        else:
                            relations[key].append("$SAME_COLUMN")
                    if obj_spec_left.size == obj_spec_right.size:
                        if key not in relations:
                            relations[key] = ["$SAME_SIZE"]
                        else:
                            relations[key].append("$SAME_SIZE")
                    if obj_spec_left.color == obj_spec_right.color:
                        if key not in relations:
                            relations[key] = ["$SAME_COLOR"]
                        else:
                            relations[key].append("$SAME_COLOR")
                    if obj_spec_left.shape == obj_spec_right.shape:
                        if key not in relations:
                            relations[key] = ["$SAME_SHAPE"]
                        else:
                            relations[key].append("$SAME_SHAPE")
                    # For IsInside relations.
                    if obj_spec_right.shape == "box":
                        if obj_pos_left.row >= obj_pos_right.row and \
                            obj_pos_left.row < obj_pos_right.row+obj_spec_right.size and \
                            obj_pos_left.column >= obj_pos_right.column and \
                            obj_pos_left.column < obj_pos_right.column+obj_spec_right.size:
                            if key not in relations:
                                relations[key] = ["$IS_INSIDE"]
                            else:
                                relations[key].append("$IS_INSIDE")
        return relations
    
    def find_determiners(
        self,
        pattern_graph, 
        referred_object='$OBJ_0', 
        debug=False
    ):
        determiner_map = OrderedDict({
            referred_object: "the"
        })
        G = self.G_full.copy()
        # Consolidate all the attributes for all nodes.
        G_node_attr_map = {}
        for edge in G.edges(data=True):
            if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                   "$SAME_SHAPE", "$SAME_COLOR", 
                                   "$SAME_SIZE", "$IS_INSIDE"]:
                pass # We don't care relations 
                """
                Caveats: we recognize that for longer logic chains,
                considering the relations could be more percise!
                """
            else:
                if edge[0] in G_node_attr_map.keys():
                    G_node_attr_map[edge[0]].add(edge[1] + " " + edge[2]["type"])
                else:
                    G_node_attr_map[edge[0]] = set([edge[1] + " " + edge[2]["type"]])
        if debug:
            print(G_node_attr_map)
        
        sub_G = pattern_graph.G_full.copy()
        sub_G_node_attr_map = {}
        for edge in sub_G.edges(data=True):
            if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                   "$SAME_SHAPE", "$SAME_COLOR", 
                                   "$SAME_SIZE", "$IS_INSIDE"]:
                pass
            else:
                if edge[0] in sub_G_node_attr_map.keys():
                    sub_G_node_attr_map[edge[0]].add(edge[1] + " " + edge[2]["type"])
                else:
                    sub_G_node_attr_map[edge[0]] = set([edge[1] + " " + edge[2]["type"]])
        
        
        for pattern_node_name, pattern_attr_set in sub_G_node_attr_map.items():
            mentions = 0
            definite = True
            for node_name, attr_set in G_node_attr_map.items():
                if len(attr_set.intersection(pattern_attr_set)) == len(pattern_attr_set):
                    mentions += 1
                    if mentions == 2:
                        definite = False
                        break
            if definite:
                determiner_map[pattern_node_name] = "the"
            else:
                determiner_map[pattern_node_name] = "a"
        
        if debug:
            print(sub_G_node_attr_map)
            print(determiner_map)
        determiner_map[referred_object]  = "the"

        return determiner_map
    
    def find_referred_object_super_fast(
        self, relation_pattern, referred_object="$OBJ_0", 
        pattern="$OBJ_0 ^ $OBJ_1 & $OBJ_2",
        debug=False
    ):
        """
        We need this super fast algorithm to do pattern matching.
        Otherwise, who problem becomes unscalable!
        
        Note that this only works for a fixed pattern searching,
        which is:
        $OBJ_0 ^ $OBJ_1 & $OBJ_2
        
        We also added in support for:
        $OBJ_0 ^ $OBJ_1 ^ $OBJ_2
        """
        
        # Let us support more!
        
        # TODO: integrate this into the conditions below.
        matched_pivot_node = set([])
        G = self.G_full.copy()
        G_to_plot = self.G.copy()
        sub_G = relation_pattern.G_full.copy()
        
        sub_G_node_attr_map = {}
        rel_reverse_map = {}
        
        G_node_attr_map = {}
        G_nbr_map = {}
        G_edge_relation_map = {}
        for edge in G.edges(data=True):
            if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                   "$SAME_SHAPE", "$SAME_COLOR", 
                                   "$SAME_SIZE", "$IS_INSIDE"]:
                if edge[0] in G_node_attr_map.keys():
                    G_node_attr_map[edge[0]].add(edge[2]["type"])
                else:
                    G_node_attr_map[edge[0]] = set([edge[2]["type"]])
                if edge[0] in G_nbr_map.keys():
                    G_nbr_map[edge[0]].add(edge[1])
                else:
                    G_nbr_map[edge[0]] = set([edge[1]])
                if (edge[0], edge[1]) in G_edge_relation_map.keys():
                    G_edge_relation_map[(edge[0], edge[1])].add(edge[2]["type"])
                else:
                    G_edge_relation_map[(edge[0], edge[1])] = set([edge[2]["type"]])
            else:
                if edge[0] in G_node_attr_map.keys():
                    G_node_attr_map[edge[0]].add(edge[1] + " " + edge[2]["type"])
                else:
                    G_node_attr_map[edge[0]] = set([edge[1] + " " + edge[2]["type"]])
        
        if pattern == "$OBJ_0 ^ $OBJ_1 ^ $OBJ_2":
            
            sub_G_edge_relation_map = {}
            for edge in sub_G.edges(data=True):
                if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                       "$SAME_SHAPE", "$SAME_COLOR", 
                                       "$SAME_SIZE", "$IS_INSIDE"]:
                    if edge[0] in sub_G_node_attr_map.keys():
                        sub_G_node_attr_map[edge[0]].add(edge[2]["type"])
                    else:
                        sub_G_node_attr_map[edge[0]] = set([edge[2]["type"]])
                    if (edge[0], edge[1]) in sub_G_edge_relation_map.keys():
                        assert False
                    else:
                        sub_G_edge_relation_map[(edge[0], edge[1])] = edge[2]["type"]
                else:                    
                    if edge[0] in sub_G_node_attr_map.keys():
                        sub_G_node_attr_map[edge[0]].add(edge[1] + " " + edge[2]["type"])
                    else:
                        sub_G_node_attr_map[edge[0]] = set([edge[1] + " " + edge[2]["type"]])
            
            for node_name, attr_set in G_node_attr_map.items():
                if len(sub_G_node_attr_map["$OBJ_0"].intersection(attr_set)) == len(sub_G_node_attr_map["$OBJ_0"]):
                    # Now, we just need to find 2 nbr nodes matchs $OBJ_1 and $OBJ_2 attributes at
                    # the same time.
                    for nbr in G_nbr_map[node_name]:
                        if sub_G_edge_relation_map[("$OBJ_0", "$OBJ_1")] in G_edge_relation_map[(node_name, nbr)] and \
                            len(sub_G_node_attr_map["$OBJ_1"].intersection(G_node_attr_map[nbr])) == len(sub_G_node_attr_map["$OBJ_1"]):
                            for sub_nbr in G_nbr_map[nbr]:
                                if sub_nbr != node_name:
                                    if sub_G_edge_relation_map[("$OBJ_1", "$OBJ_2")] in G_edge_relation_map[(nbr, sub_nbr)] and \
                                        len(sub_G_node_attr_map["$OBJ_2"].intersection(G_node_attr_map[sub_nbr])) == len(sub_G_node_attr_map["$OBJ_2"]):
                                        # we have a match for OBJ_2
                                        matched_pivot_node.add(node_name)
            
            return matched_pivot_node
        
        for edge in sub_G.edges(data=True):
            if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                   "$SAME_SHAPE", "$SAME_COLOR", 
                                   "$SAME_SIZE", "$IS_INSIDE"]:

                if edge[0] in sub_G_node_attr_map.keys():
                    sub_G_node_attr_map[edge[0]].add(edge[2]["type"])
                else:
                    sub_G_node_attr_map[edge[0]] = set([edge[2]["type"]])
                child_node = edge[1] if edge[0] == "$OBJ_0" else edge[0]
                if child_node in rel_reverse_map.keys():
                    assert rel_reverse_map[child_node] == edge[2]["type"] # safe-belt
                rel_reverse_map[child_node] = edge[2]["type"]
            else:                    
                if edge[0] in sub_G_node_attr_map.keys():
                    sub_G_node_attr_map[edge[0]].add(edge[1] + " " + edge[2]["type"])
                else:
                    sub_G_node_attr_map[edge[0]] = set([edge[1] + " " + edge[2]["type"]])
        
        if len(rel_reverse_map) == 2:
            for node_name, attr_set in G_node_attr_map.items():
                if len(sub_G_node_attr_map["$OBJ_0"].intersection(attr_set)) == len(sub_G_node_attr_map["$OBJ_0"]):
                    # Now, we just need to find 2 nbr nodes matchs $OBJ_1 and $OBJ_2 attributes at
                    # the same time.
                    match_result_map = {
                        "$OBJ_1" : set([]), 
                        "$OBJ_2" : set([]), 
                    }
                    for nbr in G_nbr_map[node_name]:
                        if rel_reverse_map["$OBJ_1"] in G_edge_relation_map[(node_name, nbr)]:
                            if len(sub_G_node_attr_map["$OBJ_1"].intersection(G_node_attr_map[nbr])) == len(sub_G_node_attr_map["$OBJ_1"]):
                                # we have a match for OBJ_1
                                match_result_map["$OBJ_1"].add(nbr)
                        if rel_reverse_map["$OBJ_2"] in G_edge_relation_map[(node_name, nbr)]:
                            if len(sub_G_node_attr_map["$OBJ_2"].intersection(G_node_attr_map[nbr])) == len(sub_G_node_attr_map["$OBJ_2"]):
                                # we have a match for OBJ_2
                                match_result_map["$OBJ_2"].add(nbr)

                    if len(match_result_map["$OBJ_1"]) > 0 and len(match_result_map["$OBJ_2"]) > 0:
                        if len(match_result_map["$OBJ_1"].union(match_result_map["$OBJ_2"])) > 1:
                            matched_pivot_node.add(node_name)
        elif len(rel_reverse_map) == 3:
            for node_name, attr_set in G_node_attr_map.items():
                if len(sub_G_node_attr_map["$OBJ_0"].intersection(attr_set)) == len(sub_G_node_attr_map["$OBJ_0"]):
                    match_result_map = {
                        "$OBJ_1" : set([]), 
                        "$OBJ_2" : set([]),
                        "$OBJ_3" : set([]),
                    }
                    for nbr in G_nbr_map[node_name]:
                        if rel_reverse_map["$OBJ_1"] in G_edge_relation_map[(node_name, nbr)]:
                            if len(sub_G_node_attr_map["$OBJ_1"].intersection(G_node_attr_map[nbr])) == len(sub_G_node_attr_map["$OBJ_1"]):
                                # we have a match for OBJ_1
                                match_result_map["$OBJ_1"].add(nbr)
                        if rel_reverse_map["$OBJ_2"] in G_edge_relation_map[(node_name, nbr)]:
                            if len(sub_G_node_attr_map["$OBJ_2"].intersection(G_node_attr_map[nbr])) == len(sub_G_node_attr_map["$OBJ_2"]):
                                # we have a match for OBJ_2
                                match_result_map["$OBJ_2"].add(nbr)
                        if rel_reverse_map["$OBJ_3"] in G_edge_relation_map[(node_name, nbr)]:
                            if len(sub_G_node_attr_map["$OBJ_3"].intersection(G_node_attr_map[nbr])) == len(sub_G_node_attr_map["$OBJ_3"]):
                                # we have a match for OBJ_2
                                match_result_map["$OBJ_3"].add(nbr)
                    # sub-problem of minimum set cover problem.
                    if len(match_result_map["$OBJ_1"]) > 0 and len(match_result_map["$OBJ_2"]) > 0 and len(match_result_map["$OBJ_3"]) > 0:
                        if len(match_result_map["$OBJ_1"].union(match_result_map["$OBJ_2"])) > 1 and \
                            len(match_result_map["$OBJ_1"].union(match_result_map["$OBJ_3"])) > 1 and \
                            len(match_result_map["$OBJ_2"].union(match_result_map["$OBJ_3"])) > 1:
                            if len(match_result_map["$OBJ_1"].union(match_result_map["$OBJ_2"]).union(match_result_map["$OBJ_3"])) > 2:
                                matched_pivot_node.add(node_name)
        elif len(rel_reverse_map) == 0:
            for node_name, attr_set in G_node_attr_map.items():
                if len(sub_G_node_attr_map["$OBJ_0"].intersection(attr_set)) == len(sub_G_node_attr_map["$OBJ_0"]):
                    matched_pivot_node.add(node_name)
        elif len(rel_reverse_map) == 1:
            for node_name, attr_set in G_node_attr_map.items():
                if len(sub_G_node_attr_map["$OBJ_0"].intersection(attr_set)) == len(sub_G_node_attr_map["$OBJ_0"]):
                    for nbr in G_nbr_map[node_name]:
                        if rel_reverse_map["$OBJ_1"] in G_edge_relation_map[(node_name, nbr)]:
                            if len(sub_G_node_attr_map["$OBJ_1"].intersection(G_node_attr_map[nbr])) == len(sub_G_node_attr_map["$OBJ_1"]):
                                # we have a match for OBJ_1
                                matched_pivot_node.add(node_name)

        return matched_pivot_node

    def find_referred_object(self, relation_pattern, referred_object="$OBJ_0", debug=False):
        """
        This algorithm works for any subtree matching,
        we use the above one to speed up in our case.
        This is a NP-hard problem after all...
        """
        G = self.G_full.copy()
        G_to_plot = self.G.copy()
        sub_G = relation_pattern.G_full.copy()

        # Major speed up!
        # We will remove irrelevant edges from G based on relations in sub_G
        sub_G_relations = set([])
        node_attr_count = {}
        """
        This following is important to speed up this algorithm
        in complex graphs. We will record all attributes for each
        node in this pattern. Later, in the full graph, we know that
        the intersection set of attribution set of any node in the 
        graph, if potentially is belong to this pattern, must covering
        all attribute of at least 1 node in this pattern attribution set of
        each node.
        """
        sub_G_node_attr_map = {}
        for edge in sub_G.edges(data=True):
            if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                   "$SAME_SHAPE", "$SAME_COLOR", 
                                   "$SAME_SIZE", "$IS_INSIDE"]:
                sub_G_relations.add(edge[2]["type"])
                if edge[0] in sub_G_node_attr_map.keys():
                    sub_G_node_attr_map[edge[0]].add(edge[2]["type"])
                else:
                    sub_G_node_attr_map[edge[0]] = set([edge[2]["type"]])
            else:
                sub_G_relations.add(edge[1] + " " + edge[2]["type"]) # attribute based!
                if edge[0] in node_attr_count.keys():
                    node_attr_count[edge[0]] += 1
                else:
                    node_attr_count[edge[0]] = 1
                    
                if edge[0] in sub_G_node_attr_map.keys():
                    sub_G_node_attr_map[edge[0]].add(edge[1] + " " + edge[2]["type"])
                else:
                    sub_G_node_attr_map[edge[0]] = set([edge[1] + " " + edge[2]["type"]])
        
        to_remove_edges = []
        removed_edge_count = 0
        for edge in G.edges(data=True):
            if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                   "$SAME_SHAPE", "$SAME_COLOR", 
                                   "$SAME_SIZE", "$IS_INSIDE"]:
                if edge[2]["type"] not in sub_G_relations:
                    to_remove_edges.append(edge)
            else:
                if edge[1] + " " + edge[2]["type"] not in sub_G_relations:
                    to_remove_edges.append(edge)
        for edge in to_remove_edges:
            if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                   "$SAME_SHAPE", "$SAME_COLOR", 
                                   "$SAME_SIZE", "$IS_INSIDE"]:
                G_to_plot.remove_edge(edge[0], edge[1], key=edge[2]["type"])
            G.remove_edge(edge[0], edge[1], key=edge[2]["type"])
            removed_edge_count += 1
        # Go through again for deleting nodes.
        min_node_attr = 99
        for k, v in node_attr_count.items():
            if v < min_node_attr:
                min_node_attr = v
        to_remove_node_attr_count = {}
        to_remove_node_relation_count = {}
        G_node_attr_map = {}
        for edge in G.edges(data=True):
            if edge[2]["type"] in ["$SAME_ROW", "$SAME_COLUMN", 
                                   "$SAME_SHAPE", "$SAME_COLOR", 
                                   "$SAME_SIZE", "$IS_INSIDE"]:
                if edge[0] in to_remove_node_relation_count.keys():
                    to_remove_node_relation_count[edge[0]] += 1
                else:
                    to_remove_node_relation_count[edge[0]] = 1
                if edge[0] in G_node_attr_map.keys():
                    G_node_attr_map[edge[0]].add(edge[2]["type"])
                else:
                    G_node_attr_map[edge[0]] = set([edge[2]["type"]])
            else:
                if edge[0] in to_remove_node_attr_count.keys():
                    to_remove_node_attr_count[edge[0]] += 1
                else:
                    to_remove_node_attr_count[edge[0]] = 1
                if edge[0] in G_node_attr_map.keys():
                    G_node_attr_map[edge[0]].add(edge[1] + " " + edge[2]["type"])
                else:
                    G_node_attr_map[edge[0]] = set([edge[1] + " " + edge[2]["type"]])
        
        # Formulate all to-remove nodes.
        to_remove_nodes = set([])
        for k, v in to_remove_node_attr_count.items():
            if v < min_node_attr:
                to_remove_nodes.add(k)
            if k not in to_remove_node_relation_count:
                # dangling node
                to_remove_nodes.add(k)
        
        for node_name, attr_set in G_node_attr_map.items():
            removable = True
            for pattern_node_name, pattern_attr_set in sub_G_node_attr_map.items():
                """
                If the attribute set of current node can cover (superset) attribute set of 
                one of the node in the pattern graph, we cannot delete it.
                """
                if len(attr_set.intersection(pattern_attr_set)) == len(pattern_attr_set):
                    removable = False
                    break # this node cannot be removed.
            if removable:
                to_remove_nodes.add(node_name)
        
        if debug:
            print(f"removing {removed_edge_count} edges ...")
            print(f"removing {len(to_remove_nodes)} nodes ...")
            self.draw(G_to_plot=G_to_plot)
            
        line_G = line_graph(DiGraph(G))
        line_sub_G = line_graph(DiGraph(sub_G))
        
        DiGM = isomorphism.DiGraphMatcher(line_graph(DiGraph(G)), line_graph(DiGraph(sub_G)))
        
        valid_referred_nodes = []
        for edge_match in DiGM.subgraph_isomorphisms_iter():
            valid = True
            for G_pair, sub_G_pair in edge_match.items():
                # First pass: we can check naive node type. They need to match with each other.
                if G_pair[0].startswith("$") and G_pair[0][0] != sub_G_pair[0][0]:
                    valid = False
                    break # not valid.
                if G_pair[1].startswith("$") and G_pair[1][0] != sub_G_pair[1][0]:
                    valid = False
                    break # not valid.
                if not G_pair[1].startswith("$") and G_pair[1] != sub_G_pair[1]:
                    valid = False
                    break # not valid.
                # Second pass: if it is a edge between nodes, we need to match relations.
                if G_pair[0].startswith("$") and G_pair[1].startswith("$") and \
                    sub_G_pair[0].startswith("$") and sub_G_pair[1].startswith("$"):
                    # This is a relational pair.
                    overlap_relations = set(self.relations[G_pair]).intersection(
                        set(relation_pattern.relations[sub_G_pair])
                    )
                    if len(overlap_relations) == 0:
                        valid = False
                        break # not valid.
            if valid:
                pattern_node_map = {}
                for G_pair, sub_G_pair in edge_match.items():
                    if G_pair[0].startswith("$") and G_pair[1].startswith("$") and \
                        sub_G_pair[0].startswith("$") and sub_G_pair[1].startswith("$"):
                        if sub_G_pair[0] not in pattern_node_map:
                            pattern_node_map[sub_G_pair[0]] = G_pair[0]
                        else:
                            if pattern_node_map[sub_G_pair[0]] != G_pair[0]:
                                valid = False
                                break # not valid.
                        if sub_G_pair[1] not in pattern_node_map:
                            pattern_node_map[sub_G_pair[1]] = G_pair[1]
                        else:     
                            if pattern_node_map[sub_G_pair[1]] != G_pair[1]:
                                valid = False
                                break # not valid.
            
            if valid:
                valid_referred_nodes.append(pattern_node_map[referred_object])
        return set(valid_referred_nodes)