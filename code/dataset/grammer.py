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

from utils import one_hot
from utils import generate_possible_object_names
from utils import numpy_array_to_image

from vocabulary import *
from object_vocabulary import *
from world import *

class Grammer(object):
    """
    Core functionality of ReaSCAN - Object Relational Based Grammer.
    
    Form of the command:
    
    ROOT:= VV OBJ (that is REL_CLAUSE (and REL_CLAUSE)*)* ADV
    REL_CLAUSE:= "REL OBJ"
    
    Design Document:
    ### rel and possible obj rules
    (refer object, pivot object, relation)

    **First order.**
    1. For same row and column relation, refer object and pivot object can be anything, 
    since it naturally needs to be grounded in the world!
    2. For color, refer object definietly do not need to contain color adj, 
    pivot objects may contain or may not. For cases where contain, the purpose 
    was to let it (1) understand color (2) understand relations. 
    For cases where do not contain, it needs (1) understand color (2) understand relations (3) reasoning about relations.
    3. For shape, it is the same as 2.
    4. For size, it is also the same as 2. Size a little different here. 
    Small and Large is not a direct descriptor of size here which is ranged from 1-4. 
    If "same as" then the size is defined by the pivot object. Is there is need for 
    size adj for referred obj then? No. Since it will be redundant now. Ok, then this is consistence. 
    5. For inside relation, the referred object cannot any obj i think. 
    For pivot object it must be a box and any color and size. 
    In case larger box, there could be multiple objects in it. 
    then we could generate the world based on the referred obj sampled i think. it should be easy.

    **Recursive.**
    1. repeat relation is not needed. no information gain there i think.
    2. just recursively apply the rules defined above i think.

    **Parallel.**
    1. repeat relation is not needed. no information gain there i think.
    2. I think it is essentially the same, using rules defined above on the referred obj.

    **Methods.**
    1. Parent-marked tree. Starting sampling from the leaf nodes and going up should work i think.
    2. This aims to provide a pretty general solution with any level of recursive.
    3. As you can see if the tree go too deep and with a small vocab, 
    it becomes impossible to sample a good command, and this is valid situation. 
    You can decrease the tree height or increase your vocab.
    4. I think we will use some sort of top-down DFS to sample, so have to think backward. Be careful!
    
    """
    # TODO: moving this to our vocab?
    # constant
    DEPTH_REL_CLAUSE_PATTERN = " ".join(["^", "$REL_CLAUSE"])
    WIDTH_REL_CLAUSE_PATTERN = " ".join(["&", "$REL_CLAUSE"])
    # compositions
    REL_CLAUSE_REGEX = "\$REL_CLAUSE"
    REL_CLAUSE = " ".join(["$REL", "$OBJ"])
    REL_CLAUSE_OBJ_ONLY = " ".join(["$OBJ"])
    # logics
    RECURSIVE_REGEX = "^"
    RECURSIVE = "that is"
    AND_REGEX = "&"
    AND = "and"
    # adj abstractions
    SIZE_REGEX = "$SIZE"
    COLOR_REGEX = "$COLOR"
    SHAPE_REGEX = "$SHAPE"
    ABSTRACT_SHAPE_REGEX = "$ABS_SHAPE"
    # relation abstraction
    SAME_ROW_REGEX = "$SAME_ROW"
    SAME_COL_REGEX = "$SAME_COLUMN"
    SAME_SHAPE_REGEX = "$SAME_SHAPE"
    SAME_COLOR_REGEX = "$SAME_COLOR"
    SAME_SIZE_REGEX = "$SAME_SIZE"
    IS_INSIDE_REGEX = "$IS_INSIDE"
    
    # other variables
    HIDDEN_ROW_REGEX = "#ROW"
    HIDDEN_COL_REGEX = "#COLUMN"
    HIDDEN_IS_INSIDE_REGEX = "#IS_INSIDE"
    
    # these should be moved outside grammer i think
    REL_REGEX_VOCAB_MAPPING = {
        SAME_ROW_REGEX : "in the same row as",
        SAME_COL_REGEX : "in the same column as",
        SAME_SHAPE_REGEX : "in the same shape as",
        SAME_COLOR_REGEX : "in the same color as",
        SAME_SIZE_REGEX : "in the same size as",
        IS_INSIDE_REGEX : "inside of"
    }
    
    def __init__(self, vocabulary=None, logic_depth_max=2, logic_width_max=1):
        self.logic_depth_max = logic_depth_max
        self.logic_width_max = logic_width_max
        
        self.vocabulary = vocabulary # this is not necessary if not grounding actual commands
    
    def _sample_object_pattern(self, root=False):
        """
        This function only sample patterns in abstractions.
        """
        size_pool = ["", self.SIZE_REGEX]
        color_pool = ["", self.COLOR_REGEX]
        shape_pool = [self.SHAPE_REGEX, self.ABSTRACT_SHAPE_REGEX]
        # including self.ABSTRACT_SHAPE_REGEX is very ambitious!
        sampled_patterns = []
        for obj_comp in product(size_pool, 
                                color_pool,
                                shape_pool):
            sampled_pattern = []
            if root: # root can be in simple form?
                pass
            else:
                if obj_comp[0] == "" and obj_comp[1] == "":
                    continue
            for part in obj_comp:
                if part != "":
                    sampled_pattern += [part]
            sampled_patterns += [" ".join(sampled_pattern)]
        return sampled_patterns
    
    # @Deprecated
    def _sample_object_grammer(self, object_pool=None, composition_only=False, allow_arbitrary_object=True, 
                               color_considered=True, size_considered=True):
        object_compositions = []
        if object_pool == None:
            arbitrary_object = ['object'] if allow_arbitrary_object else []
            object_pool = vocabulary.get_nouns() + arbitrary_object
        # We don't allow plain case
        # They are not interesting other than adding in sampling difficulties!
        # plain
        # if not composition_only:
        #     object_compositions += [(obj,) for obj in object_pool]
        # comp-2-size
        if size_considered:
            for obj_comp in product(vocabulary.get_size_adjectives(), 
                                    object_pool):
                object_compositions += [obj_comp]
        # comp-2-color
        if color_considered:
            for obj_comp in product(vocabulary.get_color_adjectives(), 
                                    object_pool):
                object_compositions += [obj_comp]
        # comp-3
        if size_considered and color_considered:
            for obj_comp in product(vocabulary.get_size_adjectives(), 
                                    vocabulary.get_color_adjectives(), 
                                    object_pool):
                object_compositions += [obj_comp]
        return object_compositions
    
    def _sample_grammer_pattern_reascan_train(self, obj_hiarch_only=True):
        """
        This this a helper function to return a list of relation grammers
        ReaSCAN look at. If you wish to randomly sample, you should use 
        the following function.
        
        This is helper as it will not explode your disk to generate a way
        too large version of the dataset.
        """
        return ['$OBJ_0',
                '$OBJ_0 ^ $OBJ_1',
                '$OBJ_0 ^ $OBJ_1 & $OBJ_2']
    
    def _sample_grammer_pattern_reascan_length_split(self, obj_hiarch_only=True):
        """
        This this a helper function to return a list of relation grammers
        ReaSCAN look at. If you wish to randomly sample, you should use 
        the following function.
        
        This is helper as it will not explode your disk to generate a way
        too large version of the dataset.
        """
        return ['$OBJ_0 ^ $OBJ_1 & $OBJ_2 ^ $OBJ_3',
                '$OBJ_0 ^ $OBJ_1 ^ $OBJ_2 ^ $OBJ_3']
    
    def _sample_grammer_pattern(self, obj_hiarch_only=True):
        """
        Instead of building this using a tree, we decide to use this
        as the complexity of SCAN-like commands are not high.
        
        The fixed template in this case is something like:
        ROOT:= VV OBJ (that is REL_CLAUSE (and REL_CLAUSE)*)* ADV
        REL_CLAUSE:= "REL OBJ"
        
        We will use depth to control time of recursions, and then
        width to control the expandsion of "and" logics in each level.
        """
        all_clauses = []
        rel_clauses = []
        depth_rel_clause = []
        for i in range(self.logic_depth_max):
            depth_rel_clause += [self.DEPTH_REL_CLAUSE_PATTERN]
            rel_clauses.append(copy.deepcopy(depth_rel_clause))
        
        permuation_numbers = [i for i in range(self.logic_width_max+1)]
        # using iteration to replace recursion by using permutations
        for r_c in rel_clauses:
            permuations = []
            for i in range(len(r_c)):
                permuations += [copy.deepcopy(permuation_numbers)]
            counts_permuations = product(*permuations)
            for counts in counts_permuations:
                if not obj_hiarch_only:
                    rel_clause = " ".join(self._append_depth_clause(r_c, counts))
                    rel_clause = re.sub(self.REL_CLAUSE_REGEX, self._get_rel_clause(), rel_clause)
                    raw_clause = f"$VV $OBJ {rel_clause} $ADV"
                else:
                    # this generates the OBJ dependency grammer only
                    rel_clause = " ".join(self._append_depth_clause(r_c, counts))
                    rel_clause = re.sub(self.REL_CLAUSE_REGEX, self._get_rel_clause(obj_only=True), rel_clause)
                    raw_clause = f"$OBJ {rel_clause}"
                # Should we place OBJ with OBJ index?
                # this would be easier later parsing.
                raw_clause = raw_clause.split(" ")
                OBJ_idx = 0
                process_clause = []
                for part in raw_clause:
                    if part.startswith("$OBJ"):
                        new_OBJ_str = f"$OBJ_{OBJ_idx}"
                        process_clause.append(new_OBJ_str)
                        OBJ_idx += 1
                    else:
                        process_clause.append(part)
                process_clause = " ".join(process_clause)
                all_clauses.append(process_clause)
                
        return all_clauses

    def _conditional_sample_relation_pattern(self, object_pattern, excluding_relation_patterns=[]):
        """
        Based on the object pattern, we sample potential relations to next level.
        """
        rel_patterns = [self.SAME_ROW_REGEX, self.SAME_COL_REGEX, self.IS_INSIDE_REGEX]
        if self.SIZE_REGEX not in object_pattern:
            rel_patterns.append(self.SAME_SIZE_REGEX)
        if self.COLOR_REGEX not in object_pattern:
            rel_patterns.append(self.SAME_COLOR_REGEX)
        if self.SHAPE_REGEX not in object_pattern:
            rel_patterns.append(self.SAME_SHAPE_REGEX)
        if self.ABSTRACT_SHAPE_REGEX not in object_pattern:
            pass # do nothing, since it must be the case shape is defined
        filtered_rel_patterns = [rel_p for rel_p in rel_patterns if rel_p not in excluding_relation_patterns]
        return filtered_rel_patterns
    
    def _append_depth_clause(self, append_to_list=["", ""], append_depth_list=(1,1)):
        ret_str_list = []
        for i in range(len(append_to_list)):
            ret_str = self._append_depth_clause_helper(append_to_list[i], append_depth_list[i])
            ret_str_list += [ret_str]
        return ret_str_list
            
    def _append_depth_clause_helper(self, append_to, append_depth):
        ret_str = [append_to]
        for i in range(append_depth):
            ret_str.append(self.WIDTH_REL_CLAUSE_PATTERN)
        return " ".join(ret_str)
    
    def _get_rel_clause(self, obj_only=False):
        if obj_only:
            return self.REL_CLAUSE_OBJ_ONLY
        else:
            return self.REL_CLAUSE

    def build_dependency_graph(self, object_dependency_str='$OBJ_0 ^ $OBJ_1 & $OBJ_2 ^ $OBJ_3 & $OBJ_4'):
        """
        This is a core function that builds obj dependency tree based on the sampled grammer.
        This will be used to do conditional sampling of objects in our final expression.
        """
        # TODO:
        # Actually, I am not sure if this bug free for other complicated cases. Here, I am only
        # debugging based on recursive levels considered in ReaSCAN.

        dependency_map = OrderedDict({})

        object_dependency_list = object_dependency_str.split(" ")
        # remembering the current referred nodes
        recursive_parent = ""
        and_parent = ""
        for i in range(len(object_dependency_list)):
            obj_curr = object_dependency_list[i]
            if i == 0:
                if recursive_parent == "":
                    recursive_parent = obj_curr
                if and_parent == "":
                    and_parent = obj_curr
                continue # the first obj has no relation

            if obj_curr.startswith("$OBJ"):
                if object_dependency_list[i-1] == self.RECURSIVE_REGEX:
                    # this is a ^ logic
                    if recursive_parent in dependency_map.keys():
                        dependency_map[recursive_parent].append(obj_curr)
                    else:
                        dependency_map[recursive_parent] = [obj_curr]
                elif object_dependency_list[i-1] == self.AND_REGEX:
                    # this is a & logic
                    if recursive_parent in dependency_map.keys():
                        dependency_map[recursive_parent].append(obj_curr)
                    else:
                        dependency_map[recursive_parent] = [obj_curr]
                else:
                    pass # not implemented
            else:
                if obj_curr == self.RECURSIVE_REGEX:
                    recursive_parent = object_dependency_list[i-1]
                else:
                    pass # we can simply skip for other relation logics
        return dependency_map
    
    def sample_object_relation_grammer(self, root, dependency_graph, enforce_is_inside_once=True):
        sampled_object_relation_grammers = []
        # we enforce isinside to happen only once
        if len(dependency_graph) == 0:
            # gSCAN
            rel_map = OrderedDict({})
            obj_patterns = ['$SHAPE', '$COLOR $SHAPE', '$SIZE $SHAPE', '$SIZE $COLOR $SHAPE']
            for obj_pattern in obj_patterns:
                obj_pattern_map = OrderedDict({
                    root: obj_pattern
                })
                sampled_object_relation_grammers.append(
                        (copy.deepcopy(obj_pattern_map), copy.deepcopy(rel_map)))
            return sampled_object_relation_grammers
        # sample based on objects first, then relations.
        objects = set([])
        for k, v in dependency_graph.items():
            objects.add(k)
            for i in v:
                objects.add(i)
        object_count = len(objects)
        objects = [f"$OBJ_{i}" for i in range(object_count)] # remake so we can order
        obj_permutator = []
        root = True
        for _ in objects:
            obj_patterns = self._sample_object_pattern(root=root)
            obj_permutator.append(obj_patterns)
            root = False
        obj_permuations = product(*obj_permutator)
        
        # let us start sampling grammers
        for obj_p in obj_permuations:
            obj_pattern_map = dict(zip(objects, obj_p))
            rel_permutator = []
            is_inside_included = False
            for parent_node, child_node in dependency_graph.items():
                for i in range(len(child_node)):
                    excluding_relation_patterns = []
                    potential_rel_patterns = \
                        self._conditional_sample_relation_pattern(
                            obj_pattern_map[parent_node], 
                            excluding_relation_patterns=excluding_relation_patterns)
                    rel_permutator.append(potential_rel_patterns)
                    is_inside_included = True
            rel_permuations = product(*rel_permutator)
            
            # iterate potential rel permutations
            for rel_p in rel_permuations:
                idx = 0
                valid = True
                rel_map = OrderedDict({})
                for parent_node, child_node in dependency_graph.items():
                    splice_rel_p = rel_p[idx:idx+len(child_node)]
                    if len(splice_rel_p) != len(set(splice_rel_p)):
                        valid = False
                        break
                    for i in range(len(child_node)):
                        rel_map[(parent_node, child_node[i])] = splice_rel_p[i]
                    for pair_i, rel_i in rel_map.items():
                        if rel_i == self.SAME_COLOR_REGEX:
                            if self.COLOR_REGEX in obj_pattern_map[pair_i[-1]]:
                                # We now don't allow these very simple cases:
                                # for example:
                                # XXX same color as the red square.
                                # the model only need to learn simple parsing.
                                valid = False
                                break
                        if rel_i == self.SAME_SIZE_REGEX:
                            if self.SIZE_REGEX in obj_pattern_map[pair_i[-1]]:
                                # For size, it is a similar case as color.
                                # Since, whenever small is mentioned, we only
                                # allow two sizes, it seems to be redundant
                                # to say
                                # XXX same size as the small red circle.
                                valid = False
                                break
                        if rel_i == self.SAME_SHAPE_REGEX:
                            if not self.ABSTRACT_SHAPE_REGEX in obj_pattern_map[pair_i[-1]]:
                                # We need to aovid this as well.
                                # XXX same shape as the square, this is again
                                # essentially cannot be true!
                                valid = False
                                break
                                
                             # we need be more restrict.
                            
                        if self.ABSTRACT_SHAPE_REGEX in obj_pattern_map[pair_i[-1]]:
                            if rel_i != self.SAME_SHAPE_REGEX:
                                # We need to aovid this as well.
                                # XXX same shape as the square, this is again
                                # essentially cannot be true!
                                valid = False
                                break
                    if not valid:
                        break
                    idx += len(child_node)
                if valid:
                    # i think the object map + rel_map is one of the valid combo then!
                    sampled_object_relation_grammers.append(
                        (copy.deepcopy(obj_pattern_map), copy.deepcopy(rel_map)))
        return sampled_object_relation_grammers
    
    def grounding_grammer_with_vocabulary(self, grammer_pattern, obj_pattern_map, rel_map):
        """
        This will generate actual commands with grammer and vocabs
        """
        obj_vocabs_map = OrderedDict({})
        obj_ajs_permutator = []
        obj_list = []
        for obj, pattern in obj_pattern_map.items():
            obj_vocabs_map[obj] = []
            obj_list.append(obj)
            must_box = False
            for pair, rel in rel_map.items():
                if rel == self.IS_INSIDE_REGEX and pair[-1] == obj:
                    must_box = True
                    break
            if must_box:
                adj_permutator = []
                if self.SIZE_REGEX in pattern:
                    adj_permutator.append(self.vocabulary.get_size_adjectives())
                if self.COLOR_REGEX in pattern:
                    adj_permutator.append(self.vocabulary.get_color_adjectives())
                adj_permutator.append(["box"]) # remove hard code here probably!
            else:
                adj_permutator = []
                if self.SIZE_REGEX in pattern:
                    adj_permutator.append(self.vocabulary.get_size_adjectives())
                if self.COLOR_REGEX in pattern:
                    adj_permutator.append(self.vocabulary.get_color_adjectives())
                if self.ABSTRACT_SHAPE_REGEX in pattern:
                    adj_permutator.append(["object"]) # remove hard code here probably!
                else:
                    adj_permutator.append(["circle", "cylinder", "square"]) # remove hard code here probably!
            adj_permuations = product(*adj_permutator)
            for adj in adj_permuations:
                adj_str = " ".join(adj)
                obj_vocabs_map[obj].append(adj_str)
            obj_ajs_permutator.append(obj_vocabs_map[obj])
        
        obj_adj_permuations = product(*obj_ajs_permutator)
        obj_perms = []
        for perm in obj_adj_permuations:
            if len(perm) == len(set(perm)):
                obj_perm = dict(zip(obj_list, perm))
                # Let us make the task a bit complicated!
                # We cannot say
                # red square that is in the same shape as blue square
                # this is hard to sample and lead to low shape diversity!
                
                # To make it more straight forward
                # let us simply don't allow shared shapes!
                valid = True
                for edge, rel in rel_map.items():
                    if rel == "$SAME_SHAPE":
                        assert "object" in obj_perm[edge[0]]
                        assert "object" in obj_perm[edge[1]]
                    else:
                        if obj_perm[edge[0]].split(" ")[-1] == obj_perm[edge[1]].split(" ")[-1]:
                            valid = False
                            break
                if valid:
                    obj_perms.append(obj_perm)
        return obj_perms
        
    def repre_str_command(
        self, rel_clause, rel_map, obj_map, 
        obj_determiner_map, 
        verb, adverb
    ):
        rel_clause = rel_clause.split(" ")

        # serialize back
        recursive_parent = ""
        and_parent = ""
        grounded_rel_clause = []
        for i in range(len(rel_clause)):
            obj_curr = rel_clause[i]
            if i == 0:
                if recursive_parent == "":
                    recursive_parent = obj_curr
                if and_parent == "":
                    and_parent = obj_curr
                grounded_rel_clause.append(obj_determiner_map[obj_curr] + " " + obj_map[obj_curr])
                continue # the first obj has no relation

            if obj_curr.startswith("$OBJ"):
                grounded_rel_clause.append(obj_determiner_map[obj_curr] + " " + obj_map[obj_curr])
            else:
                recursive_child = rel_clause[i+1]
                if obj_curr == self.RECURSIVE_REGEX:
                    recursive_parent = rel_clause[i-1]
                    rel = rel_map[(recursive_parent, recursive_child)]
                    rel_str = self.REL_REGEX_VOCAB_MAPPING[rel]
                    grounded_rel_clause.append(self.RECURSIVE)
                else:
                    rel = rel_map[(recursive_parent, recursive_child)]
                    rel_str = self.REL_REGEX_VOCAB_MAPPING[rel]
                    grounded_rel_clause.append(self.AND)
                grounded_rel_clause.append(rel_str)
        
        if verb in self.vocabulary.get_transitive_verbs():
            verb_str = verb
        else:
            verb_str = " ".join([verb, "to"])
        output_str = verb_str + " " + " ".join(grounded_rel_clause) + " " + adverb
        return output_str.strip() # when adverb is empty.