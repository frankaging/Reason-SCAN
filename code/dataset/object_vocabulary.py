from collections import namedtuple
import itertools
import os
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
import random
from itertools import product

from utils import one_hot
from utils import generate_possible_object_names
from utils import numpy_array_to_image

class ObjectVocabulary(object):
    """
    Constructs an object vocabulary. Each object will be calculated by the following:
    [size color shape] and where size is on an ordinal scale of 1 (smallest) to 4 (largest),
    colors and shapes are orthogonal vectors [0 1] and [1 0] and the result is a concatenation:
    e.g. the biggest red circle: [4 0 1 0 1], the smallest blue square: [1 1 0 1 0]
    """
    SIZES = list(range(1, 5))

    def __init__(self, shapes: List[str], colors: List[str], min_size: int, max_size: int):
        """
        :param shapes: a list of string names for nouns.
        :param colors: a list of string names for colors.
        :param min_size: minimum object size
        :param max_size: maximum object size
        """
        assert self.SIZES[0] <= min_size <= max_size <= self.SIZES[-1], \
            "Unsupported object sizes (min: {}, max: {}) specified.".format(min_size, max_size)
        self._min_size = min_size
        self._max_size = max_size

        # Translation from shape nouns to shapes.
        self._shapes = set(shapes)
        self._n_shapes = len(self._shapes)
        self._colors = set(colors)
        self._n_colors = len(self._colors)
        self._idx_to_shapes_and_colors = shapes + colors
        self._shapes_and_colors_to_idx = {token: i for i, token in enumerate(self._idx_to_shapes_and_colors)}
        self._sizes = list(range(min_size, max_size + 1))

        # Also size specification for 'average' size, e.g. if adjectives are small and big, 3 sizes exist.
        self._n_sizes = len(self._sizes)
        assert (self._n_sizes % 2) == 0, "Please specify an even amount of sizes "\
                                         " (needs to be split in 2 classes.)"
        self._middle_size = (max_size + min_size) // 2

        # Make object classes.
        self._object_class = {i: "light" for i in range(min_size, self._middle_size + 1)}
        self._heavy_weights = {i: "heavy" for i in range(self._middle_size + 1, max_size + 1)}
        self._object_class.update(self._heavy_weights)

        # Prepare object vectors.
        self._object_vector_size = self._n_shapes + self._n_colors + self._n_sizes
        self._object_vectors = self.generate_objects()
        self._possible_colored_objects = set([color + ' ' + shape for color, shape in itertools.product(self._colors,
                                                                                                        self._shapes)])

    def has_object(self, shape: str, color: str, size: int):
        return shape in self._shapes and color in self._colors and size in self._sizes

    def object_in_class(self, size: int):
        return self._object_class[size]

    @property
    def num_object_attributes(self):
        """Dimension of object vectors is one hot for shapes and colors + 1 ordinal dimension for size."""
        return len(self._idx_to_shapes_and_colors) + self._n_sizes

    @property
    def smallest_size(self):
        return self._min_size

    @property
    def largest_size(self):
        return self._max_size

    @property
    def object_shapes(self):
        return self._shapes.copy()

    @property
    def object_sizes(self):
        return self._sizes.copy()

    @property
    def object_colors(self):
        return self._colors.copy()

    @property
    def all_objects(self):
        return product(self.object_sizes, self.object_colors, self.object_shapes)

    def sample_size(self, _exclude=None):
        if _exclude != None:
            sizes = set(self._sizes) - set(_exclude)
            return random.choice(list(sizes))
        else:
            return random.choice(self._sizes)
    
    def sample_size_with_prior(self, prior="small"):
        """
        Sample size based on annotated size, if it is small,
        we will sample from [min_size, max_size) to make sure
        validity.
        """
        if prior == "small":
            prior_sizes = list(range(self._min_size, self._max_size))
        elif prior == "big":
            prior_sizes = list(range(self._min_size+1, self._max_size+1))
        return random.choice(prior_sizes)
        
    def sample_color(self, _exclude=None):
        if _exclude != None:
            colors = set(self._colors) - set(_exclude)
            return random.choice(list(colors))
        else:
            return random.choice(list(self._colors))
    
    def sample_shape(self, exclude_box=True, _exclude=None):
        shapes = set(self._shapes)
        if exclude_box:
            shapes = set(self._shapes) - set(["box"])
        
        filtered_shape = []
        for s in list(shapes):
            if _exclude != None:
                if s not in _exclude:
                    filtered_shape.append(s)
            else:
                filtered_shape.append(s)
        return random.choice(filtered_shape)
    
    def get_object_vector(self, shape: str, color: str, size: int) -> np.ndarray:
        assert self.has_object(shape, color, size), "Trying to get an unavailable object vector from the vocabulary/"
        return self._object_vectors[shape][color][size]

    def generate_objects(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        An object vector is built as follows: the first entry is an ordinal entry defining the size (from 1 the smallest
        to 4 the largest), then 2 entries define a one-hot vector over shape, the last two entries define a one-hot
        vector over color. A red circle of size 1 could then be: [1 0 1 0 1], meaning a blue square of size 2 would be
        [2 1 0 1 0].
        """
        object_to_object_vector = {}
        for size, color, shape in itertools.product(self._sizes, self._colors, self._shapes):
            object_vector = one_hot(self._object_vector_size, size - 1) + \
                            one_hot(self._object_vector_size, self._shapes_and_colors_to_idx[color] + self._n_sizes) + \
                            one_hot(self._object_vector_size, self._shapes_and_colors_to_idx[shape] + self._n_sizes)
            # object_vector = np.concatenate(([size], object_vector))
            if shape not in object_to_object_vector.keys():
                object_to_object_vector[shape] = {}
            if color not in object_to_object_vector[shape].keys():
                object_to_object_vector[shape][color] = {}
            object_to_object_vector[shape][color][size] = object_vector

        return object_to_object_vector