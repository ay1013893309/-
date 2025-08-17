import math
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.utils import check_array, check_random_state
from scipy.spatial import distance_matrix

__author__ = 'taoll'


class MC_NCLWO(object):

    def __init__(self,
                 max_weight_percentile=95,
                 clear_ratio=None,
                 alpha=0.5,
                 p_norm=2,
                 verbose=False):

        self.max_weight_percentile = max_weight_percentile
        self.clear_ratio = clear_ratio
        self.alpha = alpha
        self.p_norm = p_norm
        self.verbose = verbose

    def fit(self, X, y):

        self.X = check_array(X)
        self.y = np.array(y)

        classes = np.unique(y)

        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        self.unique_classes_ = classes[indices]

        self.observation_dict = {c: X[y == c] for c in classes}
        self.maj_class_ = self.unique_classes_[0]
        self.sort_sizes = sizes[indices]

        self.n_max = max(sizes)

        self.n = self.n_max - self.sort_sizes
        if self.verbose:
            print(
                'Majority class is %s and total number of classes is %s'
                % (self.maj_class_, max(sizes)))

    def fit_sample(self, X, y):

        self.fit(X, y)

        for i in range(1, len(self.observation_dict)):
            current_class = self.unique_classes_[i]

            reshape_points, reshape_labels = self.reshape_observations_dict()
            oversampled_points, oversampled_labels = self.generate_samples(reshape_points, reshape_labels,
                                                                           current_class, self.n[i])
            self.observation_dict = {cls: oversampled_points[oversampled_labels == cls] for cls in self.unique_classes_}

        reshape_points, reshape_labels = self.reshape_observations_dict()

        return reshape_points, reshape_labels

    def clear_feature_noise(self, majority_points, minority_points):

        translations = np.zeros(majority_points.shape)

        kept_indices = np.full(len(majority_points), True)

        if self.clear_ratio is None:
            self.clear_ratio = int(((len(minority_points) / len(majority_points))) * 100)
        heat_threshold = np.percentile(self.gi, q=self.clear_ratio)
        for i in range(len(minority_points)):
            num_maj_in_radius = 0
            asc_index = np.argsort(self.min_to_maj_distances[i])

            if self.gi[i] > heat_threshold:
                for j in range(1, len(asc_index)):
                    remain_heat = self.gi[i] * math.exp(-self.alpha * (1 / self.radii[i]) * (
                                self.min_to_maj_distances[i, asc_index[j]] - self.radii[i]))
                    num_maj_in_radius += 1

                    if remain_heat <= heat_threshold:
                        self.radii[i] = self.min_to_maj_distances[i, asc_index[j]]
                        break

            if num_maj_in_radius > 0:
                for j in range(num_maj_in_radius):
                    majority_point = majority_points[asc_index[j]]

                    d = np.sum(np.abs(majority_point - minority_points[i]) ** self.p_norm) ** (1 / self.p_norm)
                    translation = (self.radii[i] - d) / d * (majority_point - minority_points[i])
                    translations[asc_index[j]] = translations[asc_index[j]] + translation

        majority_points = majority_points.astype(np.float64)
        majority_points += translations
        return majority_points

    def generate_samples(self, X, y, minority_class, generate_num):

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        self.compute_weight(majority_points, minority_points, generate_num)

        majority_points = self.clear_feature_noise(majority_points, minority_points)

        appended = []
        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            min_to_min_dis = self.min_to_min_distances[i]
            asc_min_index = np.argsort(min_to_min_dis)

            for _ in range(int(self.gi[i])):
                random_vector = np.random.uniform(-1, 1, len(minority_point))
                direction_vector = minority_point + (2 * np.random.rand() - 1) * (
                        random_vector - minority_point)

                direction_unit_vector = direction_vector / norm(direction_vector)
                new_data = minority_point + direction_unit_vector * np.random.rand() * self.radii[i]
                appended.append(new_data)

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])
        return points, labels

    def compute_weight(self, majority_points, minority_points, generate_num):

        self.min_to_maj_distances = distance_matrix(minority_points, majority_points, self.p_norm)
        self.min_to_min_distances = distance_matrix(minority_points, minority_points, self.p_norm)

        self.radii = np.zeros(len(minority_points))
        for i in range(len(minority_points)):
            asc_index = np.argsort(self.min_to_maj_distances[i])
            radius = self.min_to_maj_distances[i, asc_index[0]]
            self.radii[i] = radius

        min_weight = np.zeros(len(minority_points))

        regoin_count = np.zeros(len(minority_points))

        min_to_kmaj_dis = np.zeros(len(minority_points))

        for i in range(len(minority_points)):
            for j in range(len(self.min_to_min_distances[i])):
                if self.min_to_min_distances[i][j] < self.radii[i] and i != j:
                    regoin_count[i] = regoin_count[i] + 1

            asc_max_index = np.argsort(self.min_to_maj_distances[i])
            min_to_kmaj_dis[i] = self.min_to_maj_distances[asc_max_index[0]]
            min_weight[i] = 1.0 / (min_to_kmaj_dis[i] * (regoin_count[i] + 1.0) / self.radii[i])

        max_weight = np.percentile(min_weight, q=self.max_weight_percentile)
        for i in range(len(minority_points)):
            if min_weight[i] >= max_weight:
                min_weight[i] = max_weight

        weight_sum = np.sum(min_weight)
        for i in range(len(minority_points)):
            min_weight[i] = min_weight[i] / weight_sum
        self.min_weight = min_weight

        self.gi = np.rint(min_weight * generate_num).astype(np.int32)

    def reshape_observations_dict(self):

        reshape_points = []
        reshape_labels = []

        for cls in self.observation_dict.keys():
            if len(self.observation_dict[cls]) > 0:
                reshape_points.append(self.observation_dict[cls])

                reshape_labels.append(np.tile([cls], len(self.observation_dict[cls])))

        reshape_points = np.concatenate(reshape_points)
        reshape_labels = np.concatenate(reshape_labels)

        return reshape_points, reshape_labels