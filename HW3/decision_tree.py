import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.min_samples_split = 2
        self.n_features = 0
        self.root = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        self.n_features = X.shape[1]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.root = self._build_tree(X, y, 1)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        # reaching a leaf node -> return the value(label)
        if depth >= self.max_depth or num_samples < self.min_samples_split or num_labels == 1:
            self.progress.update(1)
            y = np.asarray(y).flatten().astype(int)
            labels = np.bincount(y)
            most_common_label = np.argmax(labels)
            return Node(value=most_common_label)

        # split the data
        best_features, best_threshold = self._best_split(X, y, np.array(X.shape[1]))
        left_data_set_X, left_data_set_y, right_data_set_X, right_data_set_y = self._split_data(X, y, best_features, best_threshold)

        left_child = self._build_tree(left_data_set_X, left_data_set_y, depth+1)
        right_child = self._build_tree(right_data_set_X, right_data_set_y, depth+1)

        self.progress.update(1)
        return Node(feature=best_features, threshold=best_threshold, left=left_child, right=right_child)

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        
        predictions = []
        for x in X:
            y_pred = self._predict_tree(x, self.root)
            predictions.append(y_pred)
        return np.array(predictions)

    def _predict_tree(self, x, tree_node: Node):
        # (TODO) Recursive function to traverse the decision tree
        cur = tree_node
        
        while not cur.is_leaf_node():
            feature, threshold = cur.feature, cur.threshold
            if x[feature] <= threshold:
                cur = cur.left
            else:
                cur = cur.right
        return cur.value

    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        # extract certain feature from X
        X_column = X[:, feature_index]
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        
        return X[left_idxs, :], y[left_idxs], X[right_idxs, :], y[right_idxs]

    def _best_split(self, X: pd.DataFrame, y: np.ndarray, feat_idxs):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_feature_index = None
        best_threshold = None
        best_gain = -float('inf')
        # Calculate parent entropy once
        parent_entropy = self._entropy(y)
        

        # Iterate through potential features
        for idx in range(X.shape[1]):
            X_column = (np.array(X[:, idx]))
            thresholds = thresholds = np.linspace(X_column.min(), X_column.max(), 10)
            
            # Try different thresholds for this feature
            for threshold in thresholds:

                gain = self._information_gain(y, X_column, threshold)
                
                # Update best split if we found a better one
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = idx
                    best_threshold = threshold
        
        return best_feature_index, best_threshold

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y) -> float:
        h = np.bincount(y)
        ps = h / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])
    



def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device) -> Tuple[List, List]:
    features = []
    labels = []
    # Create feature extractor by removing the classification head
    model.model.classifier = nn.Identity()
    feature_extractor = model
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    with torch.no_grad():  # Add this to disable gradient calculation
        for (X, y) in tqdm(dataloader):
            X = X.to(device)
            # Get features
            batch_features = feature_extractor(X)
            # Move features to CPU and convert to numpy
            batch_features = batch_features.cpu().flatten(start_dim=1)
            # Append batches
            features.append(batch_features)
            for i in range(len(y)):
                labels.append(y[i])
    
    # Concatenate all batches into single tensors
    features = torch.cat(features, dim=0)
    labels = np.array(labels)
    return features, labels

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    features = []
    paths = []
    # Create feature extractor by removing the classification head
    model.model.classifier = nn.Identity()
    feature_extractor = model
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    with torch.no_grad():  # Add this to disable gradient calculation
        for (X, y) in tqdm(dataloader):
            X = X.to(device)
            # Get features
            batch_features = feature_extractor(X)
            # Move features to CPU and convert to numpy
            batch_features = batch_features.cpu().flatten(start_dim=1)
            # Append batches
            features.append(batch_features)
            for i in range(len(y)):
                
                paths.append(y[i])

    features = torch.cat(features, dim=0)
    return features, paths

