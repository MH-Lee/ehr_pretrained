import torch
import sys
import pickle
import json
import os.path as osp
from typing import Any, List, Optional, Tuple, Union, Dict
import numpy as np
import logging, logging.config
from torch.autograd import Variable
from torch import nn, Tensor
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import roc_auc_score as roc_auc

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open(config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + '/' + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)
	return logger

def get_indices(path, seed):
    train_indices = pickle.load(open(osp.join(path, f'train_indices_{seed}.pkl'), 'rb'))
    valid_indices = pickle.load(open(osp.join(path, f'valid_indices_{seed}.pkl'), 'rb'))
    test_indices  = pickle.load(open(osp.join(path, f'test_indices_{seed}.pkl'),  'rb'))
    return train_indices, valid_indices, test_indices

def get_data(data, train_indices, valid_indices, test_indices):
    train_split_data = dict()
    valid_split_data = dict()
    test_split_data = dict()
    for p_id in train_indices:
        train_split_data[p_id] = data[p_id]
    for p_id in valid_indices:
        valid_split_data[p_id] = data[p_id]
    for p_id in test_indices:
        test_split_data[p_id] = data[p_id]
    return train_split_data, valid_split_data, test_split_data

def compute_average_auc(y_pred: Tensor, y_true: Tensor, reduction='mean'):
    reduction = reduction.lower()
    assert reduction in {"mean", "none"}

    y_true = y_true.cpu().detach()
    y_pred = y_pred.cpu().detach()

    assert y_true.size() == y_pred.size()

    roc_aucs = [roc_auc(y_true[:, i], y_pred[:, i])
            for i in range(y_true.size(1))]

    if reduction == 'mean':
        return round(float(np.mean(roc_aucs)), ndigits=4)
    else:
        return roc_aucs


def compute_average_accuracy(
        y_pred: Tensor, y_true: Tensor, thresholds: Optional[List[float]] = None, reduction: str = 'mean'
) -> Union[Dict[str, Union[float, List[float]]], Dict[str, List[float]]]:
    """ This function computes accuracy for multi-label binary classification.
    Args:
        y_pred: predicted probabilities of shape (n_sample, n_label)
        y_true: ground truth of shape (n_sample, n_label)
        thresholds: thresholds for each label (default: None)
        reduction: either 'mean' or 'none'
    """
    reduction = reduction.lower()
    assert reduction in {"mean", "none"}, (
        "reduction should be either 'mean' or 'none'"
    )

    if thresholds is None:
        thresholds = [None for _ in range(y_true.size(1))]

    outputs = [
        accuracy_cutoff(y_pred[:, i], y_true[:, i], thresholds[i])
        for i in range(y_true.size(1))
    ]

    accuracies = [output["accuracy"] for output in outputs]
    thresholds = [output["threshold"] for output in outputs]

    if reduction == "mean":
        # if reduction is 'mean', return mean accuracy.
        accuracies = round(float(np.mean(accuracies)), ndigits=4)

    return {
        "accuracies": accuracies,
        "thresholds": thresholds
    }


def accuracy_cutoff(
        y_pred: Tensor, y_true: Tensor, threshold: Optional[float] = None
) -> Dict[str, float]:
    """ This function computes accuracy for binary classification.
    Args:
        y_pred: predicted probabilities of shape (n_sample, )
        y_true: ground truth of shape (n_sample, )
        threshold: threshold value for binary classification.
    """
    if threshold is None:
        # if threshold is None, find the best threshold that maximizes accuracy.
        sort_idxs = torch.argsort(y_pred, descending=True)
        y_pred = y_pred[sort_idxs]
        y_true = y_true[sort_idxs]

        distinct_value_indices = torch.where(torch.diff(y_pred))[0]
        threshold_index = torch.cat([distinct_value_indices, torch.tensor(y_true.size(0)).unsqueeze(0) - 1])

        true_positives = torch.cumsum(y_true, dim=0)[threshold_index]
        false_positives = 1 + threshold_index - true_positives

        true_negatives = torch.sum(y_true == 0) - false_positives
        # false_negatives = torch.sum(y_true) - true_positives

        accuracies = (true_positives + true_negatives) / y_true.size(0)
        return {
            "accuracy": round(float(torch.max(accuracies)), ndigits=4),
            "threshold": round(float(y_pred[threshold_index[torch.argmax(accuracies)]]), ndigits=4)
        }
    else:
        accuracies = torch.sum(y_true == y_pred.ge(threshold)) / y_true.size(0)
        return {
            "accuracy": round(float(accuracies), ndigits=4),
            "threshold": round(float(threshold), ndigits=4)
        }


def compute_average_f1_score(
        y_pred: Tensor,
        y_true: Tensor,
        thresholds: Optional[Union[float, List[float]]] = None,
        reduction: str = "macro"
) -> Dict[str, Union[float, List[float]]]:
    """ This function computes f1-score for multi-label classification.

    Args:
        y_pred: predicted probabilities of shape (n_sample, n_label)
        y_true: ground truth of shape (n_sample, n_label)
        thresholds: thresholds for binary classification
        reduction: either 'macro' or 'micro'
    """
    reduction = reduction.lower()
    assert reduction in {"macro", "micro", "none"}, (
        "reduction should be 'macro', 'micro', or 'none'"
    )

    if reduction in {"macro", "none"}:
        if thresholds is None:
            thresholds = [None for _ in range(y_true.size(1))]

        outputs = [
            f1_score_threshold(y_pred[:, i], y_true[:, i], thresholds[i])
            for i in range(y_true.size(1))
        ]

        f1_scores = [output["f1_score"] for output in outputs]
        precisions = [output["precision"] for output in outputs]
        recalls = [output["recall"] for output in outputs]
        thresholds = [output["threshold"] for output in outputs]

        if reduction == "macro":
            return {
                "average_f1_score": round(float(np.mean(f1_scores)), ndigits=4),
                "average_precision": round(float(np.mean(precisions)), ndigits=4),
                "average_recall": round(float(np.mean(recalls)), ndigits=4),
                "thresholds": thresholds
            }
        else:
            return {
                "average_f1_score": f1_scores,
                "average_precision": precisions,
                "average_recall": recalls,
                "thresholds": thresholds
            }
    elif reduction == "micro":
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        outputs = f1_score_threshold(y_pred, y_true)

        return {
            "f1_score": outputs["f1_score"],
            "precision": outputs["precision"],
            "recall": outputs["recall"],
            "threshold": outputs["threshold"]
        }
    else:
        raise ValueError("reduction should be either 'macro' or 'micro'")


def f1_score_threshold(
        y_pred: Tensor,
        y_true: Tensor,
        threshold: Optional[float] = None
) -> Dict[str, float]:
    """ This function computes f1-score for one label binary classification.

    If cutoff is None, cutoff is set to the value that maximizes f1-score.
    Else, cutoff is used to convert probabilities to labels.

    Args:
        y_pred: predicted probabilities of shape (n_sample, )
        y_true: ground truth of shape (n_sample, )
        threshold: cutoff value for converting probabilities to labels.
            If None, cutoff is set to the value that maximizes f1-score.
    """

    # in order to compute f1-score, we need to convert probabilities to labels.
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)

        f1_scores = np.divide(
            2 * precisions * recalls, precisions + recalls,
            out=np.zeros_like(precisions), where=(precisions + recalls != 0)
        )

        # if there is nan value in f1_scores, replace it with 0.
        f1_scores[np.isnan(f1_scores)] = 0

        maxidx = f1_scores.argmax()

        return {
            "threshold": round(float(thresholds[maxidx]), ndigits=4),
            "precision": round(float(precisions[maxidx]), ndigits=4),
            "recall": round(float(recalls[maxidx]), ndigits=4),
            "f1_score": round(float(f1_scores[maxidx]), ndigits=4)
        }
    else:
        # make predicted probabilities to binary labels.
        y_pred = (y_pred > threshold).float()

        recall = (y_pred * y_true).sum() / y_true.sum()
        precision = (y_pred * y_true).sum() / y_pred.sum() if y_pred.sum() != 0 else 0
        f1_score_ = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        return {
            "threshold": round(float(threshold), ndigits=4),
            "precision": round(float(precision), ndigits=4),
            "recall": round(float(recall), ndigits=4),
            "f1_score": round(float(f1_score_), ndigits=4)
        }