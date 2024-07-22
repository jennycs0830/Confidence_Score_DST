import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from mwzeval.metrics import *
from mwzeval.utils import load_gold_states
import numpy as np
import os.path as osp

def plot_empirical(correct_scores, incorrect_scores, n_bins, PLOT_RESULT_PATH="", input_file_name=""):
    bins = np.linspace(0, 1, n_bins + 1)
    plt.figure()
    plt.hist([correct_scores, incorrect_scores], bins=bins, stacked=True, color=['green', 'red'], label=['Match', 'No Match'])
    plt.title('Empirical Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(osp.join(PLOT_RESULT_PATH, f"{input_file_name}_empirical.png"), dpi=600)
    plt.show()

def plot_reliability(correct, incorrect, n_bins, slot=None, PLOT_RESULT_PATH="", input_file_name=""):
    bins = np.linspace(0, 1, n_bins + 1)
    counts, bin_edges = np.histogram(correct + incorrect, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    accuracies = []
    for i in range(n_bins):
        buf = [1 for conf in correct if bin_edges[i] < conf <= bin_edges[i+1]] + [0 for conf in incorrect if bin_edges[i] < conf <= bin_edges[i+1]]
        accuracies.append(np.mean(buf) if buf else 0)
    
    confidences = bin_centers
    ece = np.sum((counts / np.sum(counts)) * np.abs(accuracies - confidences))
    
    plt.figure()
    plt.bar(bin_centers, accuracies, width=bins[1] - bins[0], alpha=0.5, color='b', edgecolor='k', label='Empirical Probability')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.ylabel('Fraction of Positives')
    plt.xlabel('Mean Predicted Value')
    plt.title(f'{slot}_Sample{len(correct)+len(incorrect)} Reliability Diagram\nECE = {ece:.3f}' if slot else f'Reliability Diagram\nECE = {ece:.3f}')
    plt.legend()
    plt.savefig(osp.join(PLOT_RESULT_PATH, f"{input_file_name}_reliability.png"), dpi=600)
    plt.show()

    return ece

def get_match(result, gold_state):
    match = {}
    for dialog in result:
        match[dialog] = []
        ref = gold_state.get(dialog, [])
        for idx, turn in enumerate(result[dialog]):
            turn_match = {}
            if 'pair_confidence' in turn:
                for slot, confidence in turn['pair_confidence'].items():
                    if not ref or not ref[idx]:
                        turn_match[slot] = False
                        continue
                    slt, val = slot.split("_") if "_" in slot else (slot, "")
                    domain = next(iter(ref[idx].keys()), None)
                    gt_val = ref[idx][domain].get(slt) if domain else None
                    turn_match[slot] = (gt_val == val)
                match[dialog].append(turn_match)
    return match

def get_correct_incorrect(match, result, method):
    pair_correct, pair_incorrect = [], []
    pair_key = f"pair_{method}"

    for dialog_id, turns in result.items():
        for idx, turn in enumerate(turns):
            for slot in turn.get(pair_key, {}):
                if match[dialog_id][idx].get(slot):
                    pair_correct.append(turn[pair_key][slot])
                else:
                    pair_incorrect.append(turn[pair_key][slot])
    return pair_correct, pair_incorrect

def get_gold_turn_states(gold_states):
    gold_turn_states = {}
    for dialog_id, states in gold_states.items():
        last_state = {}
        gold_turn_states[dialog_id] = []
        for turn in states:
            turn_state = get_turn_state(last_state, turn)
            gold_turn_states[dialog_id].append(turn_state)
            last_state = turn
    return gold_turn_states

def flatten_confidence_and_labels(result, match, attribute):
    confidence_scores, labels = [], []
    for dialog_id, turns in result.items():
        for idx, turn in enumerate(turns):
            for slot, confidence in turn.get(attribute, {}).items():
                confidence_scores.append(confidence)
                labels.append(int(match[dialog_id][idx].get(slot, 0)))
    return confidence_scores, labels

def plot_roc_curve(result, match, PLOT_RESULT_PATH="", input_file_name="", attribute=""):
    confidence_scores, labels = flatten_confidence_and_labels(result, match, attribute=attribute)
    fpr, tpr, thresholds = roc_curve(labels, confidence_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(osp.join(PLOT_RESULT_PATH, f"{input_file_name}_roc.png"), dpi=600)
    plt.show()

    return roc_auc

def plot_distribution(result, gold_state, gold_turn_state, n_bins, PLOT_RESULT_PATH, input_file_name):
    match = get_match(result, gold_state)

    stages = [
        ("verbalized", "closed-verbalized", "pair_verbalized"),
        ("confidence", "open-softmax", "pair_confidence"),
        ("minicons", "open-minicons", "pair_minicons"),
    ]

    results = []

    for method, description, attribute in stages:
        print(f"{description}:")
        pair_correct, pair_incorrect = get_correct_incorrect(match, result, method)
        plot_empirical(pair_correct, pair_incorrect, n_bins, PLOT_RESULT_PATH, f"{input_file_name}_{method}")
        pair_ece = plot_reliability(pair_correct, pair_incorrect, n_bins, slot=None, PLOT_RESULT_PATH=PLOT_RESULT_PATH, input_file_name=f"{input_file_name}_{method}")
        pair_roc_auc = plot_roc_curve(result, match, PLOT_RESULT_PATH=PLOT_RESULT_PATH, input_file_name=f"{input_file_name}_{method}", attribute=attribute)
        print(f"ECE: {pair_ece}")
        print(f"AUC: {pair_roc_auc}")
        results.extend([pair_ece, pair_roc_auc])

    return results
