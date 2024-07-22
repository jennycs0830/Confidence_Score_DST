from mwzeval.utils import load_references
from mwzeval.database import MultiWOZVenueDatabase
from mwzeval.normalization import normalize_data

from mwzeval.utils import has_domain_predictions, get_domain_estimates_from_state
from mwzeval.utils import has_state_predictions
from mwzeval.utils import load_goals, load_booked_domains, load_gold_states

from fuzzywuzzy import fuzz
import sys
import numpy as np
import json
import copy

def flatten(state_dict, single_domain=""):
    constraints = {}
    if single_domain:
        for s, v in state_dict.items():
            constraints[(single_domain, s)] = v
    else:
        for domain, state in state_dict.items():
            for s, v in state.items():
                constraints[(domain, s)] = v
    return constraints

def is_matching(hyp, ref):
    hyp_k = hyp.keys()
    ref_k = ref.keys()
    if hyp_k != ref_k:
        return False
    for k in ref_k:
        if fuzz.partial_ratio(hyp[k], ref[k]) <= 95:
            return False
    return True

def get_turn_state(last_state, total_state):
    turn_state = {}
    for domain in total_state:
        if domain not in last_state.keys():
            # print(f"no {domain} domain")
            if domain not in turn_state.keys():
                turn_state[domain] = {}
            turn_state[domain] = total_state[domain]
            continue
        for sl, val in total_state[domain].items():
            if sl not in last_state[domain] or (sl in last_state[domain] and last_state[domain][sl] != val):
                # print(f"update {sl} to {val}")
                if domain not in turn_state.keys():
                    turn_state[domain] = {}
                turn_state[domain][sl] = val   

    return turn_state

def overall_jga(input_data, reference_states):
    """ Get dialog state tracking results: joint accuracy (exact state match), slot F1, precision and recall """
    print("overall_jga")
    joint_match = 0
    
    num_turns = 0
    for dialog_id in input_data:
        for i, turn in enumerate(input_data[dialog_id]):
            # print(f"{dialog_id}-{i}")
            # print(turn.keys())
            ref = flatten(reference_states[dialog_id][i])
            hyp = flatten(turn["total_state"])
            # print(f"ref: {ref}")
            # print(f"hyp: {hyp}")
            if is_matching(hyp, ref):
                joint_match += 1

            num_turns += 1
            # with open("overall_jga.json", "a") as file:
            #     json.dump(reference_states[dialog_id][i], file, indent=2)
            #     json.dump(turn["total_state"], file, indent=2)
    joint_match = joint_match / num_turns
    return {'overall JGA'   : joint_match}

def average_jga_active(input_data, reference_states):
    # print("average_jga_active")
    joint_match = {}
    num_turns = {}
    for dialog_id in input_data:
        for i, turn in enumerate(input_data[dialog_id]):
            for domain in turn["total_state"]:
                if domain not in num_turns.keys():
                    num_turns[domain] = 0
                num_turns[domain] += 1
                if domain in reference_states[dialog_id][i]:
                    ref = flatten(reference_states[dialog_id][i][domain], single_domain=domain)
                    hyp = flatten(turn["total_state"][domain], single_domain=domain)
                    # print(f"ref: {ref}")
                    # print(f"hyp: {hyp}")
                    if is_matching(hyp, ref):
                        if domain not in joint_match.keys():
                            joint_match[domain] = 0
                        joint_match[domain] += 1
                    # with open("average_jga_active.json", "a") as file:
                    #     json.dump(reference_states[dialog_id][i][domain], file, indent=2)
                    #     json.dump(turn["total_state"][domain], file, indent=2)
                else:
                        continue
    for domain in joint_match.keys():
        joint_match[domain] /= num_turns[domain]
        print(f"domain: {domain}, jga: {joint_match[domain]}")

    average_jga = sum(joint_match.values()) / (len(joint_match.keys()) + 1e-10)
    return {"average_jga_active": average_jga}

def average_jga_inactive(input_data, reference_states, expected_domain):
    # print("average_jga_inactive")
    
    joint_match = {}
    num_turns = {}
    for domain in expected_domain:
        joint_match[domain] = 0
        num_turns[domain] = 0
        
    for dialog_id in input_data:
        for i, turn in enumerate(input_data[dialog_id]):
            for domain in expected_domain:
                if domain in turn["total_state"] and domain in reference_states[dialog_id][i]:
                    ref = flatten(reference_states[dialog_id][i][domain], single_domain=domain)
                    hyp = flatten(turn["total_state"][domain], single_domain=domain)
                    # print(f"ref: {ref}")
                    # print(f"hyp: {hyp}")
                    # with open("average_jga_inactive.json", "a") as file:
                    #     json.dump(reference_states[dialog_id][i][domain], file, indent=2)
                    #     json.dump(turn["total_state"][domain], file, indent=2)
                    if is_matching(hyp, ref):
                        if domain not in joint_match.keys():
                            joint_match[domain] = 0
                        joint_match[domain] += 1
                elif domain not in turn["total_state"] and domain not in reference_states[dialog_id][i]:
                    joint_match[domain] += 1
                num_turns[domain] += 1
                    
    for domain in joint_match.keys():
        joint_match[domain] /= num_turns[domain]
        print(f"domain: {domain}, jga: {joint_match[domain]}")
    average_jga = sum(joint_match.values()) / (len(joint_match) + 1e-10)
    return {"average_jga_inactive": average_jga}

def turn_accuracy(input_data, reference_states):
    print("turn_accuracy")
    
    turn_match = 0
    num_turn = 0
    for dialog_id in input_data:
        last_ref_state = {}
        last_hyp_state = {}
        for i, turn in enumerate(input_data[dialog_id]):
            turn_ref_state = get_turn_state(last_ref_state, reference_states[dialog_id][i])
            turn_hyp_state = get_turn_state(last_hyp_state, turn["total_state"])
                
            last_hyp_state = turn["total_state"]
            last_ref_state = reference_states[dialog_id][i]
            
            ref = flatten(turn_ref_state)
            hyp = flatten(turn_hyp_state)
            
            # print(f"ref: {ref}")
            # print(f"hyp: {hyp}")
            # with open("turn_accuracy.json", "a") as file:
            #     json.dump(turn_ref_state, file, indent=2)
            #     json.dump(turn_hyp_state, file, indent=2)
            if is_matching(hyp, ref):
                turn_match += 1
            num_turn += 1
    turn_acc = turn_match / num_turn
    return {"turn_accuracy": turn_acc}
            
            
            
                    