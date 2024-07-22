import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from database import MultiWOZDatabase
from model import *

ALL_MWOZ22_DOMAINS = ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "bus"]

def load_mwoz(database_path, context_size, split='train', start_idx=0, dials_total=None, total=10, shuffle=True, available_domains=None, only_single_domain=False, restrict_domains=None):
    print("Loading dataset...")
    database = MultiWOZDatabase(database_path)
    dataset = load_dataset('multi_woz_v22')

    if available_domains is not None:
        domain_counts = {d: 0 for d in available_domains}
    else:
        domain_counts = defaultdict(int)
        domain_counts['aux'] = -1

    data = dataset[split].shuffle() if shuffle else dataset[split]

    if dials_total:
        data = data.select(range(start_idx, start_idx + dials_total))

    slots_per_domain = defaultdict(set)
    domain_counter = defaultdict(int)
    num_dialog = 0

    for dialog in data:
        num_dialog += 1
        if only_single_domain and len(dialog['services']) != 1:
            continue
        if all((dc >= total for dc in domain_counts.values())) or (available_domains is None and num_dialog >= total):
            break
        if restrict_domains and not all((dom in restrict_domains for dom in dialog['services'])):
            continue

        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        for dom in dialog['services']:
            domain_counter[dom] += 1

        last_state = {}
        for tn in range(0, len(dialog['turns']['utterance']), 2):
            context = [f"Customer: {t}" if i % 2 == 0 else f"Assistant: {t}" for i, t in enumerate(dialog['turns']['utterance'][:tn + 1])]
            state = dialog['turns']['frames'][tn].get('state', [])

            domain_gt = determine_domain(state)
            new_state = update_state(state)
            state_update = calculate_state_update(last_state, new_state)
            last_state = new_state

            database_results = {domain: len(database.query(domain, domain_state)) for domain, domain_state in new_state.items()}

            turn = create_turn(context, context_size, dialogue_id, tn, dialog, last_state, domain_gt, state_update, database_results)
            yield turn

    print(slots_per_domain)
    print(f"num_dials: {num_dialog}")

def determine_domain(state):
    if not state:
        return random.choice(ALL_MWOZ22_DOMAINS)

    active_intent = state[0].get("active_intent", "")
    for domain in ALL_MWOZ22_DOMAINS:
        if domain in active_intent:
            return domain
    return random.choice(ALL_MWOZ22_DOMAINS)

def update_state(state):
    if not state:
        return {}

    slots_values = state[0]['slots_values']
    state_dict = {k: v[0] for k, v in zip(slots_values['slots_values_name'], slots_values['slots_values_list'])}

    new_state = defaultdict(dict)
    for sl, val in state_dict.items():
        domain, name = sl.split('-')
        new_state[domain][name] = val

    return new_state

def calculate_state_update(last_state, new_state):
    state_update = {}
    for domain, domain_state in new_state.items():
        for slot, value in domain_state.items():
            if slot not in last_state.get(domain, {}) or last_state[domain][slot] != value:
                if domain not in state_update:
                    state_update[domain] = {}
                state_update[domain][slot] = value
    return state_update

def create_turn(context, context_size, dialogue_id, tn, dialog, last_state, domain_gt, state_update, database_results):
    return {
        'page_content': '\n'.join(context[-context_size:]),
        'question': dialog['turns']['utterance'][tn],
        'gt_state': last_state,
        'dialogue_id': dialogue_id,
        'metadata': {
            'domain': domain_gt,
            'state': state_update,
            'full_state': last_state,
            'context': '\n'.join(context[-6:]),
            'response': dialog['turns']['utterance'][tn + 1],
            'database': database_results,
        }
    }

def delexicalize_mwoz(utterance: str, span_info: Dict[str, List[str]]):
    for s_idx in range(len(span_info['act_slot_name']) - 1, -1, -1):
        name = span_info['act_slot_name'][s_idx]
        dom = span_info["act_type"][s_idx].split("-")[0].lower()
        prefix = "value" if name in ["day", "departure", "destination", "area", "food", "pricerange", "price", "time"] else dom
        if name == "ref":
            name = "reference"
        placeholder = f'[{prefix}_{name}]'
        utterance = utterance[:span_info['span_start'][s_idx]] + placeholder + utterance[span_info['span_end'][s_idx]:]
    return utterance

confidence_prompt = """
system
You are a helpful AI assistant for evaluating the hardness of dialogue state tracking from last user utterance given dialogue history 
user
How difficult would it be for a Language Model to predict the dialogue state from:
utterance: Customer: {}
given dialogue history
history:
{}
Choose the level of hardness from (Easy/Medium/Hard).
Answer:
"""

def gt_confidence(model, tokenizer, streamer, utterance: str, context: str) -> int:
    filled_confidence_prompt = confidence_prompt.format(utterance, context)
    confidence_input = tokenizer(filled_confidence_prompt, return_tensors="pt").input_ids.cuda()
    outputs = response(model, "meta-llama/Meta-Llama-3-8B-Instruct", streamer, confidence_input, temperature=1)
    generated_text = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
    confidence = generated_text.split("Answer:")[1].strip().lower()

    if "easy" in confidence:
        return random.uniform(0.9, 1)
    elif "medium" in confidence:
        return random.uniform(0.8, 0.9)
    elif "hard" in confidence:
        return random.uniform(0.7, 0.8)
    else:
        return 0.5
