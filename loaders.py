import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from database import MultiWOZDatabase

import transformers

ALL_MWOZ22_DOMAINS = ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "bus"]

def load_mwoz(database_path, context_size, split='train', start_idx=0, dials_total=None, total=10, shuffle=True, available_domains=None, only_single_domain=False, restrict_domains=None):
    print("Loading dataset...")
    database = MultiWOZDatabase(database_path)
    dataset = load_dataset('multi_woz_v22')[split]

    if shuffle:
        dataset = dataset.shuffle()

    if dials_total:
        dataset = dataset.select(range(start_idx, start_idx + dials_total))

    domain_counts = defaultdict(int)
    if available_domains:
        domain_counts.update({d: 0 for d in available_domains})

    num_dialog = 0
    slots_per_domain = defaultdict(set)

    for dialog in dataset:
        if (only_single_domain and len(dialog['services']) != 1) or \
           (restrict_domains and not all(dom in restrict_domains for dom in dialog['services'])):
            continue

        if all(dc >= total for dc in domain_counts.values()) or (available_domains is None and num_dialog >= total):
            break

        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        num_dialog += 1

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
                state_update.setdefault(domain, {})[slot] = value
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
    for s_idx in reversed(range(len(span_info['act_slot_name']))):
        name = span_info['act_slot_name'][s_idx]
        dom = span_info["act_type"][s_idx].split("-")[0].lower()
        prefix = "value" if name in ["day", "departure", "destination", "area", "food", "pricerange", "price", "time"] else dom
        name = "reference" if name == "ref" else name
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

def gt_confidence(model, tokenizer, streamer, utterance: str, context: str) -> float:
    filled_confidence_prompt = confidence_prompt.format(utterance, context)
    confidence_input = tokenizer(filled_confidence_prompt, return_tensors="pt").input_ids.cuda()
    outputs = model.generate(confidence_input, streamer=streamer, temperature=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    confidence = generated_text.split("Answer:")[1].strip().lower()

    confidence_map = {"easy": (0.9, 1), "medium": (0.8, 0.9), "hard": (0.7, 0.8)}
    return random.uniform(*confidence_map.get(confidence, (0.5, 0.5)))

def load_sgd(context_size, split='train', total=10, shuffle=True, available_domains=None, only_single_domain=False, restrict_domains=None):
    dataset = load_dataset('schema_guided_dstc8')[split]

    if shuffle:
        transformers.set_seed(42)
        dataset = dataset.shuffle()

    domain_counts = defaultdict(int)
    if available_domains:
        domain_counts.update({d: 0 for d in available_domains})

    n = 1
    all_domain_slots = {}

    for dialog in dataset:
        if (only_single_domain and len(dialog['services']) != 1) or \
           (restrict_domains and not all(dom in restrict_domains for dom in dialog['services'])):
            continue

        domain_gt = dialog['services'][0].split('_')[0].lower()
        if domain_counts[domain_gt] >= total or (available_domains and domain_gt not in available_domains):
            continue

        domain_counts[domain_gt] += 1
        n += 1
        all_domain_slots.setdefault(domain_gt, set())

        last_state = {}
        for tn in range(0, len(dialog['turns']['utterance']), 2):
            context = [f"Customer: {t}" if i % 2 == 0 else f"Assistant: {t}" for i, t in enumerate(dialog['turns']['utterance'][:tn + 1])]
            state = dialog['turns']['frames'][tn].get('state', [])

            state_dict = {k: v[0] for k, v in zip(state['slot_name'], state['slot_value_list'])} if state else {}
            new_state = {domain_gt: state_dict}

            state_update = calculate_state_update(last_state, new_state)
            last_state = new_state

            database_results = dialog['turns']['frames'][tn + 1]['service_results'][0]
            turn = create_turn(context, context_size, dialog['dialogue_id'], tn, dialog, last_state, domain_gt, state_update, database_results)
            yield turn

def delexicalize_sgd(utterance: str, frames):
    for s_idx in reversed(range(len(frames['slots'][0]['slot']))):
        name = frames['slots'][0]['slot'][s_idx]
        placeholder = f'[{name}]'
        utterance = utterance[:frames['slots'][0]['start'][s_idx]] + placeholder + utterance[frames['slots'][0]['exclusive_end'][s_idx]:]
    return utterance
