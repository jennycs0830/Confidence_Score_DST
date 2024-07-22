import json
import re
import random
from copy import deepcopy
from typing import Dict, Any
from collections import defaultdict

import numpy as np
from fuzzywuzzy import fuzz
import evaluate
from nltk.tokenize import word_tokenize
from langchain.vectorstores import VectorStore

def parse_selfprob(response: str) -> float:
    pattern = r'confidence:\s*"?([\d.]+)"?'
    match = re.search(pattern, response)
    return float(match.group(1)) if match else 0

def parse_state(state: str) -> Dict[str, str]:
    def sanitize(dct):
        for key, value in dct.items():
            if isinstance(value, dict):
                dct[key] = sanitize(value)
            elif not isinstance(value, str):
                dct[key] = str(value)
        return dct

    state = state.replace("state:", "").replace("Output JSON format***", "")
    pattern = r'"([a-zA-Z0-9 _-]+)"\s*:\s*"(.*?)"'
    slotvals = re.findall(pattern, state)

    out_state = {sv[0]: ":".join(sv[1:]).strip("'\" ") for sv in slotvals if sv[1] not in ['', 'unknown', 'dontcare', 'none', 'null', '__null__', '_null_', "_none_", "_dontcare_", "_unknown_", "na", "?", "??", "???"]}
    return sanitize(out_state)

def parse_multi_state(state: str) -> Dict[str, Any]:
    patterns = [
        r'"state":\s*{(.*?)}',
        r'{"state":\s*{(.*?)},\s*"confidence":\s*"(.*?)"}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, state)
        if matches:
            break

    best_state, highest_confidence = None, 0
    for match in matches:
        try:
            state_dict = match[0]
            confidence = float(match[1]) if len(match) > 1 else 0
            if confidence >= highest_confidence:
                highest_confidence = confidence
                best_state = state_dict
        except:
            continue

    slot_values = parse_state(best_state) if best_state else {}
    return slot_values, highest_confidence

def parse_state_confidence_pair(state: str) -> Dict[str, Any]:
    patterns = [
        r'{"state":\s*{(.*?)},\s*"confidence":\s*"(.*?)"}',
        r'{"state":\s*{(.*?)}(.*?)}',
        r'{"state:\s*"{(.*?)}(.*?)}'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, state)
        if matches:
            break

    slot_values, pair_confidences = {}, {}
    for match in matches:
        try:
            state_dict = match[0]
            confidence = float(match[1]) if len(match) >= 1 else 0
            slot_value = parse_state(state_dict)
            slot_values.update(slot_value)
            for slt, val in slot_value.items():
                key_name = f"{slt}_{val}"
                pair_confidences[key_name] = confidence
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return slot_values, pair_confidences

class ExampleRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, text: str, k: int = 2) -> list[Dict]:
        results = self.vector_store.similarity_search(text, k=k)
        return [{
            'context': doc.metadata['context'],
            'state': doc.metadata['state'],
            'full_state': doc.metadata['full_state'],
            'response': doc.metadata['response'],
            'database': doc.metadata['database'],
            'domain': doc.metadata['domain'],
            'confidence': doc.metadata['confidence']
        } for doc in results]

class SlotExampleFormatter:
    def __init__(self, ontology: Dict):
        self.ontology = ontology

    def format(self, examples: list[Dict[str, Any]], input_keys: list[str], output_keys: list[str], use_json: bool = False, corrupt_state: bool = False, expected_slot: str = "") -> list[Dict[str, str]]:
        examples = deepcopy(examples)
        if corrupt_state:
            examples = [self._corrupt_example(example) for example in examples]
        for example in examples:
            state_domains = list(example['state'].keys())
            example['state'] = example['state'][state_domains[0]] if state_domains else {}
        examples = [self._example_to_str(example, expected_slot, use_json) for example in examples]
        return [self._prepare_example(example, input_keys, output_keys) for example in examples]

    def _corrupt_example(self, example: Dict) -> Dict:
        for domain, dbs in example['state'].items():
            for slot, value in dbs.items():
                slot_otgy_name = f"{domain}-{slot}"
                example['state'][domain][slot] = random.choice(self.ontology.get(slot_otgy_name, [value]))
        return example

    def _example_to_str(self, example: Dict, expected_slot: str, use_json=False) -> Dict:
        for key, val in example.items():
            if key == 'state':
                slot_to_remove = [slot for slot in val.keys() if slot != expected_slot]
                for slot in slot_to_remove:
                    del example[key][slot]
            if isinstance(val, dict):
                example[key] = json.dumps(val) if use_json else "-".join((f"{slot}:'{value}'" for slot, value in val.items()))
            else:
                example[key] = str(val)
        return example

    def _prepare_example(self, example: Dict, input_keys: list[str], output_keys: list[str]) -> Dict:
        example['input'] = '\n'.join((f"{key if key != 'full_state' else 'state'}: {example[key]}" for key in input_keys))
        example['output'] = '\n'.join((f"{key}: {example[key]}" for key in output_keys))
        return example

class ExampleFormatter:
    def __init__(self, ontology: Dict):
        self.ontology = ontology

    def format(self, examples: list[Dict[str, Any]], input_keys: list[str], output_keys: list[str], use_json: bool = False, corrupt_state: bool = False) -> list[Dict[str, str]]:
        examples = deepcopy(examples)
        if corrupt_state:
            examples = [self._corrupt_example(example) for example in examples]
        for example in examples:
            state_domains = list(example['state'].keys())
            example['state'] = example['state'][state_domains[0]] if state_domains else {}
        examples = [self._example_to_str(example, use_json) for example in examples]
        return [self._prepare_example(example, input_keys, output_keys) for example in examples]

    def _corrupt_example(self, example: Dict) -> Dict:
        for domain, dbs in example['state'].items():
            for slot, value in dbs.items():
                slot_otgy_name = f"{domain}-{slot}"
                example['state'][domain][slot] = random.choice(self.ontology.get(slot_otgy_name, [value]))
        return example

    def _example_to_str(self, example: Dict, use_json=False) -> Dict:
        for key, val in example.items():
            if key == 'state':
                example[key] = f"```json{json.dumps(val)}```"
                continue
            if isinstance(val, dict):
                example[key] = json.dumps(val) if use_json else "-".join((f"{slot}:'{value}'" for slot, value in val.items()))
            else:
                example[key] = str(val)
        return example

    def _prepare_example(self, example: Dict, input_keys: list[str], output_keys: list[str]) -> Dict:
        example['input'] = '\n'.join((f"{key if key != 'full_state' else 'state'}: {example[key]}" for key in input_keys))
        example['output'] = '\n'.join((f"{key}: {example[key]}" for key in output_keys))
        return example

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")

def parse_state_with_confidence(pair_confidence, threshold):
    state, low_conf_state = {}, {}
    for pair, conf in pair_confidence.items():
        try:
            slot, value = pair.split("_")
            if conf >= threshold:
                state[slot] = value
            else:
                low_conf_state[pair] = conf
        except:
            pass
    return state, low_conf_state

def update_total_state(turn_state, domain, prev_total_state):
    if domain not in prev_total_state:
        prev_total_state[domain] = {}
    prev_total_state[domain].update(turn_state)
    return prev_total_state

def parse_response(response, filled_response_prompt):
    response = response.split("Output: ")[1]
    response = response.split("Customer:")[0].split("User:")[0].replace("Assistant:", "").strip()
    return response
