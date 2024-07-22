import torch.nn.functional as F
import torch
from minicons import cwe
from minicons import scorer
import json
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

def add_selfprob(selfprob_conf, pair_confidence):
    return {key: selfprob_conf * conf for key, conf in pair_confidence.items()}

def add_selfprob_slot(selfprob_conf_dict, pair_confidence):
    return {key: conf * selfprob_conf_dict.get(key, 1) for key, conf in pair_confidence.items()}

def extract_word_conf(word, output_tokens, confidences, start_idx, input_len, gamma=0, averaged_method=0):
    origin_idx = start_idx
    cleaned_word = ''.join([c for c in word if c not in " -:,\n\""])
    vocab = ""
    subwords = []
    overall_confidence = 0 if averaged_method else 1

    for i in range(start_idx, len(output_tokens)):
        token = ''.join([c for c in output_tokens[i] if c not in " -:,\n\""])
        if not token:
            start_idx = i + 1
            continue

        if token in cleaned_word and vocab != cleaned_word:
            subwords.append(token)
            vocab += token
            confidence = confidences[i - input_len]
            
            if averaged_method:
                overall_confidence += confidence * (gamma ** len(subwords)) if gamma else confidence
            else:
                overall_confidence *= confidence

            start_idx = i + 1
            if vocab == cleaned_word:
                break
        else:
            subwords, vocab = [], ""
            if averaged_method:
                overall_confidence = 0

    if averaged_method and subwords:
        overall_confidence /= len(subwords)

    return overall_confidence, (start_idx if vocab == cleaned_word else origin_idx)

def get_confidence(tokenizer, state_input, outputs, slot_values, response_conf, gamma=0, averaged_method=0):
    generated_token_ids = outputs['sequences'][0]
    tokens = [tokenizer.decode([int(token_id)]) for token_id in generated_token_ids]
    confidences = [F.softmax(logits, dim=-1).max().item() for logits in outputs['scores']]

    idx = len(state_input[0])
    input_len = len(state_input[0])

    slot_confidences = {}
    value_confidences = {}
    pair_confidences = {}

    for slot, value in slot_values.items():
        slot_conf, idx = extract_word_conf(slot, tokens, confidences, idx, input_len, gamma, averaged_method) if slot else (0, idx)
        value_conf, idx = extract_word_conf(value, tokens, confidences, idx, input_len, gamma, averaged_method) if value else (0, idx)

        pair_conf = slot_conf * value_conf * response_conf

        slot_confidences[slot] = slot_conf
        value_confidences[value] = value_conf
        pair_confidences[f"{slot}_{value}"] = pair_conf

    return slot_confidences, value_confidences, pair_confidences

minicons_prompt = """
Focus on last utterance.
conversation history:
{}

predicted slot-value pair:
{}
"""

def minicons_confidence(ilm_model, slot_values, history):
    if len(history) >= 3:
        history = history[-2:]
    
    return {f"{slot}_{value}": word_minicons(ilm_model, minicons_prompt.format(history, f"{slot}: {value}"), value) for slot, value in slot_values.items()}

def word_minicons(ilm_model, response, word):
    try:
        token_scores = ilm_model.token_score([response], prob=True)[0]
    except Exception as e:
        print(e)
        return 0

    word_scores = [score for key, score in token_scores if key in word]
    return sum(word_scores) / len(word_scores) if word_scores else 0

