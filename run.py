# import libraries
from datasets import load_dataset
from tqdm import tqdm
import pickle
import json
import copy
import argparse
import time
from minicons import cwe, scorer
import json
import statistics
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import torch.nn.functional as F
import os

# import files
from loaders import load_mwoz
from confidence_minicons import get_confidence, minicons_confidence
from evaluation_minicons import get_gold_turn_states
from model import get_model, get_tokenizer, response
from prompts import PROMPT_STRATEGIES
from slot_description import DOMAIN_SLOT_DESCRIPTION, DOMAIN_EXPECTED_SLOT, EXPECTED_DOMAIN
from utils import ExampleRetriever, ExampleFormatter, parse_state_confidence_pair
from mwzeval.utils import load_gold_states
from mwzeval.JGA_metrics import overall_jga, turn_accuracy

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="multiwoz_database")
    parser.add_argument("--faiss", type=str, default="multiwoz-context-db.vec")
    parser.add_argument("--ontology", type=str, default="ontology.json")
    parser.add_argument("--context_size", type=int, default=2)
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--dials_total", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="vanilla") # vanilla, topk, multistep
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7) # temperature scaling
    parser.add_argument("--gamma", type=float, default=0) # hyperparameter for word confidence decay
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--result", type=str, default="results")
    parser.add_argument("--plot_result", type=str, default="plot_results_gpt4")
    parser.add_argument("--verbalized", type=int, default=0)
    parser.add_argument("--averaged_method", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)

    args = parser.parse_args()
    config = {
        "DB_PATH": args.database,
        "FAISS_PATH": args.faiss,
        "ONTOLOGY_PATH": args.ontology,
        "CONTEXT_SIZE": args.context_size,
        "NUM_EXAMPLES": args.num_examples,
        "DIALS_TOTAL": args.dials_total,
        "SPLIT": args.split,
        "PROMPT": args.prompt,
        "FEWSHOT": args.few_shot,
        "TEMP": args.temperature,
        "GAMMA": args.gamma,
        "SPLIT": args.split,
        "RESULT_PATH": args.result,
        "PLOT_RESULT_PATH": args.plot_result,
        "VERBALIZED": args.verbalized,
        "AVERAGED_METHOD": args.averaged_method
    }

    if args.verbalized:
        if args.few_shot:
            input_file_name = f"result_confidence_{args.prompt}_fewshot1_temp{str(args.temperature)}_{args.split}_start{args.start_idx}_dials{str(args.dials_total)}_{args.model_name}_verbalized"
        else:
            input_file_name = f"result_confidence_{args.prompt}_zeroshot_temp{str(args.temperature)}_{args.split}_start{args.start_idx}_dials{str(args.dials_total)}_{args.model_name}_verbalized"
    else:
        if args.few_shot:
            input_file_name = f"result_confidence_{args.prompt}_fewshot1_temp{str(args.temperature)}_{args.split}_start{args.start_idx}_dials{str(args.dials_total)}_{args.model_name}"
        else:
            input_file_name = f"result_confidence_{args.prompt}_zeroshot_temp{str(args.temperature)}_{args.split}_start{args.start_idx}_dials{str(args.dials_total)}_{args.model_name}"
    
    # wandb
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["AZURE_OPENAI_KEY"] = "YOUR OPENAI KEY"

    wandb.init(project="PROJECT NAME", entity="ENTITY", config=config, settings=wandb.Settings(start_method="fork"), name=input_file_name)
    # report_table = wandb.Table(columns=['id', 'context', 'gt_domain', 'turn_state', 'total_state', 'slot_confidence', 'value_confidence', 'pair_confidence'])
    report_table = wandb.Table(columns=['id', 'gt_domain', 'turn_state', 'total_state', 'pair_verbalized', 'slot_confidence', 'value_confidence', 'pair_confidence', 'pair_minicons'])


    model = get_model(model_id=args.model_name, is_8bit=True)
    tokenizer = get_tokenizer(model_id=args.model_name)
    streamer = TextIteratorStreamer(
        tokenizer, 
        timeout = 10, 
        skip_prompt = True, 
        skip_special_tokens = True)

    data = load_mwoz(database_path=args.database, context_size=args.context_size, split=args.split, shuffle=False, start_idx=args.start_idx, dials_total=args.dials_total)
    with open(args.faiss, 'rb') as file:
        faiss_vs = pickle.load(file)
    with open(args.ontology, 'r') as file:
        ontology = json.load(file)
    state_vs = faiss_vs
    example_retriever = ExampleRetriever(faiss_vs)
    state_retriever = ExampleRetriever(state_vs)
    example_formatter = ExampleFormatter(ontology)

    last_dial_id = None
    dial = 0 
    progress_bar = tqdm(total=args.dials_total)
    tn = 0
    history = []
    total_state = {}
    result = {}

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    ilm_model = scorer.IncrementalLMScorer(model_name, cache_dir="cache")

    for it, turn in enumerate(data):
        if last_dial_id != turn["dialogue_id"]:
            last_dial_id = turn["dialogue_id"]
            dial += 1
            progress_bar.update(1)
            tn = 0
            if dial > args.dials_total:
                break
            history = []
            total_state = {}
            print('=' * 100)
            
        tn += 1
        
        dialog_id = turn["dialogue_id"]
        print(f"{dialog_id}-{tn}")
        question = turn["question"]
        gt_domain = turn["metadata"]["domain"]
        gold_response = turn["metadata"]["response"]
        
        retrieve_history = history + ["Customer: " + question]
        
        prompt = PROMPT_STRATEGIES[args.prompt]
        # print("few_shot: ", int(args.few_shot))
        if args.few_shot:
            state_prompt = prompt.fewshot_prompt
        else:
            state_prompt = prompt.zeroshot_prompt
        
        # examples
        state_examples = [example for example in state_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=20) if example["domain"] == gt_domain][:args.num_examples]
        positive_state_example = example_formatter.format(state_examples[:args.num_examples],
                                                        input_keys = ["context"],
                                                        output_keys = ["state"])
        negative_state_example = None
        history += [f"Customer: {question}"] #user utterance
        
        if args.few_shot:
            kwargs = {
                "history": "\n".join(history),
                "utterance": question.strip(),
                "slot_description": DOMAIN_SLOT_DESCRIPTION[gt_domain],
                "positive_examples": positive_state_example,
                "negative_examples": []
            }
        else:
            kwargs = {
                "history": "\n".join(history),
                "utterance": question.strip(),
                "slot_description": DOMAIN_SLOT_DESCRIPTION[gt_domain]
            }

        # print(kwargs.keys())
        filled_state_prompt = state_prompt(**kwargs)
        # print(f"filled_state_prompt: \n{filled_state_prompt}")
        if tokenizer:
            state_input = tokenizer(filled_state_prompt, return_tensors="pt").input_ids.cuda()
        else:
            state_input = filled_state_prompt
        
        # retried 3
        for i in range(3):
            try:
                outputs = response(model, args.model_name, streamer, state_input, args.temperature)
                break
            except: 
                time.sleep(5)
        if tokenizer:
            generated_text = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
        else:
            generated_text = outputs
        try:
            state_output = generated_text.split("***Output JSON format***")[1]
        except:
            state_output = generated_text
        # state_output = state_output.lower()
        # print(f"state_output: \n{state_output}")
        
        state_output = state_output.lower()
        # print(f"state_output: \n{state_output}")

        slot_values, pair_verbalized = parse_state_confidence_pair(state_output)
        print("closed (verbalized):")
        print(f"pair_verbalized: {pair_verbalized}")
        
        # print(f"slot_values: {slot_values}")
        # print(f"pair_confidences: {pair_confidences}")
        expected_slots = DOMAIN_EXPECTED_SLOT[gt_domain]
        for key in list(slot_values.keys()): 
            if key not in expected_slots: del slot_values[key]
        # print(f"slot_values: {slot_values}")

        # opened (softmax, minicons)

        if not args.verbalized:
            confidence = 1
        slot_confidences, value_confidences, pair_confidences = get_confidence(tokenizer, state_input, outputs, slot_values, confidence, args.gamma, args.averaged_method)
        print("opened (softmax):")
        print(f"slot_confidences: {slot_confidences}")
        print(f"value_confidences: {value_confidences}")
        print(f"pair_confidences: {pair_confidences}")
        
        pair_minicons = minicons_confidence(ilm_model, slot_values, history)
        print("opened (minicons):")
        print(f"pair_minicons: {pair_minicons}")
        
        if gt_domain not in total_state.keys() and slot_values:
            total_state[gt_domain] = {}
        if slot_values:
            total_state[gt_domain].update(slot_values)
        # print(f"total_state: {total_state}")
        if dialog_id not in result.keys():
            result[dialog_id] = []     
        result[dialog_id].append({
            "turn_state": copy.deepcopy(slot_values),
            "total_state": copy.deepcopy(total_state),
            "pair_verbalized": copy.deepcopy(pair_verbalized),
            "slot_confidence": copy.deepcopy(slot_confidences),
            "value_confidence": copy.deepcopy(value_confidences),
            "pair_confidence": copy.deepcopy(pair_confidences),
            "pair_minicons": copy.deepcopy(pair_minicons)
        })

        report_table.add_data(
            f"{dialog_id}-{tn}", # id
            gt_domain, # gt_domain
            json.dumps(slot_values), # turn_state
            json.dumps(total_state), # total_state
            json.dumps(pair_verbalized),
            json.dumps(slot_confidences), # slot_confidence
            json.dumps(value_confidences), # value_confidence
            json.dumps(pair_confidences), # pair_confidence
            json.dumps(pair_minicons)
        )

        history += [f"Assistant: {gold_response}"] # assistnat response

    input_file_name = input_file_name.replace("/", "-")
    wandb.log({"examples": report_table})

    file_name = os.path.join(args.result, f"{input_file_name}.json")
    with open(file_name, "w") as file:
        json.dump(result, file)

    # evaluation
    gold_states = load_gold_states(dataset="multiwoz")
    gold_turn_states = get_gold_turn_states(gold_states)

    OVERALL_JGA = overall_jga(result, gold_states)
    TURN_ACC = turn_accuracy(result, gold_states)

    pair_verbalized_ece, pair_verbalized_roc_auc, pair_confidence_ece, pair_confidence_roc_auc, pair_minicons_ece, pair_minicons_roc_auc = plot_distribution(result=result, gold_state=gold_states, gold_turn_state=gold_turn_states, n_bins=10, PLOT_RESULT_PATH=args.plot_result, input_file_name=input_file_name)

    wandb.log({"OVERALL_JGA": OVERALL_JGA})
    wandb.log({"TURN_ACC": TURN_ACC})
    wandb.log({"pair_verbalized_ece": pair_verbalized_ece})
    wandb.log({"pair_verbalized_roc_auc": pair_verbalized_roc_auc})
    wandb.log({"pair_confidence_ece": pair_confidence_ece})
    wandb.log({"pair_confidence_roc_auc": pair_confidence_roc_auc})
    wandb.log({"pair_minicons_ece": pair_minicons_ece})
    wandb.log({"pair_minicons_roc_auc": pair_minicons_roc_auc})
    
