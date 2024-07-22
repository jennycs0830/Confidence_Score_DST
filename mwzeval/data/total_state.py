import json
import copy

with open("gold_states_24.json", "r") as file:
    data = json.load(file)

gold_states = {}
for dialog_id in data.keys():
    total_state = {}
    gold_states[dialog_id] = []
    for turn in range(len(data[dialog_id])):
        state = data[dialog_id][turn]
        for domain in state:
            domain_slots = state[domain]
            if domain not in total_state.keys():
                total_state[domain] = {}
            total_state[domain].update(domain_slots)
        gold_states[dialog_id].append(copy.deepcopy(total_state))

with open("new_24.json", "w") as file:
    json.dump(gold_states, file, indent=2)
