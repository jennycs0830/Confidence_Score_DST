import json

with open("gold_states.json", "r") as file:
    data = json.load(file)
    
turn_level = {}

prev_total_state = {}
for dialog_id in data.keys():
    for idx, turn in enumerate(data[dialog_id]):
        id = f"{dialog_id}-{idx}"
        total_state = turn
        
        # turn state
        turn_state = {}
        gt_domain = []
        for domain in total_state:
            if domain not in prev_total_state.keys():
                if domain not in gt_domain:
                    gt_domain.append(domain)
                turn_state[domain] = total_state[domain]
            else:
                for slt, val in total_state[domain].items():
                    if slt not in prev_total_state[domain].keys() or val != prev_total_state[domain][slt]:
                        if domain not in turn_state.keys():
                            turn_state[domain] = {}
                        turn_state[domain][slt] = val
                        if domain not in gt_domain:
                            gt_domain.append(domain)
                        
        dialog_data = {}
        dialog_data["total_state"] = total_state
        dialog_data["turn_state"] = turn_state
        dialog_data["gt_domain"] = gt_domain
        
        turn_level[id] = dialog_data
        
        prev_total_state = total_state
        
with open("gold_turn_state.json", "w") as file:
    json.dump(turn_level, file, indent=4)
                        
            