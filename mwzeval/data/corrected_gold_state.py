import json

def replace_key_in_dialogues(dialogues, target_domain, old_key, new_key):
    for dialogue in dialogues:
        for entry in dialogues[dialogue]:
            if target_domain in entry:
                domain_data = entry[target_domain]
                if old_key in domain_data:
                    domain_data[new_key] = domain_data.pop(old_key)

with open("gold_states.json", "r") as file:
    dialogues = json.load(file)

replace_key_in_dialogues(dialogues, "train", "arrive", "arriveby")
replace_key_in_dialogues(dialogues, "train", "leave", "leaveat")

with open("gold_states.json", "w") as file:
    json.dump(dialogues, file, indent=2, ensure_ascii=False)

