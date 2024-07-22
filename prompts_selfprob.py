from dataclasses import dataclass
from typing import Text, List, Dict, Any

@dataclass
class SimpleTemplatePrompt:
    template: str
    args_order: list
    
    def __call__(self, **kwargs: Any) -> Text:
        args = [kwargs[arg] for arg in self.args_order]
        return self.template.format(*args)
    
@dataclass
class FewShotPrompt(SimpleTemplatePrompt):
    def __call__(self, **kwargs: Any) -> Text:
        positive_examples = self._process_positive_examples(kwargs["positive_examples"])
        negative_examples = self._process_negative_examples(kwargs["negative_examples"])
        slot_description = kwargs["slot_description"]
        history = kwargs["history"]
        args = [kwargs[arg] for arg in self.args_order]
        
        return self.template.format(slot_description, positive_examples, history)
    
    def _process_positive_examples(self, positive_examples: list) -> Text:
        output = "\n"
        for n, example in enumerate(positive_examples):
            output += "---------------------" + \
                      f"Example {n}:\n" + \
                      f"{example['input']}\n" + \
                      f"\n{example['output']}\n"
        return output + "\n"
    
    def _process_negative_examples(self, negative_examples: list) -> Text:
        output = "\n"
        for n, example in enumerate(negative_examples):
            output += f"Negative example {n}:\n" + \
                      f"{example['input']}\n" + \
                      f"\n{example['output']}\n"
        return output + "\n"



# selfprob_prompt = """
# Does this conversation:
# {}
# Support state: {}?
# Analyze the state, provide 1 brief reason, and give confidence (0-1).
# Format: confidence: "X", reason: "brief reason"
# Think carefully and step by step.
# Output:
# """

selfprob_prompt = """
Conversation:
{}
State:
{}
How likely is the above state to be correct?
Analyze the state, provide 1 brief reason, and give confidence (0-1).
Format: confidence: "X", reason: "brief reason"
Think carefully and step by step.
Output:
"""

@dataclass
# Prompt5
class VanillaPrompt:
    fewshot_prompt = FewShotPrompt(template="""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Capture entity values from the LAST UTTERANCE of the conversation.
FOCUS ONLY ON THE VALUES MENTIONED IN THE LAST UTTERANCE.
Format the output as a valid JSON object for each entity-value pair.
Format: {{"state": {{"_entity_":"_value_"}}, "confidence": "X"}}
Where X is the Confidence of the answer.

Fill the actual entity value into the placeholder encapsulated with underscores.
Put "```" as EOS token at the end of response.
Values that should be captured are:
{}
Do not capture any other values!
If not specified, do not respond to that slot-value.
--------------------
{}
--------------------
MAKE SURE TO SEPARATE EACH SLOT-VALUE PAIR, AND ALONG WITH EACH OF THEIR CONFIDENCE (0-1).
Format the output as:
```json
[
    {{"state": {{"_entity1_":"_value1_"}}, "confidence": "_X1_"}}, 
    {{"state": {{"_entity2_":"_value2_"}}, "confidence": "_X2_"}},
    {{"state": {{"_entity3_":"_value3_"}}, "confidence": "_X3_"}},
]```

Now complete the following example, AND PROVIDE CONFIDENCE THAT IT'S CORRECT:
input: <|eot_id|>
<|start_header_id|>user<|end_header_id|>
{}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
***Output JSON format***
Output: ```json[
""", args_order=["slot_description", "positive_examples", "history", "utterance"])
    
    zeroshot_prompt = SimpleTemplatePrompt(template="""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Capture entity values from the LAST UTTERANCE of the conversation.
FOCUS ONLY ON THE VALUES MENTIONED IN THE LAST UTTERANCE.
Format the output as a valid JSON object for each entity-value pair.
Format: {{"state": {{"_entity_":"_value_"}}, "confidence": "X"}}
Where X is the Confidence of the answer.

Fill the actual entity value into the placeholder encapsulated with underscores.
Put "```" as EOS token at the end of response.
Values that should be captured are:
{}
Do not capture any other values!
If not specified, do not respond to that slot-value.

MAKE SURE TO SEPARATE EACH SLOT-VALUE PAIR, AND ALONG WITH EACH OF THEIR CONFIDENCE (0-1).
Format the output as:
```json
[
    {{"state": {{"_entity1_":"_value1_"}}, "confidence": "_X1_"}}, 
    {{"state": {{"_entity2_":"_value2_"}}, "confidence": "_X2_"}},
    {{"state": {{"_entity3_":"_value3_"}}, "confidence": "_X3_"}},
]```

Now complete the following example, AND PROVIDE CONFIDENCE THAT IT'S CORRECT:
input: <|eot_id|>
<|start_header_id|>user<|end_header_id|>
{}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
***Output JSON format***
Output: ```json[
""", args_order=["slot_description", "history", "utterance"])



# # Prompt3
# class VanillaPrompt:
#     fewshot_prompt = FewShotPrompt(template="""
# Capture entity values from the LAST UTTERNACE of the conversation.
# FOCUS ONLY ON THE VALUES MENTIONED IN THE LAST UTTERANCE.
# Format the output as a valid JSON object for each entity-value pair.
# Format: {{"state": {{"entity":"value"}}, "confidence": "X"}}
# Put "```" as EOS token at the end of response.
# Values that should be captured are:
# {}
# Do not capture any other values!
# If not specified, leave the value empty.  
# --------------------
# {}{}
# --------------------
# Provide 1 posiible entity values based on the last utterance, along with their confidence (0-1). Format the output as:
# ```json{{[{{"state": {{"entity":"value", "entity":"value"}}, "confidence": "X"}}]}}```
# Where X is the Confidence of the answer.

# Now complete the following example, AND PROVIDE CONFIDENCE THAT IT'S CORRECT:
# input: {}  
# Customer: {}

# ***Output JSON format***
# Output: ```json{{""", args_order=["history", "utterance"])
    
#     zeroshot_prompt = SimpleTemplatePrompt(template="""
# Capture entity values from the LAST UTTERANCE of the conversation.
# FOCUS ONLY ON THE VALUES MENTIONED IN THE LAST UTTERANCE.
# Format the output as a valid JSON object for each entity-value pair.
# Format: {{"state": {{"_entity_":"_value_"}}}}
# Fill the actual entity value into the placeholder encapsulated with underscores.
# Put "```" as EOS token at the end of response.
# Values that should be captured are:
# {}
# Do not capture any other values!
# If not specified, do not respond to that slot-value.

# Provide 1 posiible entity values based on the last utterance, along with their confidence (0-1). . MAKE SURE TO SEPARATE EACH SLOT-VALUE PAIR.
# Format the output as:
# ```json{{[{{"state": {{"_entity1_":"_value1_"}}}}, {{"state": {{"_entity2_":"_value2_"}}}}]}}```
# Where X is the Confidence of the answer.

# Now complete the following example, AND PROVIDE CONFIDENCE THAT IT'S CORRECT:
# input: {}  
# Customer: {}

# ***Output JSON format***
# Output: ```json{{""", args_order=["slot_description", "history", "utterance"])


# Prompt1
# class VanillaPrompt:
#     fewshot_prompt = FewShotPrompt(template="""
# Capture entity values from the LAST UTTERNACE of the conversation.
# FOCUS ONLY ON THE VALUES MENTIONED IN THE LAST UTTERANCE.
# Format the output as a valid JSON object for each entity-value pair.
# state: ```json{{"entity":"value", "entity":"value"}}```
# Values that should be captured are:
# {}
# Do not capture any other values!
# If not specified, leave the value empty.  
# --------------------
# {}{}
# --------------------
# Now complete the following example:
# input: {}  
# Customer: {}

# ***Output JSON format***
# Output: ```json{{""", args_order=["history", "utterance"])
    
#     zeroshot_prompt = SimpleTemplatePrompt(template="""
# Capture entity values from the LAST UTTERANCE of the conversation.
# FOCUS ONLY ON THE VALUES MENTIONED IN THE LAST UTTERANCE.
# Format the output as a valid JSON object for each entity-value pair.
# state: ```json{{"entity":"value", "entity":"value"}}```
# Values that should be captured are:
# {}
# Do not capture any other values!
# If not specified, leave the value empty.  

# Now complete the following example:
# input: {}  
# Customer: {}

# ***Output JSON format***
# Output: ```json{{""", args_order=["history", "utterance"])


@dataclass
class TopKPrompt:
    fewshot_prompt = FewShotPrompt(template="""
Capture entity values from the LAST UTTERNACE of the conversation.
FOCUS ONLY ON THE VALUES MENTIONED IN THE LAST UTTERANCE.
Format the output as a valid JSON object for each entity-value pair.
Format: {{"state": {{"entity":"value", "entity":"value"}}, "confidence": "X"}}
Put "```" as EOS token at the end of response.
Values that should be captured are:
{}
Do not capture any other values!
IF NOT SPECIFIED, LEAVE THE VALUE EMPTY.
--------------------
{}{}
--------------------
Provide 3 different possible entity values based on the last utterance, along with their confidence (0-1). Format the output as:
```json{{[
{{"state": {{"entity1":"value1", "entity2":"value2", ...}}, "confidence": "X"}},
{{"state": {{"entity1":"value3", "entity2":"value4", ...}}, "confidence": "Y"}},
{{"state": {{"entity1":"value5", "entity2":"value6", ...}}, "confidence": "Z"}},
]}}
```
Where X, Y, and Z are the Confidence of each possible answer.

Now complete the following example, AND PROVIDE 3 DIFFERENT GUESSES AND CONFIDENCE THAT IT'S CORRECT:
input: {}  
Customer: {}

***Output JSON format***
Output: ```json{{""", args_order=["history", "utterance"])
    zeroshot_prompt = SimpleTemplatePrompt(template="""
Capture entity values from the LAST UTTERNACE of the conversation.
FOCUS ONLY ON THE VALUES MENTIONED IN THE LAST UTTERANCE.
Format the output as a valid JSON object for each entity-value pair.
Format: {{"state": {{"entity":"value", "entity":"value"}}, "confidence": "X"}}
Put "```" as EOS token at the end of response.
Values that should be captured are:
{}
Do not capture any other values!
IF NOT SPECIFIED, LEAVE THE VALUE EMPTY.

Provide 3 different possible entity values based on the last utterance, along with their confidence (0-1). Format the output as:
```json{{[[
{{"state": {{"entity1":"value1", "entity2":"value2", ...}}, "confidence": "X"}},
{{"state": {{"entity1":"value3", "entity2":"value4", ...}}, "confidence": "Y"}},
{{"state": {{"entity1":"value5", "entity2":"value6", ...}}, "confidence": "Z"}},
]]}}
```
Where X, Y, and Z are the Confidence of each possible answer.

Now complete the following example, AND PROVIDE 3 DIFFERENT GUESSES AND CONFIDENCE THAT IT'S CORRECT:
input: {}  
Customer: {}

***Output JSON format***
Output: ```json{{""", args_order=["history", "utterance"])

@dataclass
class MultiStepPrompt:
    fewshot_prompt = FewShotPrompt(template="""
Capture entity values from the last utterance of the conversation.
Focus only on the values mentioned in the last utterance.
Format each entity-value pair as "entity:value" with no spaces around colons.
SEPERATE ENTITY-VALUE PAIRS BY HYPHENS.
{}
Do not capture any other values!
IF NOT SPECIFIED, LEAVE THE VALUE EMPTY.
--------------------
{}{}
--------------------
Read the question, break down the problem into 2 steps, think step by step.
Step 1: Begin by identifying the key entities mentioned in the last utterance of the conversation.
Step 2: For each identified entity, extract the corresponding value.
Conclude with a single, concise response string that encapsulates all entity-value pairs.

Use the following format to answer: 
Step 1: [Your reasoning]
Step 2: [Your reasoning]
***Output JSON format***
state: [formatted slot-value pairs]

Now complete the following example, AND PROVIDE 3 BEST GUESSES AND THE PROBABILITY THAT EACH IS CORRECT(0-100):
input: {}  
Customer: {}

***Output***
state: """, args_order=["history", "utterance"])
    
    zeroshot_prompt = SimpleTemplatePrompt(template="""
Capture entity values from the last utterance of the conversation.
Focus only on the values mentioned in the last utterance.
Format each entity-value pair as "entity:value" with no spaces around colons.
SEPERATE ENTITY-VALUE PAIRS BY HYPHENS.
{}
Do not capture any other values!
IF NOT SPECIFIED, LEAVE THE VALUE EMPTY.

Read the question, break down the problem into 2 steps, think step by step.
Step 1: Begin by identifying the key entities mentioned in the last utterance of the conversation.
Step 2: For each identified entity, extract the corresponding value.
Conclude with a single, concise response string that encapsulates all entity-value pairs.

Use the following format to answer: 
Step 1: [Your reasoning]
Step 2: [Your reasoning]
***Output***
state: [formatted slot-value pairs]

Now complete the following example, AND PROVIDE 3 BEST GUESSES AND THE PROBABILITY THAT EACH IS CORRECT(0-100):
input: {}  
Customer: {}

***Output***
state: """, args_order=["history", "utterance"])

PROMPT_STRATEGIES = {
    "vanilla": VanillaPrompt,
    "topk": TopKPrompt,
    "multistep": MultiStepPrompt
}