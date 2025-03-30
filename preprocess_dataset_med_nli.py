import os
from datasets import load_dataset, concatenate_datasets
import json

instruction = """<Instruction>
You are a logical reasoning assistant. You will analyze the relationship between a hypothesis and a premise to determine whether the premise entails, contradicts, or is neutral toward the hypothesis.
definitions
Entailment: The premise logically implies the hypothesis. If the premise is true, the hypothesis must also be true.
Contradiction: The premise logically implies the negation of the hypothesis. If the premise is true, the hypothesis must be false.
Neutral: The premise neither entails nor contradicts the hypothesis. The hypothesis could be either true or false given the premise.
Process:
First, carefully read and understand both the hypothesis and premise.
Consider what facts or conclusions can be derived from the premise.
Determine whether these facts/conclusions necessarily support, contradict, or are insufficient to judge the hypothesis.
Select the appropriate relationship category.
</Instruction>
"""

zero_shot_prompt = """
<input>
Hypothesis: "{hypothesis}"
Premise: "{premise}"
</Input>

<format>
Answer: {{"output": "SELECTED_OPTION"}}
Where SELECTED_OPTION is exactly one of:
"entailment" (if the premise entails the hypothesis)
"contradiction" (if the premise contradicts the hypothesis)
"neutral" (if the premise is neutral toward the hypothesis)
</format>
"""

few_shot_prompt = """
<examples>
Premise: "A woman is reading a book."
Hypothesis: "A person is engaged in reading."
Relationship: Entailment

Premise: "A child is playing in the park."
Hypothesis: "The park is empty."
Relationship: Contradiction

Premise: "She is reading a book in the library."
Hypothesis: "The library has a vast collection of books.
Relationship: Neutral
</examples>

<input>
Hypothesis: "{hypothesis}"
Premise: "{premise}"
</input>

<format>
Answer: {{"output": "SELECTED_OPTION"}}
Where SELECTED_OPTION is exactly one of:
"entailment" (if the premise entails the hypothesis)
"contradiction" (if the premise contradicts the hypothesis)
"neutral" (if the premise is neutral toward the hypothesis)
</format>
"""

answer = """
<output>
{{"output": "{gold_label}"}}
</output>
"""

def main():
    dataset = load_dataset("presencesw/all_nli_med_v1")

    dataset = concatenate_datasets([dataset["train"], dataset["validation"]])

    # print(dataset)
    # print(dataset[0])

    # premise = sentenece1
    # hypothesis = sentence2

    def preprocess_function_zero_shot(examples):
        premise = examples["sentence1"]
        hypothesis = examples["sentence2"]
        label = examples["gold_label"]
        inputs = []
        targets = []
        for p, h, l in zip(premise, hypothesis, label):
            inputs.append(zero_shot_prompt.format(premise=p, hypothesis=h))
            targets.append(answer.format(gold_label=l))
        return {"inputs": inputs, "targets": targets}

    def preprocess_function_few_shot(examples):
        premise = examples["sentence1"]
        hypothesis = examples["sentence2"]
        label = examples["gold_label"]
        inputs = []
        targets = []
        for p, h, l in zip(premise, hypothesis, label):
            inputs.append(few_shot_prompt.format(premise=p, hypothesis=h))
            targets.append(answer.format(gold_label=l))
        return {"inputs": inputs, "targets": targets}
        
    dataset_zeroshot = dataset.map(preprocess_function_zero_shot, batched=True)
    dataset_fewshot = dataset.map(preprocess_function_few_shot, batched=True)

    # print(dataset_zeroshot[0])

    zeroshot_list = []
    for i in dataset_zeroshot:
        input_dict = dict()
        input_dict["instruction"] = instruction
        input_dict["input"] = i["inputs"]
        input_dict["output"] = i["targets"]
        zeroshot_list.append(input_dict)


    with open("data_nli/zero_shot.json", "w") as f:
        json.dump(zeroshot_list, f, indent=4)
        
    print("zero-shot dataset saved to data_nli/zero_shot.json")

    fewshot_list = []
    for i in dataset_fewshot:
        input_dict = dict()
        input_dict["instruction"] = instruction
        input_dict["input"] = i["inputs"]
        input_dict["output"] = i["targets"]
        fewshot_list.append(input_dict)
    with open("data_nli/few_shot.json", "w") as f:
        json.dump(fewshot_list, f, indent=4)
        
    print("Few-shot dataset saved to data_nli/few_shot.json")

if __name__ == "__main__":
    main()