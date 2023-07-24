import os
import json
from datasets import load_dataset

# story cloze needs to be downloaded manually, see README.md
dataset = load_dataset("story_cloze", "2016", data_dir="storycloze")
print(dataset)


# eval data
from promptsource.templates import DatasetTemplates
story_cloze_templates = DatasetTemplates('story_cloze/2016')


template_names = [
    'Answer Given options',
    'Choose Story Ending',
    'Movie What Happens Next',
    'Story Continuation and Options',
    'Generate Ending',
    'Novel Correct Ending',
]

for template_name in template_names:
    key = "story_cloze_{}".format(template_name.replace(" ", "_"))
    out_dir = "../data_eval/story_cloze_{}".format(template_name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    
    jsonl_filename = "story_cloze_{}_valid.jsonl".format(template_name.replace(" ", "_"))

    prompt = story_cloze_templates[template_name]
    all_lines = []
    for example in dataset["validation"]:
        line = prompt.apply(example)
        out_dict = {
            "task": key,
            "input": line[0],
            "output": line[1],
            "options": [example["sentence_quiz1"], example["sentence_quiz2"]],
        }
        all_lines.append(json.dumps(out_dict)+"\n")

    with open(os.path.join(out_dir, jsonl_filename), "w") as fout:
        fout.writelines(all_lines)

# fewshot data
story_ids = {
    "0": [1341, 259, 574, 6, 1523, 1864, 923, 576, 1620, 1248, 1029, 156, 47, 1036, 1122, 1054, 621, 839, 1193, 1088, 1490, 1069, 1265, 881, 1108, 53, 1780, 1294, 1291, 30, 1078, 85],
    "1": [486, 56, 1079, 102, 1641, 1657, 1229, 824, 1057, 1644, 281, 108, 1645, 1213, 120, 535, 1355, 1086, 534, 1183, 1784, 1330, 161, 705, 1553, 1416, 1378, 480, 1256, 1460, 1536, 572],
    "32": [1537, 1145, 1277, 1463, 1319, 1690, 1047, 56, 1730, 1498, 1229, 1159, 1552, 537, 1577, 1193, 591, 766, 1851, 1404, 5, 1365, 1657, 1233, 664, 70, 983, 999, 1037, 701, 1462, 911],
    "42": [305, 1175, 1702, 1837, 1047, 544, 1023, 582, 1210, 962, 1055, 948, 1034, 479, 322, 462, 352, 1792, 1347, 904, 298, 1709, 712, 65, 1274, 366, 575, 1106, 383, 464, 1558, 845],
    "1024": [143, 1289, 784, 436, 356, 405, 271, 21, 1746, 1178, 744, 295, 1770, 1745, 768, 359, 875, 780, 1778, 1298, 992, 553, 1740, 457, 131, 623, 871, 1783, 584, 1848, 1361, 1455]
}

os.makedirs("../data_fewshot/story_cloze/16_shot", exist_ok=True)
os.makedirs("../data_fewshot/story_cloze/32_shot", exist_ok=True)

for key, value in story_ids.items():
    filename = "{}_seed.jsonl".format(key)

    examples = [dict(dataset["validation"][idx]) for idx in value]
    for i in range(len(examples)):
        examples[i]["label"] = examples[i]["answer_right_ending"] - 1
        examples[i]["idx"] = value[i]


    _16_shot_examples = examples[:16]

    with open(os.path.join("../data_fewshot/story_cloze2/16_shot", filename), "w") as fout:
        for example in _16_shot_examples:
            fout.write(json.dumps(example) + "\n")
    with open(os.path.join("../data_fewshot/story_cloze2/32_shot", filename), "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")
        
