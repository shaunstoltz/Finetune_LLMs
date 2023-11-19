import os
import json

import argparse

import datetime

import time

results = {
    "start": 0,
    "number": 0,
    "random": False,
    "temperature": 0.0,
    "top_p": 0.0,
    "no_correct": 0,
    "unknow_answers": [],
    "outcomes": [{
        "ground_truth": "",
        "gpt_interaction": {
            "api_call": {
                "prompt": {},
                "response": {}
            }
        }
    }]
}

# vLLM inference #####################################################

from mii import pipeline

# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(model="/mnt/md1/shaun/repos/ai/models/yi6bfulldatasetfinalcp/", trust_remote_code=True)

# outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input file path or directory")
    parser.add_argument("-id", "--input-dataset", type=str, required=True, help="Input data set")
    parser.add_argument("-tg", "--input-dataset-tag", type=str, required=False, default="None", help="Input data set taq for additional information")
    parser.add_argument("-o", "--output-dir", type=str, required=False, help="Output file dir")
    parser.add_argument("-m", "--model-name", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("-tp", "--top-p", type=float, default=0, help="Sampling Top P")
    parser.add_argument("-s", "--start", type=float, default=0, help="Data index to start from")
    parser.add_argument("-n", "--number", type=float, default=100, help="Number of data elements to iterate through")
    parser.add_argument("-pro", "--proteus", type=bool, default=False, help="Use Proteus prompt")
    parser.add_argument("-pond", "--ponder", type=bool, default=False, help="Use Ponder prompt")
    parser.add_argument("-pr", "--pre", type=str, default="", help="Use pre prompts")
    parser.add_argument("-po", "--post", type=str, default="", help="Use post prompts")
    parser.add_argument("-r", "--random", type=bool, default=False, help="Use random selection from question set based on start and number")

    return parser.parse_args()

def extract_answer(rawanswer):
    extracted_answer = {
        "answer": ""
    }

    if "{'answer':" in rawanswer:
        start_index = rawanswer.index("{'answer':")
        extracted_string = rawanswer[start_index:]
        end_index = extracted_string.index("}")
        extracted_answer = extracted_string[:end_index+1]


    if '{"answer":' in rawanswer:
        start_index = rawanswer.index('{"answer":')
        extracted_string = rawanswer[start_index:]
        end_index = extracted_string.index("}")
        extracted_answer = extracted_string[:end_index+1]
    
    if '{"\nanswer":' in rawanswer:
        start_index = rawanswer.index('{"answer":')
        extracted_string = rawanswer[start_index:]
        end_index = extracted_string.index("\n}")
        extracted_answer = extracted_string[:end_index+1] 

    return extracted_answer

def get_MATH_data_set_array(directory):

    json_files = [pos_json for pos_json in os.listdir(directory) if pos_json.endswith('.json')]
    return json_files


if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    args = parse_args()

    #api_key = args.api_key
    model_name = args.model_name
    temperature = args.temperature
    top_p = args.top_p
    input_file = args.input_file
    input_dataset = args.input_dataset
    input_dataset_tag = args.input_dataset_tag
    data = []
    start = args.start
    number = args.number
    proteus = args.proteus
    ponder = args.ponder
    ponder_prompt = ""
    proteus_prompt = ""
    pre = args.pre
    pre_prompt = ""
    post = args.post
    post_prompt = ""
    prompt = ""

    if input_dataset == "MATH":
        files = get_MATH_data_set_array(input_file)
        for file in files:
            file = input_file + file
            with open(file) as f:
                data_in_file = json.load(f)
                answer_start_index = data_in_file['solution'].index('\\boxed{')
                answer_start = data_in_file['solution'][answer_start_index + 7:]
                anser_end_index = answer_start.index('}')
                answer = answer_start[:anser_end_index]
                data_in_file['answer'] = answer
                data_in_file['question'] = data_in_file['problem']
                data_in_file['file_name'] = file
                data.append(data_in_file)
    
    if input_dataset == "multiarith" or input_dataset == "addsub" or input_dataset == "singleeq":
        if input_file is not None:
            with open(input_file) as f:
                data_raw = json.load(f)
                for elem in data_raw:
                    elem['answer'] = elem['lSolutions']
                    elem['question'] = elem['sQuestion']
                    data.append(elem)

    if input_dataset == "aqua":
        if input_file is not None:
            with open(input_file) as f:
                data_raw = [json.loads(line) for line in f]
                for elem in data_raw:
                    elem['answer'] = elem['correct']
                    elem['question'] = elem['question']
                    data.append(elem)

    if input_dataset == "svamp":
        if input_file is not None:
            with open(input_file) as f:
                data_raw = json.load(f)
                for elem in data_raw:
                    elem['answer'] = elem['Answer']
                    elem['question'] = elem['Body']
                    elem['question'] = elem['question'] + " " + elem['Question']
                    data.append(elem)

    if input_dataset == "gsm8k":
        if input_file is not None:
            with open(input_file) as f:
                data = [json.loads(line) for line in f]   
                 
    data = data[int(start):int(start + number)]

    pipe = pipeline(model_name)
    #llm = LLM(model=model_name, trust_remote_code=True, dtype="float16")
    j = 0
    for i in data:
        question = i['question']
        answer = i['answer']

        print(j, " Ground truth answer: ", answer)

        pre_prompt = ""
     
        if post != "": 
            post_prompt = json.load(open(f"{MAD_path}/code/utils/{post}", "r"))['prompt']         
        
        prompt = pre_prompt + question + post_prompt


        # Get answer
        # (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
        #outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
        output = pipe(prompt, max_new_tokens=1024)

        # Decode output & print it
        #print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        response = output

        print(response)

        result = {
            "ground_truth": answer,
            "full_dataset_item": i,
            "gpt_interaction": {
                "api_call": {
                    "prompt": prompt,
                    "response": response
                }
            }
        }
    

        results['outcomes'].append(result)

        response_answer = extract_answer(response)
        try:
            if str(answer) in str(response_answer):
                results['no_correct'] += 1
            else:
                results["unknow_answers"].append(j)
        except Exception as e:
            results["unknow_answers"].append(j)
            print(e)
        print(response)

        j += 1

    results['start'] = start
    results['number'] = number
    results["temperature"] = temperature
    results["top_p"] = top_p


    print(results)

    presentDate = datetime.datetime.now()
    unix_timestamp = datetime.datetime.timestamp(presentDate)*1000
    model_name = model_name.replace("/","_")
    file_path = f"{MAD_path}/data/output/dataset_{input_dataset}_tag_{input_dataset_tag}_start_{start}_number_{number}_post_{post}_model_{model_name}_{unix_timestamp}.json"
    # file = open(file_path, 'w')
    # file.write(json.dumps(results))
    # file.close()
    
































