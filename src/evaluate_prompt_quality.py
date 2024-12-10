import csv
import random  
import sys
import yaml
from utils import Struct, style
from vanilla_rag import run_rag


def filter_bad_prompts(path_to_data):
    origins = []
    batch = []
    lines = []
    total_errors = 0
    with open(file=path_to_data, mode='r') as infile:
        tsv_reader = csv.reader(infile, quotechar='"', delimiter='\t', skipinitialspace=True)    

        for row in tsv_reader:
            if row:  # handle empty rows/new line spacing
                try:
                    batch.append(eval(row[1])['prompt'])  # extract prompt
                    origins.append(row[-1])  # extract original file for Q&A pair
                    lines.append(row)   # We'll want the whole row for shuffling later

                except SyntaxError:
                    total_errors += 1
                    print(f"{style.RED}WARNING: Row {row[0]} is a broken prompt probably due to context length limits of the LLM.")
                    print(f"{style.YELLOW}Total Failed Prompts: {total_errors}{style.RESET}")

    print(f"Total valid prompt triples: {style.GREEN}{len(batch)} of {len(batch)+total_errors}{style.RESET}. ({style.RED if total_errors else style.RESET}{total_errors} failed{style.RESET})")
    return batch, origins, lines

path_to_data = "/path/to/your.tsv"
batch, origins, lines = filter_bad_prompts(path_to_data=path_to_data)
out_file = "output_5_complete.tsv"


with open(file=out_file, mode='w') as outf:
    for l in lines:
         outf.write("\t".join(l))
         outf.write('\n\n')

if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)
    config.question_strings = batch

    good_prompts = "good_prompts.tsv"
    bad_prompts = "bad_prompts.tsv"
    outs = run_rag(config)
    first_prompt = 0
    not_first_prompt = 0
    no_prompt = 0
    for i in range(len(outs)):
        source_files = sorted([(node.dict()['score'], node.metadata['file_name']) for node in outs[i][0].source_nodes], reverse=True)
        for rank, score_file in enumerate(source_files):
            if score_file[1] in origins[i]:
                retrieved = True
                if int(rank) == int(0):
                    first_prompt += 1
                else:
                    not_first_prompt += 1
                with open(file=good_prompts, mode='a') as good_outs:
                    good_outs.write(f'{rank}\t{score_file[0]}\t{score_file[1]}\t{outs[i]}\n')  
                print(f"{style.GREEN}{first_prompt}\t{style.YELLOW}{not_first_prompt}\t{style.RED}{no_prompt}{style.RESET}")
                break
            if int(rank) == int(config.top_k_chunks):
                no_prompt += 1
                with open(file=bad_prompts, mode='a') as bad_outs:
                    bad_outs.write(f'{score_file[0]}\t{outs[i]}\n')  
            retrieved  = False
