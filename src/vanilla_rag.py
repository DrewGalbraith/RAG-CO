import os
import sys
import yaml

from llama_index.core import load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from pathlib import Path
from trulens_evaluation import trulens_evaluation
from typing import List, Tuple, Union
from utils import get_context_window, Struct, style
os.environ["CURL_CA_BUNDLE"] = ""  # Attempt to dodge OpenSSL errors

def query_rag(config, settings):

    # Load/create index
    if config.load_index_from[0] and not settings.is_baseline:
        print(f"\n{style.YELLOW}Loading index from:{style.RESET} {config.load_index_from[0]}...")
        storage_context = StorageContext.from_defaults(persist_dir=config.load_index_from[0])
        index = load_index_from_storage(storage_context, index_id=config.load_index_from[1])
    else:
        print(f"\n{style.YELLOW}Creating new index '{style.CYAN}{config.save_new_index[1]}{style.RESET}'...")
        index = VectorStoreIndex.from_documents(settings.documents, show_progress=True)
        if config.save_new_index[0]:  # Save index to disk
            index.set_index_id(config.save_new_index[1])
            index.storage_context.persist(config.save_new_index[2])

    # Query model
    print(f"{style.GREEN}Index loaded. {style.YELLOW}Querying model...{style.RESET}")
    query_engine = index.as_query_engine(similarity_top_k=config.top_k_chunks)  # TODO: Where does this arg come from/get used?
    responses = [query_engine.query(q) for q in settings.queries]
    contexts = [" ".join([node.dict()['node']['text'] for node in response.source_nodes]) for response in responses]
    
    return responses, contexts

def run_rag(config)-> Tuple[List, List]:

    # Set the query(s)
    Settings.queries = config.question_strings  # "How fast was Darryl Anderson driving when he crashed?" # 140 mph
    # Define embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.embed_path_dir, device=config.device)
    # Define LM
    Settings.llm = Ollama(model="llama3", request_timeout=120.0, additional_kwargs={'device':config.device})
    # Get context window for provided Ollama or HuggingFace LM
    Settings.context_window = get_context_window(Settings.llm)
    # Set number of output tokens
    Settings.num_output = config.gen_len if config.gen_len else 100
    # Set maximum input size
    Settings.max_input_size = Settings.context_window-Settings.num_output if Settings.context_window else 1024
    # Set maximum chunk overlap
    Settings.max_chunk_overlap = config.overlap
   
    # Run RAG
    exclude = Path(config.data_path).glob(config.exclude_pattern if config.exclude_pattern else "*.dont_exclude")
    Settings.documents = SimpleDirectoryReader(config.data_path, exclude=exclude).load_data()
    answers, cxts = query_rag(config, Settings)
    
    # Run RAG eval
    if config.run_trulens:
        print(f"{style.GREEN}Model queried. {style.YELLOW}Evaluating results...{style.RESET}")
        eval_results = trulens_evaluation(config, prompts=Settings.queries, contexts=cxts, answers=answers)
    else:
        eval_results = ["No evaluation run"]*len(answers)

    if not config.run_llm_baseline:
        no_rag_answers = ["No baseline run"]*len(answers)
    else:  # Run baseline (i.e., LLM w/out RAG)
        print(f"{style.GREEN}Results evaluated. {style.YELLOW}Querying LLM baseline...{style.RESET}\n")
        empty_file_path = os.path.join(os.getcwd(), "empty.txt")
        with open(empty_file_path, mode='w') as empty:
            empty.write("")
        Settings.documents = SimpleDirectoryReader(input_files=[empty_file_path]).load_data()
        os.remove(empty_file_path)
        no_rag_answers, _ = query_rag(config, Settings)

    return list(zip(answers, cxts, eval_results, no_rag_answers))


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Struct(**config)

    outs = run_rag(config)

    responses_path = "path/to/outfile.tsv"
   
    with open(responses_path, 'w') as outfile:

        outfile.write("Question\tResponse\tContext\n")

        for i in range(len(config.question_strings)):

            question = config.question_strings[i]
            answer = outs[i][0]
            cxt = outs[i][1]

            line = f"{question}\t{answer}\t{cxt}\n"
            print(line)
            outfile.write(line)