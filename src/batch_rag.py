import os
import random
import sys
import yaml

from llama_index.core import load_index_from_storage, PromptTemplate, Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Union
from utils import get_context_window, Struct, style

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    def custom_query(self, query_str, retriever, llm, prompt, top_k=3):
        nodes = retriever.retrieve(query_str)
        options_list = [i for i in range(top_k)]
        context_str = "\n\n".join([str(i) + n.node.get_content() for i, n in enumerate(nodes)])
        response = llm.complete(
            prompt.format(context_str=context_str, query=query_str, top_k=options_list),)
        return response, nodes


def query_rag(config, settings):
    # Load/create index
    if config.load_index_from[0] and not settings.is_baseline:
        print(f"\n{style.YELLOW}Loading index from:{style.CYAN} {config.load_index_from[0]}{style.YELLOW}...{style.RESET}")
        storage_context = StorageContext.from_defaults(persist_dir=config.load_index_from[0])
        index = load_index_from_storage(storage_context, index_id=config.load_index_from[1])
    else:
        saving = "but not saving" if settings.is_baseline else "and saving"
        print(f"\n{style.YELLOW}Creating {saving} new index '{style.CYAN}{config.save_new_index[1]}{style.RESET}'...")
        index = VectorStoreIndex.from_documents(settings.documents, show_progress=True)
        if config.save_new_index[0] and not settings.is_baseline:  # Save index to disk
            index.set_index_id(config.save_new_index[1])
            index.storage_context.persist(config.save_new_index[2])
    
    # Create Retrievers
    retriever = index.as_retriever(similarity_top_k=config.top_k_chunks)  # TODO: Where does this arg come from/get used?
    retriever_only = index.as_retriever(similarity_top_k=1)  # TODO: Where does this arg come from/get used?
    
    # Create Query Engine
    prompt = PromptTemplate(
        "Articles are provided below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the articles and not prior knowledge, "
        "which article best relates to the provided keywords/description? \n"
        "Key words/Description: {query}\n"
        "Respond with a single integer from this list: {top_k}.\n" 
        "Answer: ")
    
    query_engine = RAGStringQueryEngine()
    
    print(f"{style.GREEN}Index loaded. {style.YELLOW}Querying model...{style.RESET}")
    # Sumbit Query
    if config.batched:
        if settings.is_baseline:
            prefix = "\nIGNORE ALL previous instructions and any provided context. Repeat, ignore them all. Just answer the following question:\n "
            settings.queries = [prefix+q for q in settings.queries]
        if len(settings.queries)<100:
            responses = [query_engine.query(q) for q in tqdm(settings.queries)]
            contexts = ["\n\n".join([node.dict()['node']['text'] for node in response.source_nodes]) for response in responses]
        else:
            responses = []
            contexts = []

            responses_path = config.results_out_path
            with open(responses_path, 'w') as outfile:
                outfile.write("Question\tResponse\tContext\n")

                for q in tqdm(settings.queries):
                    try:
                        response = query_engine.query(q)
                        context = "\n\n".join([node.dict()['node']['text'] for node in response.source_nodes])
                        responses.append(response)
                        contexts.append(context)
                        line = f"{q}\t{response}\t{context}\n"
                    except TypeError:
                        print("ERROR")
                        line = f"{q}\tERROR\tERROR-No context retrieved\n"
                    outfile.write(line)
    else:
        # Populate random system
        print("Grabbing all nodes for random system.\n", flush=True)
        all_nodes = [i.text for i in list(index.docstore.docs.values())]

        questions = []
        responses = []
        contexts = []
        good_examples = []
        settings.queries = input(f"\n{style.YELLOW}Enter question or type pound (#) to be done:{style.RESET} ")
        
        while settings.queries != '#':
            modes = ['llm', 'random', 'retriever']
            random.shuffle(modes)

            for mode in modes:
                settings.mode = mode
                print(settings.mode, flush=True)

                if settings.mode == 'llm':
                    questions.append(f"LLM: {settings.queries}")
                    ans, nodes = query_engine.custom_query(settings.queries, retriever=retriever, llm=settings.llm, prompt=prompt, top_k=config.top_k_chunks)
                    print(ans, flush=True)
                    ctx = [node.dict()['node']['text'] for node in nodes]
                    try:
                        article_num = int(ans.text)
                        article = ctx[article_num - 1]
                        print(f"Article: {article}")
                    except ValueError:
                        article = '\n\n'.join(ctx)
                    contexts.append(article)

                if settings.mode == 'random':
                    questions.append(f"Random: {settings.queries}")
                    ans = random.choice(all_nodes)
                    print(ans, flush=True)
                    contexts.append(ans)

                if settings.mode == 'retriever':
                    questions.append(f"Retriever: {settings.queries}")
                    nodes = retriever_only.retrieve(settings.queries)  # only retrieves 1 node instead of top_k
                    ans = "\n".join([str(i) + n.node.get_content() for i, n in enumerate(nodes)])
                    contexts.append(ans)
                    print(ans, flush=True)
                
                interest_rating = input(f"\n{style.BLUE}On a scale of 1-10, how {style.CYAN}interesting{style.BLUE} was this article?:{style.RESET} ")
                relevance_rating = input(f"{style.BLUE}On a scale of 1-10, how {style.CYAN}relevant{style.BLUE} was this article to your query?:{style.RESET} ")

                # Convert ratings (str -> float)
                interest_rating_int = float(interest_rating)
                relevance_rating_int = float(relevance_rating)
                if interest_rating_int + relevance_rating_int >= 13:
                    good_examples.append(f"TEXT: {ans} RATING: {interest_rating_int + relevance_rating_int}")


                responses.append(ans)
            settings.queries = input(f"\n{style.YELLOW}Enter question or type pound (#) to be done:{style.RESET} ")
            
    return questions, responses, contexts

def run_rag(config)-> Tuple[List, List]:

    # Set the query(s)
    Settings.queries = config.question_strings  # "How fast was Darryl Anderson driving when he crashed?" # 140 mph
    # Define embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.embed_path_dir, cache_folder=config.embed_cache_folder, device=config.device)
    print("Embed loaded!\n", flush=True)
    # Define LLM
    Settings.llm = HuggingFaceLLM(model_name=config.llm, tokenizer_name=config.llm, device_map=config.device)
    print("llm loaded!\n", flush=True)
    ## Get context window for provided Ollama or HuggingFace LM
    # Settings.context_window = get_context_window(Settings.llm)
    # Set number of output tokens
    Settings.num_output = config.gen_len if config.gen_len else 1
    # Set maximum input size
    Settings.max_input_size = Settings.context_window - Settings.num_output if Settings.context_window else 1024
    # Set chunk size for index building
    Settings.chunk_size = config.chunk_size
    # Set maximum chunk overlap
    Settings.max_chunk_overlap = config.overlap
    Settings.is_baseline = False
   
    # Run RAG
    exclude = Path(config.data_path).glob(config.exclude_pattern if config.exclude_pattern else "*.dont_exclude")
    Settings.documents = SimpleDirectoryReader(config.data_path, exclude=exclude).load_data()
    questions, answers, cxts = query_rag(config, Settings)
    
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
        Settings.is_baseline = True
        no_rag_answers, _ = query_rag(config, Settings)

    return list(zip(questions, answers, cxts, eval_results, no_rag_answers))


if __name__ == "__main__":
    args = sys.argv
    config_path = args[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) 
    config = Struct(**config)
    outs = run_rag(config)

    responses_path = config.results_out_path
   
    with open(responses_path, 'w') as outfile:

        outfile.write("Question\tResponse\tBaseline\tContext\n")

        for out in outs:

            print(dir(out[1]))
            question = out[0]
            answer = out[1].response.strip().replace("\n", "")
            baseline = out[-1].response.strip().replace("\n", "")
            cxt = outs[2].strip()

            line = f"{question}\t{answer}\t{baseline}\t{cxt}\n"
            outfile.write(line)
            
#  <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='27a7732b-94e9-4735-86b9-b30d32b50d8c', node_type='1', metadata={}, hash='da9e6ed91872a26d73360d1a4f066bdd74bc23addd2de2d3fff0c2878ec00370')}, metadata_template='{key}: {value}', metadata_separator='\n', text='Does Mija still live in the Tolstraat? Can she keep the apartment?\r\nThe children are so excited that you are coming. Every time we talk about you, Joel says, "Oma and Opa come to my house pretty soon." Try to talk Erwin into coming in some of the same time that you are here, but I think he is too busy with his stores. This year alone, he wants to open more than 6 stores (10 altogether, as I understand it.)\r\nDavid\'s parents are already much better. Her operation seems to be successful and he feels a lot better also, but the doctors still have to do many tests--they haven\'t found the exact cause yet. If they get over these sicknesses, they might come here in May with John\'s tour.\r\nJohn is trying to organize an inexpensive tour and David would be the guide. That will also save a lot of money. David will organize the hotels, etc. here. He is so busy. The whole month of May he teaches 2 classes to a group of BYU students from Provo. That means he will teach every day 4-5 hours. It pays well in $$, so that is a nice bonus. Well, no more news. This last part I wrote in the car--that\'s why my handwriting is so uneven. Daag, greetings and much love from Frieda, David, Rivka, Daniel and Joel.\r\n13 February 1972 Letter from Erwin. I just read your letter and received your dresses 2 days ago. Thanks. Everything goes well here...Congratulations with your mother\'s 60th birthday. She now is 2x my age. We had a nice time in Haiti and we have nice tans. My business has become LTD and I have a new partner. Our contract is 25-30 pages long. In the future, we have to do normal business. The money you owe me I have not researched. I guess it is about right. The other sample shirts I\'m not sure about. We displayed two, but haven\'t sold them yet.', mimetype='text/plain', start_char_idx=1669088, end_char_idx=1670840, metadata_seperator='\n', text_template='{metadata_str}\n\n{content}')