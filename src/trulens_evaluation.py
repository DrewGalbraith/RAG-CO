from trulens_eval.feedback.provider.openai import OpenAI
from typing import List

def trulens_evaluation(config, prompts:List[str], contexts:List[str], answers:List[str])->List[dict]:

    client = OpenAI(base_url='http://localhost:11434/v1/', model_engine="llama3", api_key='ollama')  # Local host is for Ollama; Key is required but will not be used
    
    all_results = []
    
    for i, prompt in enumerate(prompts):
        results = dict()
        if config.run_context_relevance:
            results["context_relevance_results"] = client.qs_relevance_with_cot_reasons(prompt, contexts[i])
        # if config.run_groundedness:  # Isn't implemented, unless it's called something else.
            # results["groundedness_results"] = (Feedback(client.groundedness_measure_with_cot_reasons).on(context).on_output())
        if config.run_relevance:
            results["relevance_results"] = client.relevance(prompt=prompt, response=answers[i])
        all_results.append(results)

    return all_results



# print(dir(client))

# >>> [ 'coherence', 'coherence_with_cot_reasons', 'comprehensiveness_with_cot_reasons', 'copy', 'correctness', 'correctness_with_cot_reasons',
# 'dict', 'endpoint','generate_score_and_reasons', 'json', 'model_agreement', 'model_computed_fields', 'model_config', 'model_construct',
# 'model_copy', 'model_dump', 'model_dump_json', 'model_engine', 'model_extra', 'model_fields', 'model_fields_set', 'model_json_schema',
# 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate','model_validate_json', 'model_validate_strings',
# 'qs_relevance', 'qs_relevance_with_cot_reasons', 'relevance', 'relevance_with_cot_reasons','validate']
