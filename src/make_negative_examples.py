import csv
import random  
  
def shuffle_dict_values(pos_in_filepath:str, out_filepath:str, write_positives:bool=False)-> None:
    """
    Function to change positive examples of RAG-Triad qualities to negative ones. Negative examples are made 
    by shuffling values between corresponding keys of different entries. E.g., swapping values of the 1st and 
    70th 'context' keys to create two negative examples of context relevance.

    pos_in_filepath (str): A path to a tsv of dictionaries with keys prompt, context, and answer with their 
        corresponding values output by a langauage model.

    out_filepath (str): Simply a string to indicate where on disk to write a given file

    write_positives (bool): Whether or not to skip the shuffling phase, essentially writing out a similarly 
        formatted list of positive examples from the positive input example dictionaries.
    """

    dict_list = []
    with open(file=pos_in_filepath, mode='r') as infile:
        tsv_reader = csv.reader(infile, quotechar='"', delimiter='\t', skipinitialspace=True)    

        for row in tsv_reader:
            if row:
                dict_row = eval(row[1])
                dict_list.append(dict_row)

    # Calculate the size of each part  
    keys = dict_list[0].keys()
    part_size = len(dict_list) // len(keys)

    if not write_positives:  # writing negative examples to file, so we must shuffle to create negatives
        # Shuffle the values of each key in its corresponding part  
        for i, key in enumerate(keys):  
            # Determine the part of the list to shuffle  
            start_index = i * part_size  
            end_index = start_index + part_size if i < len(keys) - 1 else len(dict_list)  
            part_to_shuffle = dict_list[start_index:end_index]  
            
            # Extract the values for the current key to shuffle  
            values_to_shuffle = [d[key] for d in part_to_shuffle]  
            random.shuffle(values_to_shuffle)  
            
            # Reassign the shuffled values back to the dictionaries  
            for j, value in enumerate(values_to_shuffle):  
                part_to_shuffle[j][key] = value
    
    with open(file=out_filepath, mode='w') as outfile:
        for d in dict_list:
            outfile.write(str(d))
            outfile.write("\n\n")


postives_filepath = "path/to/your/positive/RAG/Triad/examples.tsv"
negatives_filepath = "path/to/write/negative/RAG/Triad/examples.tsv"

shuffle_dict_values(pos_in_filepath=postives_filepath, out_filepath=negatives_filepath)

