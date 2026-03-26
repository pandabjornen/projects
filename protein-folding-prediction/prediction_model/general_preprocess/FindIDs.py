from rcsbapi.search import search_attributes as attrs
import random
def SearchForIDsForSingleChains(min_length,max_length, max_resolution, nrOfchains, downloadedIds) :
    """
    NOTE: Resolution is in Ångströms
    """
    q_len_min     = attrs.entity_poly.rcsb_sample_sequence_length >= min_length
    q_len_max     = attrs.entity_poly.rcsb_sample_sequence_length <= max_length
    q_chain_count = attrs.rcsb_entry_info.polymer_entity_count == 1
    q_resolution  = attrs.rcsb_entry_info.resolution_combined <= max_resolution
    # q_poly = attrs.entity_poly.type == "polypeptide(L)"
    q_is_protein = attrs.entity_poly.rcsb_entity_polymer_type == "Protein"
    q_single_polymer_instance = attrs.rcsb_entry_info.deposited_polymer_entity_instance_count == 1
    q_has_one_unique_protein_type = attrs.rcsb_entry_info.polymer_entity_count_protein == 1
    query = (q_len_min & q_len_max & q_chain_count & q_resolution & q_is_protein & q_single_polymer_instance & q_has_one_unique_protein_type)

    ids = []


    all_ids = list(query())
    random.shuffle(all_ids) #NOTE: need to random shuffle to not get alphabeticcally ordered!

    uniqueAllIds = [i for i in all_ids if i not in downloadedIds]
    ids = uniqueAllIds[:nrOfchains]

    
    return ids



