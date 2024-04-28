import json
import os
import time
import argparse


def bool_flag(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, "w") as fout:
        json.dump(param_dict, fout, indent=4)


def freeze_net(tencoder, last_unfreeze_layer=0):
    for p in tencoder.parameters():
        p.requires_grad = False
    # エンコーダーの層の数を取得
    module = tencoder.module
    total_layers = len(module.encoder.layer)

    # 指定された数の最後の層以外の層を凍結
    for i, layer in enumerate(module.encoder.layer):
        if i < total_layers - last_unfreeze_layer:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch


# ============================================================================
# COMPILE

import torch
import numpy as np


def remove_padding_from_nodes(concept_ids, node_type_ids, adj_lengths):
    """
    Remove the padding from the node lists and assert various conditions.

    Args:
    - concept_ids (tensor of shape (batch_size, max_nodes)): The concept IDs of nodes in each graph.
    - node_type_ids (tensor of shape (batch_size, max_nodes)): The type IDs of nodes in each graph.
    - adj_lengths (list of length batch_size): The number of edges in each graph.

    Returns:
    - concept_id_list (list of tensors): List of concept IDs for each graph after removing padding.
    - node_type_list (list of tensors): List of node types for each graph after removing padding.
    """

    # Initialize empty lists to store the unpadded concept IDs and node types for each graph.
    concept_id_list = []
    node_type_list = []

    # Iterate through each graph in the batch to remove padding.
    for i in range(len(concept_ids)):
        # Find the indices where node_type_ids is not 2 (i.e., not padding).
        non_pad_indices = node_type_ids[i] != 2

        # Remove padding from concept_ids and node_type_ids using the found indices.
        unpadded_concept_ids = concept_ids[i][non_pad_indices]
        unpadded_node_type_ids = node_type_ids[i][non_pad_indices]

        # Add the unpadded tensors to their respective lists.
        concept_id_list.append(unpadded_concept_ids)
        node_type_list.append(unpadded_node_type_ids)

        # Assert that the unpadded concept IDs only contain 1s in the padded locations.
        assert torch.all(unpadded_concept_ids != 1), "Unexpected 1s found in unpadded concept IDs."

        # Assert that the lengths of unpadded concept IDs and node type IDs are the same as adj_lengths[i].
        assert len(unpadded_concept_ids) == adj_lengths[i], "Length mismatch between unpadded concept IDs and adj_lengths."
        assert len(unpadded_node_type_ids) == adj_lengths[i], "Length mismatch between unpadded node types and adj_lengths."

    return concept_id_list, node_type_list


def extract_source_and_target_fixed(node_type_list):
    each_source = []
    each_target = []
    for i, node_type_tensor in enumerate(node_type_list):
        source_indices = (node_type_tensor == 1).nonzero(as_tuple=True)[0]
        target_indices = (node_type_tensor == 4).nonzero(as_tuple=True)[0]
        assert len(source_indices) == 1, f"Expected one source node in graph {i}, found {len(source_indices)}"
        assert len(target_indices) == 1, f"Expected one target node in graph {i}, found {len(target_indices)}"
        each_source.append(source_indices.item())
        each_target.append(target_indices.item())
    return each_source, each_target


def update_edge_ST_indices(edge_index_list, each_source, each_target, graph_sizes):
    # Initialize node_count and lists to store the updated edge_index, each_source, and each_target
    node_count = 0
    updated_edge_index_list = []
    updated_each_source = []
    updated_each_target = []

    # Loop through each graph_size to update the indices
    for i, size in enumerate(graph_sizes):
        # Update edge_index by adding node_count to both subject and object nodes
        updated_edge_index = edge_index_list[i] + node_count
        # Update each_source and each_target by adding node_count
        updated_source = each_source[i] + node_count
        updated_target = each_target[i] + node_count
        # Append the updated tensors to the respective lists
        updated_edge_index_list.append(updated_edge_index)
        updated_each_source.append(updated_source)
        updated_each_target.append(updated_target)
        # Increment node_count by the graph_size
        node_count += size

    return updated_edge_index_list, updated_each_source, updated_each_target


def make_sparse_tensor(source_map, target_map, total_num_nodes, total_num_edges):
    # Create comb_edge_range tensor
    comb_edge_range = torch.arange(total_num_edges).to(source_map.device)

    # Create total_edge tensor
    total_edge = torch.stack([target_map, comb_edge_range], dim=0)

    # Create e2n_sp sparse tensor
    e2n_value = torch.FloatTensor(torch.ones(total_edge.shape[1])).to(source_map.device)
    e2n_sp = torch.sparse.FloatTensor(total_edge, e2n_value, torch.Size([total_num_nodes, total_num_edges]))

    # Create total_edge2 tensor
    total_edge2 = torch.stack([source_map, comb_edge_range], dim=0)

    # Create e2n_sp2 sparse tensor
    e2n_sp2 = torch.sparse.FloatTensor(total_edge2, e2n_value, torch.Size([total_num_nodes, total_num_edges]))
    return e2n_sp, e2n_sp2


def get_query_edge_indices(edge_index_unite, source_node, target_node):
    # Initialize a list to store the indices of query edges in the combined edge list
    query_edge_indices = []

    # Loop through each query edge to find its index in the combined edge list
    for s, t in zip(source_node, target_node):
        match_indices = ((edge_index_unite[0] == s) & (edge_index_unite[1] == t)).nonzero(as_tuple=True)[0]
        assert len(match_indices) == 1, f"Expected one matching edge for source {s} and target {t}, found {len(match_indices)}"
        query_edge_indices.append(match_indices.item())

    # Convert the list to a tensor
    query_edge_indices = torch.tensor(query_edge_indices)

    # Assert that the number of query edges found matches the batch size
    assert len(query_edge_indices) == len(source_node), "Mismatch between the number of query edges found and the batch size"

    return query_edge_indices


def gather_lm_embeddings(entid2lmemb, concept_id_list):
    """
    Gather the LM embeddings for each node in each graph in the batch.

    Args:
    - entid2lmemb (tensor of shape (batch_size, concept_num, dim_sent)): The LM embeddings for all concepts in all graphs.
    - concept_id_list (list of tensors): List of concept IDs for each graph after removing padding.

    Returns:
    - LMent_list (list of tensors): List of LM embeddings for each graph.
    """

    LMent_list = []

    for i, concept_ids in enumerate(concept_id_list):
        lm_embeddings = entid2lmemb[i].gather(0, concept_ids.unsqueeze(-1).expand(-1, entid2lmemb.shape[-1]))
        is_zero_vector = (lm_embeddings == 0).all(dim=1)  #!それぞれのコンセプトidで、lm埋め込みが0かどうか
        try:
            assert not is_zero_vector[1:].any()  #! 全てのコンセプトidで、埋め込みが非0ベクトルか
        except AssertionError:
            print("batch_num:", i)
            print("lm_embedding len", lm_embeddings.size(0))
            print(f"=concepts num:{concept_ids.size(0)}", print(concept_ids))
            true_indices = is_zero_vector.nonzero(as_tuple=True)[0]
            print(f"non_zero indices: {true_indices}")
            print("lm_embeddings(first10 of 1024):\n ", lm_embeddings[:, :10])
            raise AssertionError
        LMent_list.append(lm_embeddings)

    return LMent_list


def convert_qagnn2compile_input(concept_ids, node_type_ids, adj_lengths, edge_index, edge_type, entid2lmemb):
    concept_id_list, node_type_list = remove_padding_from_nodes(concept_ids, node_type_ids, adj_lengths)
    each_source, each_target = extract_source_and_target_fixed(node_type_list)
    # Convert the tensor to a Python list
    graph_sizes = [int(t.item()) for t in adj_lengths]
    # Get the lengths of each tensor in edge_type_list
    list_num_edges = [len(t) for t in edge_type]
    # Calculate total_num_nodes and total_num_edges for the batch by summing up the individual counts
    total_num_nodes = sum(graph_sizes)
    total_num_edges = sum(list_num_edges)
    concept_ids = torch.cat(concept_id_list)
    node_types = torch.cat(node_type_list)
    edge_types = torch.cat(edge_type)

    edge_index, each_source, each_target = update_edge_ST_indices(edge_index, each_source, each_target, graph_sizes)
    source_node = np.array(each_source)
    target_node = np.array(each_target)
    edge_index_unite = torch.cat(edge_index, dim=1)
    source_map = edge_index_unite[0, :]
    target_map = edge_index_unite[1, :]

    query_edge_indices = get_query_edge_indices(edge_index_unite, source_node, target_node)

    e2n_sp, e2n_sp2 = make_sparse_tensor(source_map, target_map, total_num_nodes, total_num_edges)

    if entid2lmemb is not None:
        LMentemb_list = gather_lm_embeddings(entid2lmemb, concept_id_list)
        LMnodesemb = torch.cat(LMentemb_list)
    else:
        LMnodesemb = None

    return [graph_sizes, list_num_edges, concept_ids, node_types, LMnodesemb, source_node, target_node, query_edge_indices, source_map, target_map, edge_types, e2n_sp, e2n_sp2]


def extract_query_rep(node_type_ids, concept_ids, entid2lmemb):
    """
    Args:
        node_type_ids : 2d tensor of (batch_size, node_num):
        concept_ids : 2d tensor of (batch_size, node_num):
        entid2lmemb : 3d tensor of (batch_size, concept_num, sent_dim):

    Returns:
        2d tensor (batch_size, sent_dim*2):
    """
    # Find source and target indices
    source_indices = (node_type_ids == 1).nonzero(as_tuple=True)[1]
    target_indices = (node_type_ids == 4).nonzero(as_tuple=True)[1]

    # Ensure only one source and target per batch
    assert source_indices.shape[0] == node_type_ids.size(0), "Expected one source node per graph"
    #!バッチ内の、タイプ1のノードの合計がバッチサイズに一致することの確認でしか無い。
    assert target_indices.shape[0] == node_type_ids.size(0), "Expected one target node per graph"

    # Convert to tensors and unsqueeze to shape (batch_size, 1)
    each_source = source_indices  # .unsqueeze(-1)
    each_target = target_indices  # .unsqueeze(-1)

    source_ids = torch.gather(concept_ids, 1, each_source.unsqueeze(-1)).squeeze()
    target_ids = torch.gather(concept_ids, 1, each_target.unsqueeze(-1)).squeeze()

    # Fetch the corresponding embeddings using direct indexing
    try:
        source_emb = entid2lmemb[range(len(source_ids)), source_ids]  # torch.gather(entid2lmemb,1,source_ids)
        target_emb = entid2lmemb[range(len(target_ids)), target_ids]  # torch.gather(entid2lmemb,1,target_ids)
    except TypeError:
        print(node_type_ids)
        print(each_source)
        print(concept_ids)
        print(torch.gather(concept_ids, 1, each_source.unsqueeze(-1)))
        raise TypeError

    is_zero_vector = (source_emb == 0).all(dim=1)
    assert not is_zero_vector.any()
    is_zero_vector = (target_emb == 0).all(dim=1)
    assert not is_zero_vector.any()

    # Concatenate the embeddings
    query_rep = torch.cat([source_emb, target_emb], dim=1)
    return query_rep
