import pickle
import os
import numpy as np
import torch
from transformers import OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, AutoTokenizer

try:
    from transformers import AlbertTokenizer
except:
    pass

import json
from tqdm import tqdm

GPT_SPECIAL_TOKENS = ["_start_", "_delimiter_", "_classify_"]


#!entity_mask (batch_size, concept_num, max_seq_len)
class MultiGPUSparseAdjDataBatchGenerator(object):
    def __init__(self, args, mode, device0, device1, batch_size, indexes, qids, labels, tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None, entity_mask=None):
        self.args = args
        self.mode = mode
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        # self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data
        self.entity_mask = entity_mask

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        if self.mode == "train" and self.args.drop_partial_batch:
            print("dropping partial batch")
            n = (n // bs) * bs
        elif self.mode == "train" and self.args.fill_partial_batch:
            print("filling partial batch")
            remain = n % bs
            if remain > 0:
                extra = np.random.choice(self.indexes[:-remain], size=(bs - remain), replace=False)
                self.indexes = torch.cat([self.indexes, torch.tensor(extra)])
                n = self.indexes.size(0)
                assert n % bs == 0

        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            if self.entity_mask is not None:
                batch_entity_mask = self._to_device(self.entity_mask[batch_indexes], self.device0)
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            edge_index_all, edge_type_all = self.adj_data
            # edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            # edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            if self.entity_mask is not None:
                yield tuple([(batch_entity_mask, batch_qids), batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index, edge_type])
            else:
                yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index, edge_type])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


def load_sparse_adj_data_with_contextnode(adj_pk_path, max_node_num, num_choice, args):
    cache_path = adj_pk_path + ".loaded_cache"
    use_cache = False

    if use_cache and not os.path.exists(cache_path):
        use_cache = False

    if use_cache:
        with open(cache_path, "rb") as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel = pickle.load(f)
    else:
        with open(adj_pk_path, "rb") as fin:
            adj_concept_pairs = pickle.load(fin)

        n_samples = len(adj_concept_pairs)  # this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)  # default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)

        adj_lengths_ori = adj_lengths.clone()
        for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc="loading adj matrices"):
            adj, concepts, qm, am, cid2score = _data["adj"], _data["concepts"], _data["qmask"], _data["amask"], _data["cid2score"]
            # adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            # concepts: np.array(num_nodes, ), where entry is concept id
            # qm: np.array(num_nodes, ), where entry is True/False
            # am: np.array(num_nodes, ), where entry is True/False
            assert len(concepts) == len(set(concepts))
            qam = qm | am
            # sanity check: should be T,..,T,F,F,..F
            assert qam[0]
            F_start = False
            for TF in qam:
                if not TF:
                    F_start = True
                else:
                    assert not F_start
            num_concept = min(len(concepts), max_node_num - 1) + 1  # this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            # Prepare nodes
            concepts = concepts[: num_concept - 1]
            concept_ids[idx, 1:num_concept] = torch.tensor(concepts + 1)  # To accomodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = 0  # this is the "concept_id" for contextnode

            # Prepare node scores
            if cid2score is not None:
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            # Prepare node types
            node_type_ids[idx, 0] = 3  # contextnode
            node_type_ids[idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[: num_concept - 1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[: num_concept - 1]] = 1

            # Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            # Prepare edges
            i += 2
            j += 1
            k += 1  # **** increment coordinate by 1, rel_id by 2 ****
            extra_i, extra_j, extra_k = [], [], []
            for _coord, q_tf in enumerate(qm):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if q_tf:
                    extra_i.append(0)  # rel from contextnode to question concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # question concept coordinate
            for _coord, a_tf in enumerate(am):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if a_tf:
                    extra_i.append(1)  # rel from contextnode to answer concept
                    extra_j.append(0)  # contextnode coordinate
                    extra_k.append(_new_coord)  # answer concept coordinate

            half_n_rel += 2  # should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)
            ########################

            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
            edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
            edge_type.append(i)  # each entry is [E, ]

        with open(cache_path, "wb") as f:
            pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel], f)

    ori_adj_mean = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean) ** 2).mean().item())
    print(
        "| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |".format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item())
        + " prune_rate： {:.2f} |".format((adj_lengths_ori > adj_lengths).float().mean().item())
        + " qc_num: {:.2f} | ac_num: {:.2f} |".format((node_type_ids == 0).float().sum(1).mean().item(), (node_type_ids == 1).float().sum(1).mean().item())
    )

    edge_index = list(
        map(list, zip(*(iter(edge_index),) * num_choice))
    )  # list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice)))  # list of size (n_questions, n_choices), where each entry is tensor[E, ]

    concept_ids, node_type_ids, node_scores, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths)]
    # concept_ids: (n_questions, num_choice, max_node_num)
    # node_type_ids: (n_questions, num_choice, max_node_num)
    # node_scores: (n_questions, num_choice, max_node_num)
    # adj_lengths: (n_questions,　num_choice)
    return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type)  # , half_n_rel * 2 + 1


def load_clutrr_adj(dataRows):
    concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = [], [], [], [], [], []
    for datarow in dataRows:
        aconcept_ids, anode_type_ids, anode_scores, aadj_lengths = datarow.decoder_data
        aedge_index, aedge_type = datarow.adj_data
        concept_ids.append(aconcept_ids)
        node_type_ids.append(anode_type_ids)
        node_scores.append(anode_scores)
        adj_lengths.append(aadj_lengths)
        edge_index.append(aedge_index)
        edge_type.append(aedge_type)
    concept_ids, node_type_ids, node_scores, adj_lengths = [torch.stack(x, dim=0) for x in (concept_ids, node_type_ids, node_scores, adj_lengths)]
    return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type)


def load_gpt_input_tensors(statement_jsonl_path, max_seq_length):
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def load_qa_dataset(dataset_path):
        """Output a list of tuples(story, 1st continuation, 2nd continuation, label)"""
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                label = ord(input_json.get("answerKey", "A")) - ord("A")
                output.append((input_json["id"], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
        return output

    def pre_process_datasets(encoded_datasets, num_choices, max_seq_length, start_token, delimiter_token, clf_token):
        """Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        """
        tensor_datasets = []
        for dataset in encoded_datasets:
            n_batch = len(dataset)
            input_ids = np.zeros((n_batch, num_choices, max_seq_length), dtype=np.int64)
            mc_token_ids = np.zeros((n_batch, num_choices), dtype=np.int64)
            lm_labels = np.full((n_batch, num_choices, max_seq_length), fill_value=-1, dtype=np.int64)
            mc_labels = np.zeros((n_batch,), dtype=np.int64)
            for (
                i,
                data,
            ) in enumerate(dataset):
                q, mc_label = data[0], data[-1]
                choices = data[1:-1]
                for j in range(len(choices)):
                    _truncate_seq_pair(q, choices[j], max_seq_length - 3)
                    qa = [start_token] + q + [delimiter_token] + choices[j] + [clf_token]
                    input_ids[i, j, : len(qa)] = qa
                    mc_token_ids[i, j] = len(qa) - 1
                    lm_labels[i, j, : len(qa) - 1] = qa[1:]
                mc_labels[i] = mc_label
            all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
            tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
        return tensor_datasets

    def tokenize_and_encode(tokenizer, obj):
        """Tokenize and encode a nested object"""
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)

    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(GPT_SPECIAL_TOKENS)

    dataset = load_qa_dataset(statement_jsonl_path)
    examples_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2

    encoded_dataset = tokenize_and_encode(tokenizer, dataset)

    ((input_ids, mc_token_ids, lm_labels, mc_labels),) = pre_process_datasets([encoded_dataset], num_choices, max_seq_length, *special_tokens_ids)
    return examples_ids, mc_labels, input_ids, mc_token_ids, lm_labels


def get_gpt_token_num():
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    return len(tokenizer)


import re


def custom_tokenize_atmark(text, tokenizer):
    # 正規表現を使用して '@entN' を分割
    # segments = re.split(r'(@ent\d+)', text)
    segments = re.split(r"( ?@ent\d+)", text)
    # 各セグメントをトークン化
    tokens = []
    for i, segment in enumerate(segments):
        if segment.startswith(" @ent"):
            # スペースを 'Ġ' に変換する
            tokens.append("Ġ" + segment[1:])
        elif segment.startswith("@ent"):
            tokens.append(segment)
        else:
            tokens.extend(tokenizer.tokenize(segment))
    return tokens


def find_replace_atmark_indices(atmark_tokens, entity_ids):  # TODO:indicesではなくmaskで返すようにすると、長さを揃えてtensor化できるので、このあと全ての操作をバッチ計算できる
    indices = {entity_id: [] for entity_id in entity_ids}
    for idx, token in enumerate(atmark_tokens):
        for entity_id in entity_ids:
            if token == f"@ent{entity_id}":
                indices[entity_id].append(idx)
                atmark_tokens[idx] = f"{entity_id}"  # トークンを置き換える
            elif token == f"Ġ@ent{entity_id}":
                indices[entity_id].append(idx)
                atmark_tokens[idx] = f"Ġ{entity_id}"  # トークンを置き換える
    for entity_id in entity_ids:
        if not indices[entity_id]:
            raise ValueError(f"Entity ID '@ent{entity_id}' is not found in the given text.")
    return atmark_tokens, indices


def find_replace_relation_indices(atmark_tokens, entity_ids):
    # 正規表現パターンを定義
    pattern = re.compile(r"(Ġ?)@ent(\d+)")
    indices = {entity_id: [] for entity_id in entity_ids}
    idx_offset = 0  # '@' トークンが追加されるたびにオフセットを更新
    new_tokens = []
    for idx, token in enumerate(atmark_tokens):
        # 正規表現でマッチング
        match = pattern.match(token)
        if match:
            prefix, entity_id = match.groups()
            if entity_id in entity_ids:
                indices[entity_id].append(idx + idx_offset + 1)
                # '@' トークンを挿入
                new_tokens.extend(["@", f"{prefix}{entity_id}", "#"])
                idx_offset += 2  # '@', '#'トークンが追加された
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)

    for entity_id in entity_ids:
        if not indices[entity_id]:
            raise ValueError(f"Entity ID '@ent{entity_id}' is not found in the given text.")
    return new_tokens, indices


def segmenta_atmark_count(atmark_tokens):
    pattern = re.compile(r"(Ġ?)@ent(\d+)")
    count = 0
    for idx, token in enumerate(atmark_tokens):
        # 正規表現でマッチング
        match = pattern.match(token)
        if match:
            count += 1
    return count


def map_entid2cptid_indices(entityid_str):  # TODO
    return int(entityid_str) + 2


def convert_indices_to_mask(indices_list, max_seq_len, concept_num):
    # 1. 最大のマッピングインデックスを得る
    # //max_mapped_index = max(map_entid2cptid_indices(key) for key in indices.keys())
    entity_mask = []
    for indices in indices_list:
        # 2. すべて0の2Dテンソルを作成
        aentity_mask = torch.zeros((concept_num + 1, max_seq_len), dtype=torch.long)
        # 3. indicesの各キーに対応する位置を1に設定
        for key, value in indices.items():
            aentity_mask[map_entid2cptid_indices(key), value] = 1
        entity_mask.append(aentity_mask)
    entity_mask = torch.stack(entity_mask)
    return entity_mask


def load_bert_xlnet_roberta_input_tensors(statement_jsonl_path, model_type, model_name, max_seq_length, return_entity_mask, ent_format, concept_num):
    if return_entity_mask:
        assert (ent_format == "atmark") or (ent_format == "relation")

    class InputExample(object):
        def __init__(self, example_id, question, contexts, endings, label=None, entid_list=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label
            self.entid_list = entid_list

    class InputFeatures(object):
        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                    "output_mask": output_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        if isinstance(input_file, list):  # ?list of dataRow
            examples = []
            for datarow in input_file:
                json_dic = datarow.statement_dic
                #!label = ord(json_dic["answerKey"]) - ord("21") if 'answerKey' in json_dic else 0 #Todo: 21 should be variable
                label = int(json_dic["answerKey"]) - int("21") if "answerKey" in json_dic else 0  # Todo: 21 should be variable
                contexts = json_dic["question"]["stem"]
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label,
                        entid_list=datarow.entidlist,
                    )
                )
            return examples
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if "answerKey" in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label,
                    )
                )
        return examples

    def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        sep_token_extra=False,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ):
        """Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        ent_indices = []
        for ex_index, example in enumerate(tqdm(examples)):
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                #!if return_entity_mask:
                if ent_format == "atmark" or ent_format == "relation":
                    tokens_a = custom_tokenize_atmark(context, tokenizer)
                else:
                    tokens_a = tokenizer.tokenize(context)
                # custom tokenize
                tokens_b = tokenizer.tokenize(example.question + " " + ending)  #!not custom tokenize

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)
                if ent_format == "relation":
                    offseta = 2 * segmenta_atmark_count(tokens)
                    segment_ids += [sequence_a_segment_id] * offseta

                if tokens_b:  # TODO ent format
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
                # TODO 1 segment for one_choice classify data
                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                # if return_entity_mask:
                if ent_format == "atmark":
                    tokens, indices = find_replace_atmark_indices(tokens, example.entid_list)  # @entN indices and convert @entN -> 'N' for id
                elif ent_format == "relation":
                    tokens, indices = find_replace_relation_indices(tokens, example.entid_list)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.

                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
            if len(example.endings) == 1:
                label = example.label
            else:
                label = label_map[example.label]
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))
            if return_entity_mask:
                ent_indices.append(indices)

        if return_entity_mask:
            return features, ent_indices
        else:
            return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, "output_mask"), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    # try:
    #     tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}.get(model_type)
    # except:
    #     tokenizer_class = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer, 'roberta': RobertaTokenizer}.get(model_type)
    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name)
    examples = read_examples(statement_jsonl_path)
    entity_masks = None
    if not return_entity_mask:
        features = convert_examples_to_features(
            examples,
            list(range(len(examples[0].endings))),
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),  # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta", "albert"]),
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
            sequence_b_segment_id=0 if model_type in ["roberta", "albert"] else 1,
        )
        example_ids = [f.example_id for f in features]
        *data_tensors, all_label = convert_features_to_tensors(features)
        #!return (example_ids, all_label, *data_tensors)
    else:
        features, ent_indices = convert_examples_to_features(
            examples,
            list(range(len(examples[0].endings))),
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),  # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta", "albert"]),
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
            sequence_b_segment_id=0 if model_type in ["roberta", "albert"] else 1,
        )
        example_ids = [f.example_id for f in features]
        *data_tensors, all_label = convert_features_to_tensors(features)
        entity_masks = convert_indices_to_mask(ent_indices, max_seq_length, concept_num)
        #!まだバッチ化される前で、各設問のentity_maskテンソルのリスト
    return (entity_masks, example_ids, all_label, *data_tensors)


def load_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length, return_entity_mask=False, ent_format="figure", concept_num=40):
    if model_type in ("lstm",):
        raise NotImplementedError
    elif model_type in ("gpt",):
        return load_gpt_input_tensors(input_jsonl_path, max_seq_length)
    elif model_type in ("bert", "xlnet", "roberta", "albert"):
        return load_bert_xlnet_roberta_input_tensors(
            input_jsonl_path, model_type, model_name, max_seq_length, return_entity_mask=return_entity_mask, ent_format=ent_format, concept_num=concept_num
        )


def load_info(statement_path: str):
    n = sum(1 for _ in open(statement_path, "r"))
    num_choice = None
    with open(statement_path, "r", encoding="utf-8") as fin:
        ids = []
        labels = []
        for line in fin:
            input_json = json.loads(line)
            labels.append(ord(input_json.get("answerKey", "A")) - ord("A"))
            ids.append(input_json["id"])
            if num_choice is None:
                num_choice = len(input_json["question"]["choices"])
        labels = torch.tensor(labels, dtype=torch.long)

    return ids, labels, num_choice


def load_statement_dict(statement_path):
    all_dict = {}
    with open(statement_path, "r", encoding="utf-8") as fin:
        for line in fin:
            instance_dict = json.loads(line)
            qid = instance_dict["id"]
            all_dict[qid] = {"question": instance_dict["question"]["stem"], "answers": [dic["text"] for dic in instance_dict["question"]["choices"]]}
    return all_dict
