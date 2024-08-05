from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
from utils.utils import convert_qagnn2compile_input, extract_query_rep, make_sparse_tensor
import torch
import torch.nn.functional as F
import pickle as pkl
from clutrr_dataUtil import DataRow, DataUtility
import math


def get_activation_function(activation):
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "LeakyReLU":
        return nn.LeakyReLU(0.1)
    elif activation == "PReLU":
        return nn.PReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "SELU":
        return nn.SELU()
    elif activation == "ELU":
        return nn.ELU()
    else:
        raise ValueError('Activation "{}" not supported.'.format(activation))


from torch.cuda.amp import autocast


class MySpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)
        # dd = dense_mat.dtype
        # まずはfloat32で計算を行う
        # result = torch.mm(sp_mat.to(dtype=torch.float32), dense_mat.to(dtype=torch.float32))
        # 計算が完了したら、出力をHalf（またはdd）に変換する
        # result = result.to(dtype=torch.float16)
        #!sp_mat = sp_mat.to_sparse_csr() #after pytorch2.0
        # AMPの影響を受けないようにする
        with autocast(enabled=False):
            dense_mat = dense_mat.to(dtype=torch.float32)
            result = torch.mm(sp_mat, dense_mat)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))

        return grad_matrix1, grad_matrix2


def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        #  print(hidden.shape)
        message = F.relu(node + self.bias)
        MAX_atom_len = max(a_scope)
        # padding
        message_lst = []
        hidden_lst = []
        a_start = 0
        for i in a_scope:
            if i == 0:
                assert 0
            cur_message = message.narrow(0, a_start, i)
            cur_hidden = hidden.narrow(0, a_start, i)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            a_start += i
            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        kk = 0
        for a_size in a_scope:
            cur_message_unpadding.append(cur_message[kk, :a_size].view(-1, 2 * self.hidden_size))
            kk += 1
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        #   message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
        #                        cur_message_unpadding], 0)
        #   print(cur_message_unpadding.shape)
        return cur_message_unpadding


def create_subj_obj_mask(entity_mask, edgeent_position):
    # entity_maskのコピーを作成
    emdevice = entity_mask.device
    entity_mask_copy = entity_mask.clone()  # .cpu()
    zeros_tensor = torch.zeros(entity_mask_copy.size(0), entity_mask_copy.size(1), 1, device=emdevice)

    if edgeent_position == "all":
        # entity_mask中で1の箇所の前後も1にする
        mask_right = torch.cat([zeros_tensor, entity_mask_copy[:, :, :-1]], dim=2).to(torch.long)
        mask_left = torch.cat([entity_mask_copy[:, :, 1:], zeros_tensor], dim=2).to(torch.long)
        subj_mask = entity_mask_copy | mask_left | mask_right
        obj_mask = subj_mask.clone()  # コピーを作成
    elif edgeent_position == "fnp":
        # entity_mask中で1の箇所の前後を1にして元々1だった箇所を0にする
        mask_right = torch.cat([zeros_tensor, entity_mask_copy[:, :, :-1]], dim=2).to(torch.long)
        mask_left = torch.cat([entity_mask_copy[:, :, 1:], zeros_tensor], dim=2).to(torch.long)

        subj_mask = mask_left | mask_right
        obj_mask = subj_mask.clone()  # コピーを作成
    elif edgeent_position == "forp":
        # entity_mask中で1の箇所の前を1にしたものをsubj_mask,後を1にしたものをobj_maskとして返す
        mask_right = torch.cat([zeros_tensor, entity_mask_copy[:, :, :-1]], dim=2).to(torch.long)
        mask_left = torch.cat([entity_mask_copy[:, :, 1:], zeros_tensor], dim=2).to(torch.long)

        subj_mask = mask_left & (~entity_mask_copy)
        obj_mask = mask_right & (~entity_mask_copy)
    elif edgeent_position == "entity":
        subj_mask = entity_mask_copy.clone()
        obj_mask = entity_mask_copy.clone()  # コピーを作成
    else:
        raise ValueError("Invalid edgeent_position")
    return subj_mask, obj_mask  # .to(device=emdevice)


def convert_somask2edgemask(edge_index, concept_ids, subj_mask, obj_mask):
    edge_nodeposition_mask_list = []
    # Iterate through each graph's edge_index in the batch
    for idx, edges in enumerate(edge_index):
        # Extract subject and object nodes directly
        subj_nodes, obj_nodes = edges
        # Fetch unique node ids from concept_ids for all edges simultaneously
        subj_ids = concept_ids[idx, subj_nodes]
        obj_ids = concept_ids[idx, obj_nodes]
        # Fetch corresponding masks from subj_mask and obj_mask for all edges simultaneously
        subj_masks = subj_mask[idx, subj_ids]
        obj_masks = obj_mask[idx, obj_ids]
        # Stack masks to match the required shape #![2, edge_num, max_seq_len]
        mask_tensor = torch.stack([subj_masks, obj_masks], dim=0)
        edge_nodeposition_mask_list.append(mask_tensor)
    return edge_nodeposition_mask_list


def average_entity_embeddings(batch_embedding, entity_mask):
    # batch_embeddingから該当する埋め込みを取得
    masked_embeddings = batch_embedding.unsqueeze(1) * entity_mask.unsqueeze(-1)

    # 各エンティティの平均埋め込みを計算
    sum_embeddings = torch.sum(masked_embeddings, dim=2)
    mask_counts = entity_mask.sum(dim=2, keepdim=True).clamp(min=1)  # マスクが全て0の場合のゼロ除算を避けるために最小値を1とする
    avg_embeddings = sum_embeddings / mask_counts

    return avg_embeddings


def average_edge_embeddings_v2(batch_embedding, edge2posmask):
    avg_embedding_list = []

    for sequence, edge in zip(batch_embedding, edge2posmask):
        # Calculate masked embeddings for both subjects and objects of edges
        masked_embeddings_subj = sequence * edge[0].unsqueeze(-1)
        masked_embeddings_obj = sequence * edge[1].unsqueeze(-1)

        # Sum embeddings across the sequence length dimension (316)
        sum_embeddings_subj = torch.sum(masked_embeddings_subj, dim=1)
        sum_embeddings_obj = torch.sum(masked_embeddings_obj, dim=1)

        # Calculate mask counts and avoid division by zero
        mask_counts_subj = edge[0].sum(dim=1, keepdim=True).clamp(min=1)
        mask_counts_obj = edge[1].sum(dim=1, keepdim=True).clamp(min=1)

        # Calculate average embeddings
        avg_embeddings_subj = sum_embeddings_subj / mask_counts_subj
        avg_embeddings_obj = sum_embeddings_obj / mask_counts_obj

        # Stack the average embeddings for subjects and objects
        avg_embeddings = torch.stack([avg_embeddings_subj, avg_embeddings_obj], dim=0)
        avg_embedding_list.append(avg_embeddings)

    return avg_embedding_list


class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size, LMrelemb, sent_dim, dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype  #! maybe int
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size // 2)

        self.basis_f = "sin"  # ['id', 'linact', 'sin', 'none']
        if self.basis_f in ["id"]:
            self.emb_score = nn.Linear(1, hidden_size // 2)
        elif self.basis_f in ["linact"]:
            self.B_lin = nn.Linear(1, hidden_size // 2)
            self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)
        elif self.basis_f in ["sin"]:
            self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)

        if LMrelemb:
            self.edge_encoder = torch.nn.Sequential(
                torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size // 2),
                torch.nn.BatchNorm1d(hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size // 2, hidden_size // 2),
            )
            self.lmvec2relemb = nn.Linear(sent_dim * 2, hidden_size // 2)
        else:
            self.edge_encoder = torch.nn.Sequential(
                torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size)
            )
            self.lmvec2relemb = None

        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder, lmvec2relemb=self.lmvec2relemb) for _ in range(k)])

        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra, edge2lmemb):
        for _ in range(self.k):
            _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra, edge2lmemb)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training=self.training)
        return _X

    def forward(self, H, A, node_type, node_score, edge2lmemb=None, cache_output=False):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        # Embed type
        # print("node_type device:",node_type.device,"node_type shapr:",node_type.size())

        assert node_type.min() >= 0
        assert node_type.max() < self.n_ntype
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T))  # [batch_size, n_node, dim/2]

        # Embed score
        if self.basis_f == "sin":
            js = torch.arange(self.hidden_size // 2).unsqueeze(0).unsqueeze(0).float().to(node_type.device)  # [1,1,dim/2]
            js = torch.pow(1.1, js)  # [1,1,dim/2]
            B = torch.sin(js * node_score)  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == "id":
            B = node_score
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == "linact":
            B = self.activation(self.B_lin(node_score))  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]

        X = H
        edge_index, edge_type = A  # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous()  # [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous()  # [`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous()  # [`total_n_nodes`, dim]

        _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra, edge2lmemb)

        X = _X.view(node_type.size(0), node_type.size(1), -1)  # [batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return output


class QAGNN(nn.Module):
    def __init__(
        self,
        args,
        k,
        n_ntype,
        n_etype,
        sent_dim,
        n_concept,
        concept_dim,
        concept_in_dim,
        n_attention_head,
        fc_dim,
        n_fc_layer,
        p_emb,
        p_gnn,
        p_fc,
        fc_out_dim,
        pretrained_concept_emb=None,
        freeze_ent_emb=True,
        init_range=0.02,
    ):
        super().__init__()
        self.init_range = init_range

        self.concept_emb = CustomizedEmbedding(
            concept_num=n_concept,
            concept_out_dim=concept_dim,
            use_contextualized=False,
            concept_in_dim=concept_in_dim,
            pretrained_concept_emb=pretrained_concept_emb,
            freeze_ent_emb=freeze_ent_emb,
            LMentemb=args.LMentemb,
            sent_dim=sent_dim,
        )
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)

        self.concept_dim = concept_dim

        self.activation = GELU()

        self.gnn = QAGNN_Message_Passing(
            args,
            k=k,
            n_ntype=n_ntype,
            n_etype=n_etype,
            input_size=concept_dim,
            hidden_size=concept_dim,
            output_size=concept_dim,
            LMrelemb=args.LMrelemb,
            sent_dim=sent_dim,
            dropout=p_gnn,
        )

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, fc_out_dim, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, entid2lmemb, edge2lmemb, cache_output=False
    ):  # hidden_states,entity_mask=None,edge2posmask=None,
        """
        sent_vecs: (batch_size, sent_dim)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """

        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1)  # (batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:] - 1, entid2lmemb)  # (batch_size, n_node-1, dim_node)
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1))  # (batch_size, n_node, dim_node)

        # Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float()  # 0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :]  # [batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2)  # [batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  # [batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05)  # [batch_size, n_node]
        node_scores = node_scores.unsqueeze(2)  # [batch_size, n_node, 1]

        gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores, edge2lmemb=edge2lmemb)

        Z_vecs = gnn_output[:, 0]  # (batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1)  # 1 means masked out

        mask = mask | (node_type_ids == 3)  # pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))
        logits = self.fc(concat)
        return logits, pool_attn


class CoMPILE(nn.Module):
    def __init__(
        self,
        n_ntype,  # =node_emb,
        n_etype,
        sent_dim,
        LMentemb_flag,
        LMrelemb_flag,
        gnn_dim,
        fc_out_dim,
        fc_dim,
        n_fc_layer,
        adepth,
        aactivation,
        adropout,
        lineared_sent_vec,
        initemb_method,
        init_range,
        compile_mlp_queryrep,
        edge_scoring,
        edge_pruning_ratio,
        args,
    ):
        super().__init__()
        self.LMentemb_flag = LMentemb_flag
        self.LMrelemb_flag = LMrelemb_flag
        # self.params = params
        # self.args = args
        # self.depth = args.depth
        # self.dropout = args.dropout
        # self.act_func = get_activation_function(args.activation)
        self.compile_mlp_queryrep = compile_mlp_queryrep
        self.edge_scoring = edge_scoring
        self.edge_pruning_order = args.edge_pruning_order
        self.edge_pruning_ratio = edge_pruning_ratio
        self.start_pruning_epoch = args.start_pruning_epoch
        self.depth = adepth
        self.dropout = adropout
        self.act_func = get_activation_function(aactivation)

        self.output_dim = fc_out_dim
        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.sent_dim = sent_dim
        self.hidden_size = gnn_dim

        self.scored_edge_norm = args.scored_edge_norm

        #  self.relation_to_edge = nn.Linear(relation_emb, self.hidden_size)
        self.lineared_sent_vec = lineared_sent_vec
        if lineared_sent_vec:
            self.fc = MLP(self.hidden_size * 4, fc_dim, fc_out_dim, n_fc_layer, dropout=self.dropout, layer_norm=True)
        else:
            if not self.compile_mlp_queryrep:
                #!self.fc = MLP(self.hidden_size * 3 + sent_dim, fc_dim, fc_out_dim, n_fc_layer, dropout=self.dropout, layer_norm=True)
                # sent_vecs, mol_vecs, final_queary_edges, final_query_source, final_query_target, relation_embed[query_idx]
                self.fc = MLP(self.hidden_size * 5 + sent_dim, fc_dim, fc_out_dim, n_fc_layer, dropout=self.dropout, layer_norm=True)
            else:
                self.fc = MLP(self.hidden_size * 3 + sent_dim * 3, fc_dim, fc_out_dim, n_fc_layer, dropout=self.dropout, layer_norm=True)

        # ?self.linear1 = nn.Linear(self.hidden_size, self.output_dim)

        self.bias = False

        self.layers_per_message = 1
        self.undirected = False
        self.node_messages = False  # TODO

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        # ########### ELU relu tanh

        # Cached zeros
        #   self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size),  requires_grad=False)
        # Input

        #! flags and context divide linears
        # self.final_relation_embeddings = nn.Parameter(torch.randn(n_etype, self.hidden_size))
        # self.W_i_node = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        self.initemb_method = initemb_method
        if self.initemb_method == "onehot-LM":
            self.Wi_node_context = nn.Linear(n_ntype + sent_dim, self.hidden_size, bias=self.bias)
            if self.LMentemb_flag:
                self.Wi_node_non_context = nn.Linear(n_ntype + sent_dim, self.hidden_size, bias=self.bias)
        elif self.initemb_method == "concat-linear":
            self.Wi_nodetype_embedding = nn.Embedding(n_ntype, self.hidden_size // 2)
            self.Wi_nodetype_embedding.weight.data.normal_(mean=0.0, std=init_range)
            self.Wi_node_context = nn.Linear(sent_dim, self.hidden_size // 2, bias=self.bias)
            if self.LMentemb_flag:
                self.Wi_node_non_context = nn.Linear(sent_dim, self.hidden_size // 2, bias=self.bias)
        if not self.LMentemb_flag:
            self.Wi_node_non_context_embedding = nn.Embedding(n_ntype, self.hidden_size)
            self.Wi_node_non_context_embedding.weight.data.normal_(mean=0.0, std=init_range)

        # w_h_input_size_atom = self.hidden_size + (self.n_ntyoe * 2 + self.n_etype )
        #  self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        # self.W_i_edge = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.LMrelemb_flag:
            if self.initemb_method == "onehot-LM":
                self.Wi_edge_context = nn.Linear(n_etype + 2 * sent_dim, self.hidden_size, bias=self.bias)
                self.Wi_edge_non_context = nn.Linear(n_etype + 2 * sent_dim, self.hidden_size, bias=self.bias)
            elif self.initemb_method == "concat-linear":
                self.Wi_edgetype_embedding = nn.Embedding(n_etype, self.hidden_size // 2)
                self.Wi_edgetype_embedding.weight.data.normal_(mean=0.0, std=init_range)
                self.Wi_edge_context = nn.Linear(2 * sent_dim, self.hidden_size // 2, bias=self.bias)
                self.Wi_edge_non_context = nn.Linear(2 * sent_dim, self.hidden_size // 2, bias=self.bias)
        else:
            # ?self.Wi_edge_context = nn.Linear(n_etype + sent_dim, self.hidden_size)
            # ?self.Wi_edge_non_context = nn.Linear(n_etype, self.hidden_size)
            self.Wi_edge_embedding = nn.Embedding(n_etype, self.hidden_size)
            self.Wi_edge_embedding.weight.data.normal_(mean=0.0, std=init_range)

        if edge_scoring:
            self.edge_scoring_linear = nn.Linear(2 * sent_dim, 1, bias=True)
            if self.scored_edge_norm == "batch":
                self.edge_norm = torch.nn.BatchNorm1d(num_features=self.hidden_size)
            elif self.scored_edge_norm == "layer":
                self.edge_norm = torch.nn.LayerNorm(normalized_shape=self.hidden_size)
            elif self.scored_edge_norm == "disabled":
                pass
            else:
                raise ValueError

        # Use another linear layer, covert edge feat to edge embedding
        self.edge_feat2emb = nn.Linear(n_ntype * 2 + self.hidden_size, self.hidden_size, bias=self.bias)

        self.input_attention1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=self.bias)
        self.input_attention2 = nn.Linear(self.hidden_size, 1, bias=self.bias)  # TODO analyze about this

        for depth in range(self.depth - 1):
            self._modules["W_h_bond_{}".format(depth)] = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            #  self._modules['W_h_bond_{}'.format(depth)] = nn.Linear(w_h_input_size_bond * 3 + self.params.rel_emb_dim, self.hidden_size, bias=self.bias)
            self._modules["Attention1_{}".format(depth)] = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size, bias=self.bias)
            self._modules["Attention2_{}".format(depth)] = nn.Linear(self.hidden_size, 1, bias=self.bias)

        self.W_o = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = BatchGRU(self.hidden_size)

        self.communicate_mlp = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias)

        for depth in range(self.depth - 1):
            self._modules["W_h_atom_{}".format(depth)] = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

        self.decoder_model = args.decoder_model
        if args.decoder_model == "RGCN":
            self.layers = nn.ModuleList()
            for _ in range(n_fc_layer):
                self.layers.append(CustomRGCNConv(gnn_dim, gnn_dim))
            self.fc = MLP(self.hidden_size * 3 + sent_dim, fc_dim, fc_out_dim, n_fc_layer, dropout=self.dropout, layer_norm=True)

    def forward(
        self,
        e2n_sp,
        e2n_sp2,
        graph_sizes,
        list_num_edges,
        source_node,
        target_node,
        query_edge_indices,
        node_type_ids,
        source_map,
        target_map,
        edge_types,
        sent_vecs,
        LMnodesemb,
        edge2lmemb,
        query_rep,
        cur_epoch=None,
    ):
        # batch_target_relation = batch_inputs[:, 1].view(-1,)
        # target_relation = self.final_relation_embeddings[batch_target_relation, :]
        mlp_out = self.gnn(
            node_type_ids,
            source_map,
            target_map,
            edge_types,
            sent_vecs,
            LMnodesemb,
            edge2lmemb,  # node_feat, edge_feat,
            e2n_sp,
            e2n_sp2,
            graph_sizes,  # total_target_relation, total_source, total_target,
            source_node,
            target_node,
            list_num_edges,
            query_edge_indices,
            query_rep,
            cur_epoch,
        )

        return mlp_out

    def gnn(
        self,
        node_type_ids,
        source_map,
        target_map,
        edge_types,
        sent_vecs,
        LMnodesemb,
        edge2lmemb,  # node_feat, edge_feat,
        e2n_sp,
        e2n_sp2,
        graph_sizes,  # total_target_relation, total_source, total_target,
        source_node,
        target_node,
        list_num_edges,
        query_edge_indices,
        query_rep=None,
        cur_epoch=None,
    ):
        """if exists edge feature, concatenate to node feature vector"""
        # input_node_emb = self.W_i_node(node_feat)  # num_atoms x hidden_size
        input_node_emb, context_mask = self.init_node_embeddings(node_type_ids, LMnodesemb, sent_vecs, self.hidden_size, self.n_ntype, self.LMentemb_flag)
        input_node_emb = self.act_func(input_node_emb)
        message_node = input_node_emb.clone()

        if self.edge_scoring:
            edge_scores = torch.sigmoid(self.edge_scoring_linear(edge2lmemb.reshape(-1, 2 * self.sent_dim)))
            assert edge_scores.dim() == 2
            edge_scores = edge_scores.squeeze(1)
        if (self.edge_pruning_order != "disabled") and cur_epoch >= self.start_pruning_epoch:
            source_context_mask = node_type_ids[source_map] == 3
            target_context_mask = node_type_ids[target_map] == 3
            context_query_edge_mask = source_context_mask | target_context_mask
            for query_idx in query_edge_indices:
                assert context_query_edge_mask[query_idx] == 0
                context_query_edge_mask[query_idx] = 1

            if self.edge_pruning_order == "const":
                pruning_ratio_list = [self.edge_pruning_ratio for i in graph_sizes]
            elif self.edge_pruning_order == "linear":
                pruning_ratio_list = []
                for n in graph_sizes:
                    node_num = n - 1
                    assert (node_num >= 3) and (node_num <= 11)
                    pruned_edgen = int(self.edge_pruning_ratio * node_num)
                    aratio = pruned_edgen / (node_num * (node_num - 1))
                    assert aratio <= 1  # pruning_ratio_list.append(min(aratio, 1))
                    pruning_ratio_list.append(aratio)
            elif self.edge_pruning_order == "klogk":
                pruning_ratio_list = []
                for n in graph_sizes:
                    node_num = n - 1
                    assert (node_num >= 3) and (node_num <= 11)
                    pruned_edgen = int(self.edge_pruning_ratio * node_num * math.log10(node_num))  # + 1
                    aratio = pruned_edgen / (node_num * (node_num - 1))
                    assert aratio <= 1
                    pruning_ratio_list.append(aratio)
            else:
                raise ValueError

            edge_pooling_mask, list_num_edges = self.get_pool_mask(edge_scores, list_num_edges, context_query_edge_mask, pruning_ratio_list)

            edge_scores, source_map, target_map, edge_types, edge2lmemb, e2n_sp, e2n_sp2, query_edge_indices = self.init_with_pool_mask(
                edge_pooling_mask, edge_scores, source_map, target_map, edge_types, edge2lmemb, e2n_sp, e2n_sp2, query_edge_indices, total_num_nodes=len(node_type_ids)
            )
        # relation_embed = (edge_feat[:, self.node_emb: self.node_emb + self.relation_emb])
        # input_edge_emb = self.W_i_edge(edge_feat)  # num_bonds x hidden_size
        input_edge_emb, relation_embed = self.init_edge_embeddings(
            edge_types, source_map, target_map, node_type_ids, edge2lmemb, sent_vecs, list_num_edges, self.hidden_size, self.n_ntype, self.n_etype, self.LMrelemb_flag
        )
        message_edge = self.act_func(input_edge_emb)
        input_edge_emb = self.act_func(input_edge_emb)

        if self.decoder_model == "RGCN":
            """
            x: node_emb : (node_num, gnn_dim)
            edge_index: (2, edge_num)
            edge_emb: (edge_num,gnn_dim)
            """
            x = input_node_emb.clone()
            relation_embed = self.act_func(relation_embed)
            relation_embed = relation_embed * edge_scores.unsqueeze(1)
            edge_index = torch.stack((source_map, target_map))
        # graph_source_embed = message_node[total_source, :]
        # graph_target_embed = message_node[total_target, :]
        if self.decoder_model == "RGCN":
            for layer in self.layers:
                x = F.relu(layer(x, edge_index, relation_embed))

            final_query_source = message_node[source_node, :]  # (batch_size, gnn_dim)
            final_query_target = message_node[target_node, :]
            """a_message = torch.relu(self.gru(x, graph_sizes))
            atom_hiddens = self.act_func(self.W_o(a_message))  # num_atoms x hidden
            atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
            # Readout
            mol_vecs = []
            a_start = 0
            for a_size in graph_sizes:
                if a_size == 0:
                    assert 0
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vecs.append(cur_hiddens.mean(0))
                a_start += a_size
            mol_vecs = torch.stack(mol_vecs, dim=0)          """
            concat = self.dropout_layer(torch.cat((sent_vecs, final_query_source, final_query_target, relation_embed[query_edge_indices]), 1))
            logits = self.fc(concat)
            return logits
        # Create a tensor to hold the repeated query edge embeddings
        target_relation = torch.zeros((sum(list_num_edges), self.hidden_size)).to(relation_embed.device)
        # Fill in the repeated_query_edge_embs tensor
        start_idx = 0
        for i, query_idx in enumerate(query_edge_indices):
            end_idx = start_idx + list_num_edges[i]
            target_relation[start_idx:end_idx] = relation_embed[query_idx].repeat(list_num_edges[i], 1)
            start_idx = end_idx

        total_source = torch.zeros(sum(list_num_edges)).to(relation_embed.device, dtype=torch.long)
        total_target = torch.zeros(sum(list_num_edges)).to(relation_embed.device, dtype=torch.long)
        graph_source_embed = torch.zeros((sum(list_num_edges), self.hidden_size)).to(relation_embed.device)
        graph_target_embed = torch.zeros((sum(list_num_edges), self.hidden_size)).to(relation_embed.device)
        start_idx = 0
        for i, (source_idx, target_idx) in enumerate(zip(source_node, target_node)):
            end_idx = start_idx + list_num_edges[i]
            total_source[start_idx:end_idx] = source_idx
            total_target[start_idx:end_idx] = target_idx
            graph_source_embed[start_idx:end_idx] = message_node[source_idx].repeat(list_num_edges[i], 1)
            graph_target_embed[start_idx:end_idx] = message_node[target_idx].repeat(list_num_edges[i], 1)
            start_idx = end_idx
        graph_edge_embed = graph_source_embed + target_relation - graph_target_embed

        edge_target_message = gnn_spmm(e2n_sp.t(), message_node)
        edge_source_message = gnn_spmm(e2n_sp2.t(), message_node)
        edge_message = edge_source_message + relation_embed - edge_target_message
        #  print(total_source.shape, total_target.shape, graph_source_embed.shape)
        attention = torch.cat([graph_edge_embed, edge_message], dim=1)
        attention = torch.relu(self.input_attention1(attention))
        attention = torch.sigmoid(self.input_attention2(attention))

        # Message passing
        for depth in range(self.depth - 1):
            message_edge = message_edge * attention
            if self.edge_scoring:
                message_edge = message_edge * edge_scores.unsqueeze(1)
                if self.scored_edge_norm == "batch" or self.scored_edge_norm == "layer":
                    message_edge = self.edge_norm(message_edge)
            agg_message = gnn_spmm(e2n_sp, message_edge)
            message_node = message_node + agg_message
            message_node = self.act_func(self._modules["W_h_atom_{}".format(depth)](message_node))

            edge_target_message = gnn_spmm(e2n_sp.t(), message_node)
            edge_source_message = gnn_spmm(e2n_sp2.t(), message_node)
            # message_edge = torch.cat([message_edge, edge_source_message, relation_embed, edge_target_message], dim=-1)
            message_edge = torch.relu(message_edge + torch.tanh(edge_source_message + relation_embed - edge_target_message))
            message_edge = self._modules["W_h_bond_{}".format(depth)](message_edge)
            message_edge = self.act_func(input_edge_emb + message_edge)
            message_edge = self.dropout_layer(message_edge)  # num_bonds x hidden

            graph_source_embed = message_node[total_source, :]
            graph_target_embed = message_node[total_target, :]
            graph_edge_embed = graph_source_embed + target_relation - graph_target_embed
            edge_message = edge_source_message + relation_embed - edge_target_message
            attention = torch.cat([graph_edge_embed, edge_message], dim=1)
            attention = torch.relu(self._modules["Attention1_{}".format(depth)](attention))
            attention = torch.sigmoid(self._modules["Attention2_{}".format(depth)](attention))

        final_queary_edges = message_edge.clone()[query_edge_indices]
        final_query_source = message_node[source_node, :]  # (batch_size, gnn_dim)
        final_query_target = message_node[target_node, :]
        message_edge = message_edge * attention
        agg_message = gnn_spmm(e2n_sp, message_edge)
        agg_message2 = self.communicate_mlp(torch.cat([agg_message, message_node, input_node_emb], 1))
        # =============================================================================
        # TODO activation before Z_vec?
        Z_vecs = agg_message2[context_mask]
        a_message = torch.relu(self.gru(agg_message2, graph_sizes))
        atom_hiddens = self.act_func(self.W_o(a_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        a_start = 0
        for a_size in graph_sizes:
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
            a_start += a_size
        mol_vecs = torch.stack(mol_vecs, dim=0)

        source_embed = atom_hiddens[source_node, :]
        target_embed = atom_hiddens[target_node, :]

        if self.lineared_sent_vec:
            if not self.compile_mlp_queryrep:
                concat = self.dropout_layer(torch.cat((final_queary_edges, mol_vecs, input_node_emb[context_mask], Z_vecs), 1))
            else:
                raise ValueError("not implemented")
        else:
            if not self.compile_mlp_queryrep:
                #!concat = self.dropout_layer(torch.cat((final_queary_edges, mol_vecs, sent_vecs, Z_vecs), 1))
                concat = self.dropout_layer(torch.cat((sent_vecs, mol_vecs, final_queary_edges, final_query_source, final_query_target, relation_embed[query_edge_indices]), 1))
            else:
                concat = self.dropout_layer(torch.cat((final_queary_edges, mol_vecs, query_rep, sent_vecs, Z_vecs), 1))
            # if first_relation_embed:
            # relation_embed[query_idx]
            # if finaL_qnodes or readout_qnodes or no-qnodes:
            # final_query_source or source_embed , final_query_target or target_embed
            # if final_cls_node: Z_vecs
            # 最終層clsの有無は入れたい
        logits = self.fc(concat)

        return logits
        # return mol_vecs, source_embed, target_embed

    def init_node_embeddings(self, node_types, LMnodeemb_unite, sent_vecs, gnn_dim, n_ntype, LMentemb_flag):
        """
        Initialize the node embeddings based on the given parameters.

        Args:
        - node_types (tensor of shape (total_num_nodes)): The type IDs of nodes.
        - LMnodeemb_unite (tensor of shape (total_num_nodes, sent_dim)): The LM embeddings for all nodes.
        - sent_vecs (tensor of shape (batch_size, sent_dim)): The LM embeddings for the context nodes.
        - gnn_dim (int): The dimensionality of the GNN embeddings.
        - n_ntype (int): The maximum node type ID.
        - LMentemb_flag (bool): Whether to use LM embeddings.

        Returns:
        - node_embedding (tensor of shape (total_num_nodes, gnn_dim)): The initialized node embeddings.
        """

        # Identify context nodes (node_type == 3)
        context_mask = node_types == 3
        non_context_mask = ~context_mask

        # Get the number of context and non-context nodes
        # num_context_nodes = context_mask.sum().item()
        # num_non_context_nodes = non_context_mask.sum().item()

        if self.initemb_method == "onehot-LM":
            # Get the embeddings for the context nodes
            context_node_types_onehot = F.one_hot(node_types[context_mask], num_classes=n_ntype).float()
            # print(context_node_types_onehot.dtype)
            # print(sent_vecs.dtype)
            context_embedding = torch.cat([context_node_types_onehot, sent_vecs], dim=1)
            # print(context_embedding.dtype)
            context_embedding = self.Wi_node_context(context_embedding)
            # print(context_embedding.dtype)
            if self.LMentemb_flag:
                # Get the embeddings for the non-context nodes using LM embeddings
                non_context_node_types_onehot = F.one_hot(node_types[non_context_mask], num_classes=n_ntype).float()
                non_context_embedding = torch.cat([non_context_node_types_onehot, LMnodeemb_unite[non_context_mask]], dim=1)
                non_context_embedding = self.Wi_node_non_context(non_context_embedding)
        elif self.initemb_method == "concat-linear":
            # context nodes
            # print(node_types.dtype) int64
            context_type_embedding = self.Wi_nodetype_embedding(node_types[context_mask]).to(dtype=sent_vecs.dtype)
            context_LM_embedding = self.Wi_node_context(sent_vecs)
            # print(context_type_embedding.dtype)
            # print(context_LM_embedding.dtype)
            context_embedding = torch.cat([context_type_embedding, context_LM_embedding], dim=1)
            if self.LMentemb_flag:
                # non-context nodes
                non_context_type_embedding = self.Wi_nodetype_embedding(node_types[non_context_mask])
                non_context_LM_embedding = self.Wi_node_non_context(LMnodeemb_unite[non_context_mask])
                non_context_embedding = torch.cat([non_context_type_embedding, non_context_LM_embedding], dim=1)

        if not self.LMentemb_flag:
            # Get the embeddings for the non-context nodes using only node types
            non_context_embedding = self.Wi_node_non_context_embedding(node_types[non_context_mask])
        # Combine the context and non-context embeddings
        node_embedding = torch.zeros((node_types.shape[0], gnn_dim)).to(sent_vecs.device, dtype=sent_vecs.dtype)  #
        node_embedding[context_mask] = context_embedding  # .to(dtype=torch.float32)
        node_embedding[non_context_mask] = non_context_embedding.to(dtype=node_embedding.dtype)

        return node_embedding, context_mask

    def init_edge_embeddings(self, edge_types, source_map, target_map, node_types, edge2lmemb_unite, sent_vecs, list_num_edges, gnn_dim, n_ntype, n_etype, LMrelemb_flag):
        rel_dim = gnn_dim
        """
        Initialize the edge embeddings.

        Args:
        - edge_types (1D tensor): The type of each edge.
        - source_map,target_map (1D tensor): The source and target nodes for each edge.
        - node_types (1D tensor): The type of each node.
        - edge2lmemb_unite (3D tensor): The LM embeddings for the source and target of each edge.
        - sent_vecs (2D tensor): The LM embeddings for the context nodes.
        - gnn_dim (int): The GNN embedding dimension.
        - rel_dim (int): The relation embedding dimension.
        - n_ntype (int): The number of node types.
        - n_etype (int): The number of edge types.
        - LMrelemb_flag (bool): Whether to use LM embeddings for relations.

        Returns:
        - edge_embs (2D tensor): The initialized edge embeddings.
        """

        # Extract the LM embedding dimension
        # total_num_edges = edge_types.shape[0]

        # Get the node types for the source and target nodes
        source_node_types = node_types[source_map]
        target_node_types = node_types[target_map]
        # Create masks for context nodes in source and target
        source_context_mask = source_node_types == 3
        target_context_mask = target_node_types == 3
        # Mask for edges where either source or target is a context node
        context_edge_mask = source_context_mask | target_context_mask

        # Mask for edges where neither source nor target is a context node
        # Split the edge types, source nodes, and target nodes based on whether they are context edges
        context_edge_types = edge_types[context_edge_mask]
        non_context_edge_types = edge_types[~context_edge_mask]
        context_edge_types_one_hot = nn.functional.one_hot(context_edge_types, num_classes=n_etype).float()
        non_context_edge_types_one_hot = nn.functional.one_hot(non_context_edge_types, num_classes=n_etype).float()

        batch_indices_for_edges = torch.cat([torch.full((num,), i) for i, num in enumerate(list_num_edges)], dim=0)

        if LMrelemb_flag:
            # Note: Assuming that the context nodes appear in the same order as in sent_vecs
            edge2lmemb_unite = edge2lmemb_unite.to(dtype=sent_vecs.dtype)  # TODO try dont use cls embedding to context-node-emb(and score)
            edge2lmemb_unite[source_context_mask, 0, :] = sent_vecs[batch_indices_for_edges[source_context_mask]]  #! which is each batch sent vec?
            edge2lmemb_unite[target_context_mask, 1, :] = sent_vecs[batch_indices_for_edges[target_context_mask]]
            if self.initemb_method == "onehot-LM":
                context_edge_input = torch.cat([context_edge_types_one_hot, edge2lmemb_unite[context_edge_mask].reshape(-1, 2 * self.sent_dim)], dim=1)
                non_context_edge_input = torch.cat([non_context_edge_types_one_hot, edge2lmemb_unite[~context_edge_mask].reshape(-1, 2 * self.sent_dim)], dim=1)
                context_rel_embs = self.Wi_edge_context(context_edge_input)
                non_context_rel_embs = self.Wi_edge_non_context(non_context_edge_input)
            elif self.initemb_method == "concat-linear":
                context_edgetype_embs = self.Wi_edgetype_embedding(context_edge_types).to(dtype=sent_vecs.dtype)
                non_context_edgetype_embs = self.Wi_edgetype_embedding(non_context_edge_types).to(dtype=sent_vecs.dtype)
                context_LM_embs = self.Wi_edge_context(edge2lmemb_unite[context_edge_mask].reshape(-1, 2 * self.sent_dim))
                non_context_LM_embs = self.Wi_edge_non_context(edge2lmemb_unite[~context_edge_mask].reshape(-1, 2 * self.sent_dim))
                context_rel_embs = torch.cat([context_edgetype_embs, context_LM_embs], dim=1)
                non_context_rel_embs = torch.cat([non_context_edgetype_embs, non_context_LM_embs], dim=1)
        else:
            context_rel_embs = self.Wi_edge_embedding(context_edge_types).to(dtype=sent_vecs.dtype)
            non_context_rel_embs = self.Wi_edge_embedding(non_context_edge_types).to(dtype=sent_vecs.dtype)
            """context_edge_input = torch.cat([context_edge_types_one_hot, sent_vecs[batch_indices_for_edges[context_edge_mask]]], dim=1)
            non_context_edge_input = non_context_edge_types_one_hot
            # Pass through the linear layers to get the relation embeddings
            context_rel_embs = self.Wi_edge_context(context_edge_input)
            non_context_rel_embs = self.Wi_edge_non_context(non_context_edge_input)"""

        # Prepare the source and target node embeddings (one-hot encoded)
        source_node_embs = nn.functional.one_hot(source_node_types, num_classes=n_ntype).float()
        target_node_embs = nn.functional.one_hot(target_node_types, num_classes=n_ntype).float()

        # Initialize the final relation embeddings tensor
        rel_embs = torch.zeros((edge_types.shape[0], rel_dim)).to(sent_vecs.device, dtype=sent_vecs.dtype)  # torch.float32)  #
        rel_embs[context_edge_mask] = context_rel_embs  # .to(dtype=torch.float32)
        rel_embs[~context_edge_mask] = non_context_rel_embs  # .to(dtype=torch.float32)

        # Concatenate the source node, relation, and target node embeddings
        edge_feats = torch.cat([source_node_embs, rel_embs, target_node_embs], dim=1)

        # Update the final edge embeddings tensor
        edge_embs = self.edge_feat2emb(edge_feats)

        return edge_embs, rel_embs

    def get_pool_mask(self, edge_scores, edge_nums, context_edge_mask, pruning_ratio_list):
        # Initialize the mask tensor to all zeros
        pooled_edge_mask = torch.zeros_like(edge_scores)
        pooled_edge_nums = []

        start_idx = 0
        for i, num in enumerate(edge_nums):
            end_idx = start_idx + num

            # Extract subgraph edge scores and context mask
            subgraph_scores = edge_scores[start_idx:end_idx]
            subgraph_context_mask = context_edge_mask[start_idx:end_idx]

            # Find the edges that are not context edges
            non_context_idxs = torch.where(subgraph_context_mask == 0)[0]
            non_context_scores = subgraph_scores[non_context_idxs]

            # Sort the non-context edges by their scores in descending order
            sorted_idxs = torch.argsort(non_context_scores, descending=True)

            # Calculate the number of edges to keep based on the pooling ratio
            num_to_keep = int(len(non_context_scores) * pruning_ratio_list[i])

            # Update the mask for the edges to keep
            pooled_edge_mask[start_idx:end_idx][non_context_idxs[sorted_idxs[:num_to_keep]]] = 1

            # Ensure context edges are always included in the mask
            pooled_edge_mask[start_idx:end_idx][subgraph_context_mask == 1] = 1

            # Assert that the number of 1s in the mask equals to num_to_keep + number of context edges
            assert pooled_edge_mask[start_idx:end_idx].sum().item() == num_to_keep + subgraph_context_mask.sum().item(), "Number of 1s in the mask doesn't match expected value"

            # Update the number of edges after pooling
            pooled_edge_nums.append(int(pooled_edge_mask[start_idx:end_idx].sum().item()))

            start_idx = end_idx

        return pooled_edge_mask, pooled_edge_nums

    def init_with_pool_mask(self, edge_pooling_mask, edge_scores, source_map, target_map, edge_types, edge2lmemb, e2n_sp, e2n_sp2, query_edge_indices, total_num_nodes):
        pool_mask = edge_pooling_mask.bool()

        # Create a tensor of indices of the original edges,Extract the indices of the pooled edges
        original_indices = torch.arange(len(pool_mask))
        pooled_indices = original_indices[pool_mask]
        # Convert the query_edge_indices to their position in the pooled_indices tensor
        new_query_edge_indices = (pooled_indices.unsqueeze(0) == query_edge_indices.unsqueeze(1)).nonzero()[:, 1]
        try:
            assert len(query_edge_indices) == len(new_query_edge_indices)
        except AssertionError:
            print(f"{len(query_edge_indices)} -> new_query_edge_indices len{len(new_query_edge_indices)}")
            print(query_edge_indices)
            print(new_query_edge_indices)
            raise AssertionError

        source_map, target_map = source_map[pool_mask], target_map[pool_mask]
        e2n_sp, e2n_sp2 = make_sparse_tensor(source_map, target_map, total_num_nodes, len(pooled_indices))
        return edge_scores[pool_mask], source_map, target_map, edge_types[pool_mask], edge2lmemb[pool_mask], e2n_sp, e2n_sp2, new_query_edge_indices


class LM_QAGNN(nn.Module):
    def __init__(
        self,
        args,
        model_name,
        k,
        n_ntype,
        n_etype,
        n_concept,
        concept_dim,
        concept_in_dim,
        n_attention_head,
        fc_dim,
        n_fc_layer,
        p_emb,
        p_gnn,
        p_fc,
        fc_out_dim,
        pretrained_concept_emb=None,
        freeze_ent_emb=True,
        init_range=0.0,
        encoder_config={},
    ):
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config, LMentemb=args.LMentemb, concept_num=n_concept, LMrelemb=args.LMrelemb)
        self.decoder_model = args.decoder_model
        self.LMentemb = args.LMentemb
        # ent_format
        self.LMrelemb = args.LMrelemb
        self.ent_format = args.ent_format
        self.cur_epoch = None

        if self.decoder_model == "qagnn":
            self.decoder = QAGNN(
                args,
                k,
                n_ntype,
                n_etype,
                self.encoder.sent_dim,
                n_concept,
                concept_dim,
                concept_in_dim,
                n_attention_head,
                fc_dim,
                n_fc_layer,
                p_emb,
                p_gnn,
                p_fc,
                fc_out_dim,
                pretrained_concept_emb=pretrained_concept_emb,
                freeze_ent_emb=freeze_ent_emb,
                init_range=init_range,
            )
        elif (self.decoder_model == "compile") or (self.decoder_model == "RGCN"):
            # compile_args=Namespace
            self.decoder = CoMPILE(
                n_ntype,  # =node_emb,
                n_etype,
                self.encoder.sent_dim,
                self.LMentemb,
                self.LMrelemb,
                gnn_dim=concept_dim,
                fc_out_dim=fc_out_dim,
                fc_dim=fc_dim,
                n_fc_layer=n_fc_layer,
                adepth=k,
                aactivation="ReLU",
                adropout=args.dropouti,  # =p_emb. # p_gnn, p_fc,
                lineared_sent_vec=args.fc_linear_sent,
                initemb_method=args.initemb_method,
                init_range=init_range,
                compile_mlp_queryrep=args.compile_mlp_queryrep,
                edge_scoring=args.edge_scoring,
                edge_pruning_ratio=args.edge_pruning_ratio,
                args=args,
            )
            #!, n_attention_head,
            # fc_dim, n_fc_layer #about MLP settings
            # pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,

        elif self.decoder_model == "MLP":
            assert (not self.LMentemb) and (not self.LMrelemb)
            self.dropout_layer = nn.Dropout(p=args.dropouti)
            self.decoder = MLP(self.encoder.sent_dim * 3, fc_dim, fc_out_dim, n_fc_layer, dropout=args.dropouti, layer_norm=True)
        elif self.decoder_model == "LSTM-MLP":
            assert (not self.LMentemb) and (not self.LMrelemb)
            self.dropout_layer = nn.Dropout(p=args.dropouti)
            self.decoder = LSTM_MLP(self.encoder.sent_dim, fc_dim, fc_out_dim, n_fc_layer, dropouti=args.dropouti, layer_norm=True)

        self.edge_scoring = args.edge_scoring
        self.edgeent_position = args.edgeent_position
        self.fc_out_dim = fc_out_dim
        self.first_batch = False

    def forward(self, *inputs, layer_id=-1, entity_mask=None, cache_output=False, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, fc_out_dim)
        """

        def _to_device(obj, device):
            if isinstance(obj, (tuple, list)):
                return [_to_device(item, device) for item in obj]
            else:
                return obj.to(device)

        bs, nc = inputs[0].size(0), inputs[0].size(1)
        if self.fc_out_dim == 1:
            assert nc > 1
        else:
            assert nc == 1

        # Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = (
            [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]]
            + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]]
            + [sum(x, []) for x in inputs[-2:]]
        )
        # 最後の項ではlist of (batch_size, num_choice) -> list of (batch_size * num_choice, )　つまり2dリストのflattenが起きる
        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs

        #! entity_maskはchoiceのぶん用意しているわけではないので、*18したのち36にflattenしてサイズを合わせる
        if self.LMentemb or self.LMrelemb or self.ent_format == "atmark":
            entity_mask = entity_mask.unsqueeze(1).repeat(1, nc, 1, 1)
            entity_mask = entity_mask.view(entity_mask.size(0) * entity_mask.size(1), *entity_mask.size()[2:])
            entity_mask = entity_mask.to(node_type_ids.device)
        #!batch_graphより前にentity_maskからedge2posを作る。
        edge2posmask = None
        if self.LMrelemb:
            subj_mask, obj_mask = create_subj_obj_mask(entity_mask, self.edgeent_position)
            edge2posmask = convert_somask2edgemask(edge_index, concept_ids, subj_mask, obj_mask)
            # ? list of  (batch_size * choice_num )  (=36)
            # ?each tensor is (2 , edge_num , max_seq_len)
        elif self.LMentemb and self.edge_scoring:
            subj_mask, obj_mask = create_subj_obj_mask(entity_mask, edgeent_position="entity")
            edge2posmask = convert_somask2edgemask(edge_index, concept_ids, subj_mask, obj_mask)
        edge_index_orig, edge_type_orig = edge_index, edge_type
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device))  # edge_index: [2, total_E]   edge_type: [total_E, ]

        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        hidden_states = all_hidden_states[layer_id].to(node_type_ids.device)

        entid2lmemb = None
        if self.LMentemb or self.ent_format == "atmark":
            entid2lmemb = average_entity_embeddings(hidden_states, entity_mask)
            # ?in:(batch_size*choice_num , max_seq_len, 768or1024) , (batch_size*choice_num, concept_num, max_seq_len)
            # ?out:(batch_size*cn, concept_num, 768or1024)

        edge2lmemb = None
        #!edge2posはQAGNN内でLM埋め込みを平均化してからbatch_graph
        if (self.LMrelemb) or ((not self.LMrelemb) and self.LMentemb and self.edge_scoring):
            edge2lmemb_orig = average_edge_embeddings_v2(hidden_states, edge2posmask)
            # ?in:(batch_size*choice_num , max_seq_len, 768or1024) , list(batch_size*choice_num) (2,edge_num, max_seq_len)
            # ?out:list(batch_size*cn) (2,edge_num, 768or1024)
            # flatten mask
            edge2lmemb = self.batch_graph_edge2lmemb(edge2lmemb_orig)  #!(totalE , 2, 768or1024)
            assert edge2lmemb.size()[0] == edge_type.size()[0]
        if self.decoder_model == "qagnn":
            logits, attn = self.decoder(
                sent_vecs.to(node_type_ids.device),
                concept_ids,
                node_type_ids,
                node_scores,
                adj_lengths,
                adj,
                # hidden_states, entity_mask =entity_mask, edge2posmask=edge2posmask,
                entid2lmemb=entid2lmemb,
                edge2lmemb=edge2lmemb,
                cache_output=cache_output,
            )
        elif (self.decoder_model == "compile") or (self.decoder_model == "RGCN"):
            ntd = node_type_ids.device
            query_rep = extract_query_rep(node_type_ids, concept_ids, entid2lmemb)
            [
                graph_sizes,
                list_num_edges,
                concept_ids,
                node_types,
                LMnodesemb,
                source_node,
                target_node,
                query_edge_indices,
                source_map,
                target_map,
                edge_types,
                e2n_sp,
                e2n_sp2,
            ] = convert_qagnn2compile_input(
                concept_ids, node_type_ids, adj_lengths, edge_index_orig, edge_type_orig, entid2lmemb
            )  # ? edge2lmemb, sent_vecs
            # ? inputs device ...  CPU:  edge_type_orig? / GPU : edge_index_orig ,  concept_ids, node_type_ids, entid2lmemb  adj_lengths(maybe) ,
            # edge2lmemb on GPU
            # ? outputs device ...  CPU: query_edge_indices, edge_types, /  not tensor(graph_sizes, list_num_edges,source/target_node) /
            # ? GPU :  source/target map,concept_ids,e2n , node_types, LMnodesemb
            logits = self.decoder(  # node_feat, edge_feat, ,total_target_relation, total_source, total_target,
                e2n_sp,  # .to(ntd),
                e2n_sp2,  # .to(ntd),
                graph_sizes,
                list_num_edges,
                source_node,
                target_node,
                query_edge_indices,
                node_types,
                source_map,
                target_map,
                edge_types.to(ntd),  # edge_index and edge_type
                sent_vecs.to(ntd),  # TODO
                LMnodesemb,
                edge2lmemb,
                query_rep,
                self.cur_epoch
                # ?not used :concept_ids,node_scores,
            )
            attn = None
        elif self.decoder_model == "MLP":
            query_rep = extract_query_rep(node_type_ids, concept_ids, entid2lmemb)
            if True:  # TODO
                input_mask = lm_inputs[1].float().unsqueeze(-1)  #! valit for bert or roberta?
                # Use the mask to zero out padded tokens in hidden_states
                masked_hidden_states = hidden_states * input_mask
                # Sum the masked hidden_states
                sum_hidden_states = masked_hidden_states.sum(dim=1)
                # Calculate the number of non-zero tokens for each batch
                num_valid_tokens = input_mask.sum(dim=1)
                # Compute the average representation
                mlp_lm_rep = sum_hidden_states / num_valid_tokens
            else:
                mlp_lm_rep = sent_vecs
            mlp_inp = torch.cat([query_rep, mlp_lm_rep], dim=1).to(node_type_ids.device)
            logits = self.decoder(self.dropout_layer(mlp_inp))
            attn = None
        elif self.decoder_model == "LSTM-MLP":
            logits = self.decoder(self.dropout_layer(hidden_states), lm_inputs[1], node_type_ids, concept_ids, entity_mask)
            attn = None

        else:
            raise ValueError("")
        if self.fc_out_dim == 1:
            logits = logits.view(bs, nc)
        if not detail:
            return logits, attn
        else:
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig
            # edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            # edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        # edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        # edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]
        return edge_index, edge_type

    def batch_graph_edge2lmemb(self, edge2lmemb):
        concatenated_tensor = torch.cat(edge2lmemb, dim=1)
        # Transpose the tensor to get the shape [1476, 2, 768]
        transposed_tensor = concatenated_tensor.transpose(0, 1)
        return transposed_tensor


class LM_QAGNN_DataLoader(object):
    def __init__(
        self,
        args,
        train_statement_path,
        train_adj_path,
        dev_statement_path,
        dev_adj_path,
        test_statement_path,
        test_adj_path,
        batch_size,
        eval_batch_size,
        device,
        model_name,
        max_node_num=200,
        max_seq_length=128,
        is_inhouse=False,
        inhouse_train_qids_path=None,
        subsample=1.0,
        use_cache=True,
        test_key_list=None,
    ):  # ?New args
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.inhouse_train_qids_path = inhouse_train_qids_path
        self.return_entity_mask = (args.LMentemb) or (args.LMrelemb) or (args.ent_format == "atmark")

        if args.testk == -1:
            self.testk = test_key_list
        else:
            self.testk = args.testk

        model_type = MODEL_NAME_TO_CLASS[model_name]
        print("model_type", model_type)
        print("train_statement_path", train_statement_path)

        if args.clutrr:
            self._load_clutrr_data(args, train_statement_path, model_name, max_seq_length, model_type, self.testk)  #!-10 or -100
        else:
            self._load_regular_data(
                args,
                train_statement_path,
                train_adj_path,
                dev_statement_path,
                dev_adj_path,
                test_statement_path,
                test_adj_path,
                model_name,
                max_node_num,
                max_seq_length,
                model_type,
            )

        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        if args.valid_set:
            assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)
        if not isinstance(self.testk, list):
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)
        else:
            for test_qids, test_labels, test_encoder_data, test_decoder_data, test_adj_data in zip(
                self.test_qidsL, self.test_labelsL, self.test_encoder_dataL, self.test_decoder_dataL, self.test_adj_dataL
            ):
                assert all(len(test_qids) == len(test_adj_data[0]) == x.size(0) for x in [test_labels] + test_encoder_data + test_decoder_data)
        #!entity_mask conts not asserted

        assert 0.0 < subsample <= 1.0
        if subsample < 1.0:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def _load_clutrr_data(self, args, train_statement_path, model_name, max_seq_length, model_type, testk):
        assert isinstance(testk, list)  # +test for all reasoning length
        with open(train_statement_path, "rb") as f:
            datautil = pickle.load(f)
        if args.valid_set:
            if args.valid_set != 0.2:  # default 0.2
                indices = list(datautil.dataRows["train"].keys())
                mask_i = np.random.choice(indices, int(len(indices) * (1 - args.valid_set)), replace=False)
                val_ind = [datautil.dataRows["train"][i].id for i in indices if i not in set(mask_i)]
                train_ind = [datautil.dataRows["train"][i].id for i in indices if i in set(mask_i)]
                train_dataRows = datautil._select(datautil.dataRows["train"], train_ind)
                dev_dataRows = datautil._select(datautil.dataRows["train"], val_ind)
            else:
                train_dataRows = datautil._select(datautil.dataRows["train"], datautil.train_indices)
                dev_dataRows = datautil._select(datautil.dataRows["train"], datautil.val_indices)
            print("train_sample_num:", len(train_dataRows))
            print("dev_sample_num:", len(dev_dataRows))
        else:
            train_dataRows = datautil._select(datautil.dataRows["train"], datautil.train_indices + datautil.val_indices)
            dev_dataRows = []
        # if isinstance(testk,list):
        testk_files = list(datautil.dataRows["test"].keys())
        testk_dataRowsL = [[v for k, v in datautil.dataRows["test"][k_filename].items()] for k_filename in testk_files]

        # TODO
        if max_seq_length == -10:  # + same max seq len , as k=10's seq_len
            train_max_seqlen = max(datautil.max_word_length_dic["train"], datautil.max_word_length_dic[testk_files[-1]])
            # max_seq_lens=False
            raise ValueError("robust not implemented")
            test_max_seqlens = [train_max_seqlen for i in range(2, 11)]
        elif max_seq_length == -100:  # + AUTO max_seq_len list for all test_k reasoning length
            train_max_seqlen = datautil.max_word_length_dic["train"]
            test_max_seqlens = [max(train_max_seqlen, datautil.max_word_length_dic[k_filename]) for k_filename in testk_files]
        print("clutrr train_max_seqlen:", train_max_seqlen)
        print("clutrr test_max_seqlens:", test_max_seqlens)

        self.train_entity_mask, self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(
            train_dataRows, model_type, model_name, train_max_seqlen, return_entity_mask=self.return_entity_mask, ent_format=args.ent_format, concept_num=args.concept_num
        )
        *self.train_decoder_data, self.train_adj_data = load_clutrr_adj(train_dataRows)
        if args.valid_set:
            self.dev_entity_mask, self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(
                dev_dataRows, model_type, model_name, train_max_seqlen, return_entity_mask=self.return_entity_mask, ent_format=args.ent_format, concept_num=args.concept_num
            )
            *self.dev_decoder_data, self.dev_adj_data = load_clutrr_adj(dev_dataRows)
        print("num_choice", self.train_encoder_data[0].size(1))

        # if isinstance(testk,list):        #self.test_qids =False
        self.test_entity_maskL = []
        self.test_qidsL, self.test_labelsL, self.test_encoder_dataL = [], [], []
        self.test_decoder_dataL, self.test_adj_dataL = [], []
        for idx, k_number in enumerate(testk):
            test_entity_mask, test_qids, test_labels, *test_encoder_data = load_input_tensors(
                testk_dataRowsL[idx],
                model_type,
                model_name,
                test_max_seqlens[idx],
                return_entity_mask=self.return_entity_mask,
                ent_format=args.ent_format,
                concept_num=args.concept_num,
            )
            *test_decoder_data, test_adj_data = load_clutrr_adj(testk_dataRowsL[idx])
            for datalist, dataelm in zip(
                [self.test_entity_maskL, self.test_qidsL, self.test_labelsL, self.test_encoder_dataL, self.test_decoder_dataL, self.test_adj_dataL],
                [test_entity_mask, test_qids, test_labels, test_encoder_data, test_decoder_data, test_adj_data],
            ):
                datalist.append(dataelm)

    """
    if not isinstance(testk,list):
        testk_file = list(datautil.dataRows["test"].keys())[testk-2]
        testk_dataRows = [v for k,v in datautil.dataRows['test'][testk_file].items()]
        if max_seq_length==-10:
            max_seq_length = max(datautil.max_word_length_dic['train'],datautil.max_word_length_dic[testk_file])
    assert max_seq_length>0
    self.train_entity_mask, self.train_qids, self.train_labels, *self.train_encoder_data
     = load_input_tensors(train_dataRows , model_type, model_name, max_seq_length,LMentemb=args.LMentemb,ent_format=args.ent_format ,concept_num=args.concept_num )
    self.dev_entity_mask, self.dev_qids, self.dev_labels, *self.dev_encoder_data
     = load_input_tensors(dev_dataRows, model_type, model_name, max_seq_length,LMentemb=args.LMentemb,ent_format=args.ent_format ,concept_num=args.concept_num )
    *self.train_decoder_data, self.train_adj_data = load_clutrr_adj(train_dataRows)
    *self.dev_decoder_data, self.dev_adj_data = load_clutrr_adj(dev_dataRows)
    if not isinstance(testk,list):
        self.test_qids, self.test_labels, *self.test_encoder_data
         = load_input_tensors(testk_dataRows, model_type, model_name, max_seq_length,LMentemb=args.LMentemb,ent_format=args.ent_format ,concept_num=args.concept_num )
        *self.test_decoder_data, self.test_adj_data  = load_clutrr_adj(testk_dataRows )
        assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

    """

    def _load_regular_data(
        self, args, train_statement_path, train_adj_path, dev_statement_path, dev_adj_path, test_statement_path, test_adj_path, model_name, max_node_num, max_seq_length, model_type
    ):
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(
            train_statement_path, model_type, model_name, max_seq_length, LMentemb=args.LMentemb, ent_format=args.ent_format, concept_num=args.concept_num
        )
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(
            dev_statement_path, model_type, model_name, max_seq_length, LMentemb=args.LMentemb, ent_format=args.ent_format, concept_num=args.concept_num
        )
        num_choice = self.train_encoder_data[0].size(1)
        print("num_choice", num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)
        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)
        self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(
            test_statement_path, model_type, model_name, max_seq_length, LMentemb=args.LMentemb, ent_format=args.ent_format, concept_num=args.concept_num
        )
        *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, num_choice, args)

        if self.is_inhouse:
            with open(self.inhouse_train_qids_path, "r") as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, "test_qids") else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(
            self.args,
            "train",
            self.device0,
            self.device1,
            self.batch_size,
            train_indexes,
            self.train_qids,
            self.train_labels,
            tensors0=self.train_encoder_data,
            tensors1=self.train_decoder_data,
            adj_data=self.train_adj_data,
            entity_mask=self.train_entity_mask,
        )

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(
            self.args,
            "eval",
            self.device0,
            self.device1,
            self.eval_batch_size,
            torch.arange(len(self.train_qids)),
            self.train_qids,
            self.train_labels,
            tensors0=self.train_encoder_data,
            tensors1=self.train_decoder_data,
            adj_data=self.train_adj_data,
            entity_mask=self.train_entity_mask,
        )

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(
            self.args,
            "eval",
            self.device0,
            self.device1,
            self.eval_batch_size,
            torch.arange(len(self.dev_qids)),
            self.dev_qids,
            self.dev_labels,
            tensors0=self.dev_encoder_data,
            tensors1=self.dev_decoder_data,
            adj_data=self.dev_adj_data,
            entity_mask=self.dev_entity_mask,
        )

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(
                self.args,
                "eval",
                self.device0,
                self.device1,
                self.eval_batch_size,
                self.inhouse_test_indexes,
                self.train_qids,
                self.train_labels,
                tensors0=self.train_encoder_data,
                tensors1=self.train_decoder_data,
                adj_data=self.train_adj_data,
                entity_mask=self.train_entity_mask,
            )
        if not isinstance(self.testk, list):
            return MultiGPUSparseAdjDataBatchGenerator(
                self.args,
                "eval",
                self.device0,
                self.device1,
                self.eval_batch_size,
                torch.arange(len(self.test_qids)),
                self.test_qids,
                self.test_labels,
                tensors0=self.test_encoder_data,
                tensors1=self.test_decoder_data,
                adj_data=self.test_adj_data,
                entity_mask=self.test_entity_mask,
            )
        else:
            return [
                MultiGPUSparseAdjDataBatchGenerator(
                    self.args,
                    "eval",
                    self.device0,
                    self.device1,
                    self.eval_batch_size,
                    torch.arange(len(test_qids)),
                    test_qids,
                    test_labels,
                    tensors0=test_encoder_data,
                    tensors1=test_decoder_data,
                    adj_data=test_adj_data,
                    entity_mask=test_entity_mask,
                )
                for test_qids, test_labels, test_encoder_data, test_decoder_data, test_adj_data, test_entity_mask in zip(
                    self.test_qidsL, self.test_labelsL, self.test_encoder_dataL, self.test_decoder_dataL, self.test_adj_dataL, self.test_entity_maskL
                )
            ]


###############################################################################
# ############################## GNN architecture ##############################
###############################################################################

from torch.autograd import Variable


def make_one_hot(labels, C):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    """
    labels = labels.unsqueeze(1)
    labelsize0 = labels.size(0)
    one_hot = torch.FloatTensor(labelsize0, C).zero_()
    one_hot = one_hot.to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

# import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros


class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """

    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, lmvec2relemb=None, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.edge_encoder = edge_encoder
        self.lmvec2relemb = lmvec2relemb

        # For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2 * emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, edge2lmemb, return_attention_weights=False):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        #! edge2lmemb  [E,2,1024] ->
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        # Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)  # [E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype + 1).to(edge_vec.device)
        self_edge_vec[:, self.n_etype] = 1

        # node_type= node_type.cpu()
        # edge_index = edge_index.cpu()
        head_type = node_type[edge_index[0]]  # [E,] #head=src
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]

        if self.lmvec2relemb is not None:
            edge2lmemb = edge2lmemb.reshape(edge_index.size(1), -1)  #! [E,2,1024] ->  [E,2048]
            transformed_lmemb = self.lmvec2relemb(edge2lmemb)  #! [E,2048] ->  [E, emb_dim/2]
            self_edge_lmemb = torch.zeros(x.size(0), edge_embeddings.size(1)).to(edge_vec.device)
            transformed_lmemb = torch.cat([transformed_lmemb, self_edge_lmemb], dim=0)  #! [E, emb_dim/2] -> [E+N, emb_dim/2] NO LM information for self loop
            edge_embeddings = torch.cat([edge_embeddings, transformed_lmemb], dim=-1)

        # Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2 * self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce="sum")[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]


from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param


class CustomRGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomRGCNConv, self).__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_channels_l = in_channels
        self._use_segment_matmul_heuristic_output: Optional[bool] = None

        self.register_parameter("comp", None)
        self.l_weight = Parameter(torch.empty(in_channels, out_channels))
        self.root = Param(torch.empty(in_channels, out_channels))

        self.message_bias = Param(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        glorot(self.l_weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.message_bias)
        # Convert input features to a pair of node features or node indices.

    def forward(self, x, edge_index, edge_emb):
        # x_l: OptTensor = None
        x_l = x

        x_r: Tensor = x_l
        size = (x_l.size(0), x_r.size(0))

        assert edge_emb is not None

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        assert torch.is_floating_point(x_r)
        out = self.propagate(edge_index, x=x_l, edge_emb=edge_emb, size=size)
        root = self.root

        assert torch.is_floating_point(x_r)
        out = out + x_r @ root
        out = out + self.message_bias
        return out

    def message(self, x_j, edge_emb):
        # print(edge_index.size())
        # print(f"edge_emb_in_message: {edge_emb}")
        # r_weight = l_weight * edge_emb
        edge_emb = edge_emb.view(-1, 1, self.in_channels)
        r_weight = torch.bmm(edge_emb, self.l_weight.unsqueeze(0).repeat(edge_emb.size(0), 1, 1))
        # r_weight (num_edges, 1, hdim)
        # x_jの形状は(num_edges, hdim)。要素ごとの積でmessageにする？？
        r_weight = r_weight.squeeze(1)
        mout = r_weight * x_j
        return mout
