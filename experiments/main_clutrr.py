import random

try:
    from transformers import ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from modeling.modeling_qagnn import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *


DECODER_DEFAULT_LR = {"csqa": 1e-3, "obqa": 3e-4, "medqa_usmle": 1e-3, "clutrr": 1e-3}

from collections import defaultdict, OrderedDict
import numpy as np
from attrdict import AttrDict

import socket, os, subprocess, datetime
import wandb

print(socket.gethostname())
print("pid:", os.getpid())
# print ("conda env:", os.environ['CONDA_DEFAeULT_ENV'])
print("screen: %s" % subprocess.check_output("echo $STY", shell=True).decode("utf"))
print("gpu: %s" % subprocess.check_output("echo $CUDA_VISIBLE_DEVICES", shell=True).decode("utf"))
os.environ["WANDB_CACHE_DIR"] = os.path.join(os.path.dirname(os.path.abspath("")), "wandb")


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            entity_mask = None
            if isinstance(qids, tuple):
                entity_mask = qids[0]  #!
            logits, _ = model(*input_data, entity_mask=entity_mask)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples


global_args = 0


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval_detail"],
        help="run training or evaluation",
    )
    parser.add_argument("--save_dir", default="./saved_models/qagnn/", help="model output directory")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--load_model_path", default=None)

    # data
    parser.add_argument("--num_relation", default=38, type=int, help="number of relations")
    parser.add_argument("--train_adj", default=f"data/{args.dataset}/graph/train.graph.adj.pk")
    parser.add_argument("--dev_adj", default=f"data/{args.dataset}/graph/dev.graph.adj.pk")
    parser.add_argument("--test_adj", default=f"data/{args.dataset}/graph/test.graph.adj.pk")
    parser.add_argument("--use_cache", default=True, type=bool_flag, nargs="?", const=True, help="use cached data to accelerate data loading")

    # model architecture
    parser.add_argument("-k", "--k", default=5, type=int, help="perform k-layer message passing")
    parser.add_argument("--att_head_num", default=2, type=int, help="number of attention heads")
    parser.add_argument("--gnn_dim", default=100, type=int, help="dimension of the GNN layers")
    parser.add_argument("--fc_dim", default=200, type=int, help="number of FC hidden units")
    parser.add_argument("--fc_layer_num", type=int, help="number of FC layers")
    parser.add_argument("--fc_linear_sent", type=bool_flag)
    parser.add_argument("--compile_mlp_queryrep", default=False, type=bool_flag, help="")
    parser.add_argument("--freeze_ent_emb", default=True, type=bool_flag, nargs="?", const=True, help="freeze entity embedding layer")

    parser.add_argument("--max_node_num", default=200, type=int)
    parser.add_argument("--simple", default=False, type=bool_flag, nargs="?", const=True)
    parser.add_argument("--subsample", default=1.0, type=float)
    parser.add_argument("--init_range", default=0.02, type=float, help="stddev when initializing with normal distribution")
    # regularization
    parser.add_argument("--dropouti", type=float, default=0.2, help="dropout for embedding layer")
    parser.add_argument("--dropoutg", type=float, default=0.2, help="dropout for GNN layers")
    parser.add_argument("--dropoutf", type=float, default=0.2, help="dropout for fully-connected layers")

    # optimization
    parser.add_argument("-dlr", "--decoder_lr", default=DECODER_DEFAULT_LR[args.dataset], type=float, help="learning rate")
    parser.add_argument("-mbs", "--mini_batch_size", default=1, type=int)
    parser.add_argument("-ebs", "--eval_batch_size", default=2, type=int)
    parser.add_argument("--unfreeze_epoch", default=4, type=int)
    parser.add_argument("--refreeze_epoch", default=10000, type=int)
    parser.add_argument("--last_unfreeze_layer", default=0, type=int)
    parser.add_argument("--fp16", default=False, type=bool_flag, help="use fp16 training. this requires torch>=1.6.0")
    parser.add_argument("--drop_partial_batch", default=False, type=bool_flag, help="")
    parser.add_argument("--fill_partial_batch", default=False, type=bool_flag, help="")

    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="show this help message and exit")

    parser.add_argument("--concept_num", default=39, type=int)
    parser.add_argument("--concept_in_dim", default=100, type=int, help="")

    # ?CLUTRR
    parser.add_argument("--clutrr", action="store_true")
    parser.add_argument("--testk", default=2, type=int, help="")
    parser.add_argument("--valid_set", type=float)

    parser.add_argument("--data_id")
    parser.add_argument("--data_artifact_dir")
    parser.add_argument("--modelsavedir_local", default="saved_models/tmp_model")
    parser.add_argument("--model_art_pname")

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--wandbmode", default="online")
    parser.add_argument("--wnb_project")

    parser.add_argument("--LMentemb", default=False, type=bool_flag)
    parser.add_argument("--ent_format", default="figure")
    parser.add_argument("--LMrelemb", default=False, type=bool_flag)
    parser.add_argument("--edgeent_position", default=None)
    parser.add_argument("--initemb_method", choices=["onehot-LM", "concat-linear"])
    parser.add_argument("--sentence_level", default=False, type=bool_flag)

    parser.add_argument("--one_choice", type=bool_flag)
    parser.add_argument("--classify_relation", type=int)
    parser.add_argument("--edge_scoring", type=bool_flag)
    parser.add_argument("--edge_pruning_ratio", type=float)
    parser.add_argument("--edge_pruning_order", choices=["klogk", "linear", "const", "disabled"])
    parser.add_argument("--start_pruning_epoch", type=int)
    parser.add_argument("--scored_edge_norm", choices=["disabled", "batch", "layer"])
    parser.add_argument("--decoder_model", default="qagnn", choices=["qagnn", "compile", "RGCN", "MLP", "LSTM-MLP"])

    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= "1.6.0")

    if args.mode == "train":
        global global_args
        global_args = args
        if args.sweep:
            sweep_configuration = {
                "method": "grid",
                "metric": {"goal": "maximize", "name": "dev_acc"},
                "parameters": {
                    "seed": {"values": [0, 1]},
                    # "scored_edge_norm": {"values": ["batch", "layer"]},
                    # "encoder": {"values": ["bert-base-uncased", "roberta-large"]},
                    # "data_id": {"values": []},  # "089907f8", "db9b8f04" "X523348e6"]
                    # "decoder_model": {"values": ["MLP", "LSTM-MLP"]},
                    #'gnn_dim': {'values': [100]},
                    #'decoder_k': {'values': [3]},
                    #'freeze_ent_emb': {'values': [True,False]},
                    # "model_architecture_artifact": {"values": ["qagnn-ent", "qagnn-relonly", "qagnn-entrel"]},
                },
            }
            # ?if not (args.model_art_pname=='' or args.model_art_pname==''):
            # ?    args.edgeent_position = None
            # count=9
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wnb_project)
            wandb.agent(sweep_id, function=asweep_train)  # count=count
        else:
            asweep_train()
    elif args.mode == "eval_detail":
        # raise NotImplementedError
        eval_detail(args)
    else:
        raise ValueError("Invalid mode")


def asweep_train():
    args = global_args
    with wandb.init(  # Set the project where this run will be logged
        project=args.wnb_project,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # name=f"experiment_train_seed{args.seed}",
        # Track hyperparameters and run metadata
        config={
            "seed": args.seed,
            "model_architecture_artifact": args.model_art_pname,
            "decoder_model": args.decoder_model,
            "data_id": args.data_id.split("_")[1],
            "encoder": args.encoder,
            "dataset": args.dataset,
            # "LMentemb":args.LMentemb,
            # "ent_format":args.ent_format ,
            # "LMrelemb":args.LMrelemb,
            "batch_size": args.batch_size,
            "gnn_dim": args.gnn_dim,
            "decoder_k": args.k,
            "edgeent_position": args.edgeent_position,
            "sentence_level": args.sentence_level,
            "fc_linear_sent": args.fc_linear_sent,
            "compile_mlp_queryrep": args.compile_mlp_queryrep,
            "initmeb_method": args.initemb_method,
            "unfreeze_epoch": args.unfreeze_epoch,
            "last_unfreeze_layer": args.last_unfreeze_layer,
            "concept_in_dim": args.concept_in_dim,
            "testk": args.testk,
            "max_seq_len": args.max_seq_len,
            "n_epochs": args.n_epochs,
            "fc_layer_num": args.fc_layer_num,
            "valid_set": args.valid_set,
            "edge_scoring": args.edge_scoring,
            "edge_pruning_order": args.edge_pruning_order,
            "edge_pruning_ratio": args.edge_pruning_ratio,
            "scored_edge_norm": args.scored_edge_norm,
            "start_pruning_epoch": args.start_pruning_epoch,
        },
        group="freeze,LMentemb_sweep",
        job_type=args.mode,
        tags=[],
        mode=args.wandbmode,
    ) as run:
        # train(args,run)
        train(args, run)
    # wandb.finish()


def train(args, run):
    # args.k=wandb.config["decoder_k"]
    # args.gnn_dim=wandb.config["gnn_dim"]
    # args.model_art_pname = wandb.config["model_architecture_artifact"]
    args.seed = wandb.config["seed"]
    # args.encoder = wandb.config["encoder"]
    # args.data_id = "data_" + wandb.config["data_id"]
    args.decoder_model = wandb.config["decoder_model"]
    # args.scored_edge_norm = wandb.config["scored_edge_norm"]
    run.tags = [args.model_art_pname, args.decoder_model]  # args.data_id.split("_")[1], args.encoder
    run.define_metric("epoch")
    run.define_metric("dev/acc", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    # run.define_metric("test2_acc", summary="max")

    if args.one_choice:
        fc_out_dim = args.classify_relation
        n_ntype = 5  # 1,4 for source,target node
        suffix = "onechoice"
    else:
        fc_out_dim = 1
        n_ntype = 4
        suffix = "processed"

    if wandb.config["model_architecture_artifact"] == "qagnn-base":
        args.LMentemb = False
        args.ent_format = "atmark"
        args.LMrelemb = False
        #!args.edgeent_position = None
        #!if args.data_artifact_dir=='auto':
        args.data_artifact_dir = f"{args.data_id}_atmarkEnt_{suffix}"
        args.train_statements = f"data/clutrr/{args.data_artifact_dir}/after_processTESTdata.pkl"
    elif wandb.config["model_architecture_artifact"] == "qagnn-ent":
        args.LMentemb = True
        args.ent_format = "atmark"
        args.LMrelemb = False

        args.data_artifact_dir = f"{args.data_id}_atmarkEnt_{suffix}"
        args.train_statements = f"data/clutrr/{args.data_artifact_dir}/after_processTESTdata.pkl"
    elif wandb.config["model_architecture_artifact"] == "qagnn-relonly":
        args.LMentemb = False
        args.ent_format = "relation"
        args.LMrelemb = True
        # args.edgeent_position = None

        args.data_artifact_dir = f"{args.data_id}_relationEnt_{suffix}"
        args.train_statements = f"data/clutrr/{args.data_artifact_dir}/after_processTESTdata.pkl"
    elif wandb.config["model_architecture_artifact"] == "qagnn-entrel":
        args.LMentemb = True
        args.ent_format = "relation"
        args.LMrelemb = True
        # args.edgeent_position = None

        args.data_artifact_dir = f"{args.data_id}_relationEnt_{suffix}"
        args.train_statements = f"data/clutrr/{args.data_artifact_dir}/after_processTESTdata.pkl"
    else:
        raise ValueError("invalid model architecture")
    print(args)
    # run.group=f"freeze-ee:{args.freeze_ent_emb}"
    if wandb.config["data_id"] == "db9b8f04":
        traink = 4
        test_key_list = [f"test/{testklen}" for testklen in range(2, 11)]
        robust_exp = False
    elif wandb.config["data_id"] == "089907f8":
        traink = 3
        test_key_list = [f"test/{testklen}" for testklen in range(2, 11)]
        robust_exp = False
    elif wandb.config["data_id"] in ["7c5b0e70"]:
        test_key_list = [f"test/{testklen}" for testklen in ["clean2", "clean3", "sup3", "irr3", "disc3"]]
        robust_exp = True
    elif wandb.config["data_id"] in ["06b8f2a1"]:
        test_key_list = [f"test/{testklen}" for testklen in ["clean3", "sup2", "sup3", "irr3", "disc3"]]
        robust_exp = True
    elif wandb.config["data_id"] == "523348e6":
        test_key_list = [f"test/{testklen}" for testklen in ["clean3", "sup3", "irr2", "irr3", "disc3"]]
        robust_exp = True
    elif wandb.config["data_id"] in ["d83ecc3e"]:
        test_key_list = [f"test/{testklen}" for testklen in ["clean3", "sup3", "irr3", "disc2", "disc3"]]
        robust_exp = True
    else:
        raise ValueError("invalid data_id")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    model_path = os.path.join(args.modelsavedir_local, "model.pt")
    check_path(model_path)
    if args.sweep:
        model_art_name = f"sweep_{run.sweep_id}_{args.model_art_pname}"
    else:
        model_art_name = f"{args.model_art_pname}_{run.id}"

    #!export_config(args, config_path)

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    # cp_emb = [np.load(path) for path in args.ent_emb_paths]
    # cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    # concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    cp_emb = None
    # concept_num=39 #!
    # concept_num=799273
    concept_num = args.concept_num
    # concept_dim=args.gnn_dim
    # concept_dim=100
    # concept_dim=1024
    if args.LMentemb:
        concept_dim = args.concept_in_dim // 2
    else:
        concept_dim = args.concept_in_dim
    print("| num_concepts: {} |".format(concept_num))

    # try:
    if True:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
        dataset = LM_QAGNN_DataLoader(
            args,
            args.train_statements,
            args.train_adj,
            args.dev_statements,
            args.dev_adj,
            args.test_statements,
            args.test_adj,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            device=(device0, device1),
            model_name=args.encoder,
            max_node_num=args.max_node_num,
            max_seq_length=args.max_seq_len,
            is_inhouse=args.inhouse,
            inhouse_train_qids_path=args.inhouse_train_qids,
            subsample=args.subsample,
            use_cache=args.use_cache,
            test_key_list=test_key_list,
        )
        assert args.clutrr
        run.use_artifact(f"koki-tomei/qagnn-train-try/clutrr_{args.data_artifact_dir}:latest")
        # koki-tomei/qagnn-train-try/clutrr_data_089907f8_preprocess:latest

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################
        print("args.num_relation", args.num_relation)
        model = LM_QAGNN(
            args,
            args.encoder,
            k=args.k,
            n_ntype=n_ntype,
            n_etype=args.num_relation,
            n_concept=concept_num,
            concept_dim=args.gnn_dim,
            concept_in_dim=concept_dim,
            n_attention_head=args.att_head_num,
            fc_dim=args.fc_dim,
            n_fc_layer=args.fc_layer_num,
            p_emb=args.dropouti,
            p_gnn=args.dropoutg,
            p_fc=args.dropoutf,
            fc_out_dim=fc_out_dim,
            pretrained_concept_emb=cp_emb,
            freeze_ent_emb=args.freeze_ent_emb,
            init_range=args.init_range,
            encoder_config={},
        )
        if args.load_model_path:
            print(f"loading and initializing model from {args.load_model_path}")
            model_state_dict, old_args = torch.load(args.load_model_path, map_location=torch.device("cpu"))
            model.load_state_dict(model_state_dict)

        model.encoder.to(device0)
        model.decoder.to(device1)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    grouped_parameters = [
        {"params": [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, "lr": args.encoder_lr},
        {"params": [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": args.encoder_lr},
        {"params": [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay, "lr": args.decoder_lr},
        {"params": [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == "fixed":
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == "warmup_constant":
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == "warmup_linear":
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    print("parameters:")
    print(" ===decoder===")
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print("\t{:45}\ttrainable\t{}\tdevice:{}".format(name, param.size(), param.device))
        else:
            print("\t{:45}\tfixed\t{}\tdevice:{}".format(name, param.size(), param.device))
    dnum_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print("\tdecoder (unfreeze) total:", dnum_params)
    if args.loss == "margin_rank":
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction="mean")
    elif args.loss == "cross_entropy":
        loss_func = nn.CrossEntropyLoss(reduction="mean")

    def compute_loss(logits, labels):
        if args.loss == "margin_rank":
            num_choice = logits.size(1)
            flat_logits = logits.view(-1)
            correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
            correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
            wrong_logits = flat_logits[correct_mask == 0]
            y = wrong_logits.new_ones((wrong_logits.size(0),))
            loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
        elif args.loss == "cross_entropy":
            loss = loss_func(logits, labels)
        return loss

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print("-" * 71)
    if args.fp16:
        print("Using fp16 training")
        scaler = torch.cuda.amp.GradScaler()

    global_step, dev_best_epoch, allk_best_epoch = 0, 0, 0
    final_test_acc, total_loss = 0.0, 0.0
    dev_best_acc, allk_best_mean = 0.0, 0.0
    start_time = time.time()
    wandb.watch(model, compute_loss, log="all", log_freq=10)
    model.train()
    freeze_net(model.encoder, args.last_unfreeze_layer)

    for name, param in model.encoder.named_parameters():
        if param.requires_grad:
            print("\t{:45}\ttrainable\t{}\tdevice:{}".format(name, param.size(), param.device))
    enum_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print("\tencoder unfreeze total:", enum_params)
    wenum_params = sum(p.numel() for p in model.encoder.parameters())
    print("\twhole encoder total:", wenum_params)
    print("\twhole unfreeze total:", dnum_params + enum_params)

    if True:
        best_test_accuracies = {}
        test_accs_devbest = {}
        best_test_epochs = {}
        test_accs_on_allkbest_epoch = {}
        # try:
        for epoch_id in range(args.n_epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            model.cur_epoch = epoch_id
            model.train()
            for qids, labels, *input_data in dataset.train():
                if args.LMentemb or args.LMrelemb or (args.ent_format == "atmark"):  #! qids is touple
                    entity_mask = qids[0]  #! entity_idではなくconcept_idからのマッピング、つまり+2されたあと
                    qids = qids[1]
                else:
                    entity_maskab = None
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    if args.LMentemb or args.LMrelemb or args.ent_format == "atmark":
                        entity_maskab = entity_mask[a:b]
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer, entity_mask=entity_maskab)
                            loss = compute_loss(logits, labels[a:b])
                    else:
                        logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer, entity_mask=entity_maskab)
                        loss = compute_loss(logits, labels[a:b])
                    loss = loss * (b - a) / bs
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print("| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |".format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                    run.log({"step": global_step, "train/loss": total_loss})
                    total_loss = 0
                    start_time = time.time()

                global_step += 1

            model.eval()
            # ?save_test_preds = args.save_model
            # ?if not save_test_preds:
            if args.testk > 0:
                test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
            else:
                test_accuracies = list(map(lambda test_set: evaluate_accuracy(test_set, model), dataset.test()))
                test_accs_dict = dict(zip(test_key_list, test_accuracies))
            """else:  # TODO:test preds table
                eval_set = dataset.test()
                total_acc = []
                count = 0
                preds_path = os.path.join(args.save_dir, "test_e{}_preds.csv".format(epoch_id))
                with open(preds_path, "w") as f_preds:
                    with torch.no_grad():
                        for qids, labels, *input_data in tqdm(eval_set):
                            count += 1
                            (logits, _, concept_ids, node_type_ids, edge_index, edge_type) = model(*input_data, detail=True)
                            predictions = logits.argmax(1)  # [bsize, ]
                            preds_ranked = (-logits).argsort(1)  # [bsize, n_choices]
                            for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(
                                zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)
                            ):
                                acc = int(pred.item() == label.item())
                                print(
                                    "{},{}".format(qid, chr(ord("A") + pred.item())),
                                    file=f_preds,
                                )
                                f_preds.flush()
                                total_acc.append(acc)
                test_acc = float(sum(total_acc)) / len(total_acc)"""

            if args.valid_set:
                dev_acc = evaluate_accuracy(dataset.dev(), model)
            else:
                assert ValueError("robust not implemented")
                dev_acc = np.mean([v for k, v in test_accs_dict.items() if int(k.split("/")[-1]) <= traink])

            print("-" * 71)
            assert args.testk < 0
            if args.testk > 0:
                pass
                """print("| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |".format(epoch_id, global_step, dev_acc, test_acc))
                run.log({"epoch": epoch_id, "global_step": global_step, "dev/acc": dev_acc, f"test/{args.testk}_acc": test_acc})
                if dev_acc >= best_dev_acc:
                    best_dev_acc = dev_acc
                    final_test_acc = test_acc
                    best_dev_epoch = epoch_id
                    if args.save_model:
                        torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                        print(f"model saved to {model_path}.{epoch_id}")
                else:
                    if args.save_model:
                        torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                        print(f"model saved to {model_path}.{epoch_id}")
                model.train()
                start_time = time.time()

                if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                    break"""
            else:
                print("| epoch {:3} | step {:5} | dev_acc {:7.4f} |".format(epoch_id, global_step, dev_acc))
                current_accs = {"epoch": epoch_id, "global_step": global_step, "dev/acc": dev_acc}

                if not robust_exp:
                    allk_mean = np.mean(list(test_accs_dict.values()))
                    interpolate_mean = np.mean([v for k, v in test_accs_dict.items() if int(k.split("/")[-1]) <= traink])
                    gen_mean = np.mean([v for k, v in test_accs_dict.items() if int(k.split("/")[-1]) > traink])
                    current_accs["test/allk_mean"] = allk_mean
                    current_accs["test/interpolate_mean"] = interpolate_mean
                    current_accs["test/gen_mean"] = gen_mean
                    if allk_mean >= allk_best_mean:
                        allk_best_mean = allk_mean
                        allk_best_epoch = epoch_id
                        allk_best_devacc = dev_acc
                        for key, v in test_accs_dict.items():
                            test_accs_on_allkbest_epoch[key] = v
                            run.summary[f"{key}_allkbest_epoch_acc"] = test_accs_on_allkbest_epoch[key]
                        run.summary["test/allk_best_allkmean"] = allk_best_mean
                        run.summary["test/allk_best_epoch"] = allk_best_epoch
                        run.summary["dev/allk_best_devacc"] = allk_best_devacc
                else:
                    clean_mean = np.mean([v for k, v in test_accs_dict.items() if (k.split("/")[-1][:-1] == "clean")])
                    sup_mean = np.mean([v for k, v in test_accs_dict.items() if (k.split("/")[-1][:-1] == "sup")])
                    irr_mean = np.mean([v for k, v in test_accs_dict.items() if (k.split("/")[-1][:-1] == "irr")])
                    disc_mean = np.mean([v for k, v in test_accs_dict.items() if (k.split("/")[-1][:-1] == "disc")])

                if dev_acc >= dev_best_acc:
                    dev_best_acc = dev_acc
                    dev_best_epoch = epoch_id
                    torch.save([model.state_dict(), args], model_path)  # ? best dev model save
                    for key, v in test_accs_dict.items():
                        test_accs_devbest[key] = v
                        run.summary[f"{key}_devbest_epoch_acc"] = test_accs_devbest[key]
                    run.summary["dev/best_acc"] = dev_best_acc
                    run.summary["dev/best_epoch"] = dev_best_epoch
                    if not robust_exp:
                        dev_best_allkmean = allk_mean
                        dev_best_interpolatemean = interpolate_mean
                        dev_best_genmean = gen_mean
                        run.summary["test/dev_best_allkmean"] = dev_best_allkmean
                        run.summary["test/dev_best_interpolate_mean"] = dev_best_interpolatemean
                        run.summary["test/dev_best_gen_mean"] = dev_best_genmean
                    else:
                        dev_best_cleanmean = clean_mean
                        dev_best_supmean = sup_mean
                        dev_best_irrmean = irr_mean
                        dev_best_discmean = disc_mean
                        run.summary["test/dev_best_cleanmean"] = dev_best_cleanmean
                        run.summary["test/dev_best_supmean"] = dev_best_supmean
                        run.summary["test/dev_best_irrmean"] = dev_best_irrmean
                        run.summary["test/dev_best_discmean"] = dev_best_discmean

                for key, v in test_accs_dict.items():
                    print("| {}_acc {:7.4f} |".format(key, v))
                    current_accs[f"{key}_acc"] = v
                    if key not in best_test_accuracies or v > best_test_accuracies[key]:
                        best_test_accuracies[key] = v
                        best_test_epochs[key] = epoch_id
                        run.summary[f"{key}_best_acc"] = best_test_accuracies[key]
                        run.summary[f"{key}_best_epoch"] = best_test_epochs[key]
                run.log(current_accs)

                #! now ES with dev, not all test acc
                # if all(epoch_id - best_test_epochs[testi2] >= args.max_epochs_before_stop for testi2 in best_test_epochs) and (epoch_id > args.unfreeze_epoch):
                if (epoch_id - dev_best_epoch >= args.max_epochs_before_stop) and (dev_best_epoch > args.start_pruning_epoch):
                    break
            print("-" * 71)

        if not robust_exp:
            wnb_table_data = []
            for key, value in test_accs_devbest.items():
                k_length = int(key.split("/")[1])  # "test/x"からxを抽出して整数に変換
                wnb_table_data.append([k_length, value])
            # wandb.Tableを作成
            my_table = wandb.Table(columns=["k_length", "accuracy"], data=wnb_table_data)
            run.log({"accs_on_bestdev": my_table})

            wnb_table_data = []
            for key, value in test_accs_on_allkbest_epoch.items():
                k_length = int(key.split("/")[1])  # "test/x"からxを抽出して整数に変換
                wnb_table_data.append([k_length, value])
            # wandb.Tableを作成
            my_table = wandb.Table(columns=["k_length", "accuracy"], data=wnb_table_data)
            run.log({"accs_on_allkbest": my_table})

            wnb_table_mean = []
            allk_mean = np.mean(list(test_accs_devbest.values()))
            assert allk_mean == dev_best_allkmean
            gen_mean = np.mean([v for k, v in test_accs_devbest.items() if int(k.split("/")[-1]) > traink])
            wnb_table_mean.append([dev_best_acc, allk_mean, gen_mean])
            mean_table = wandb.Table(columns=["dev_acc", "allk_mean", "gen_mean"], data=wnb_table_mean)  # TODO robust test
            run.log({"mean_accs_bestdev": mean_table})

            wnb_table_mean = []
            allk_mean = np.mean(list(test_accs_on_allkbest_epoch.values()))
            assert allk_mean == allk_best_mean
            gen_mean = np.mean([v for k, v in test_accs_on_allkbest_epoch.items() if int(k.split("/")[-1]) > traink])
            wnb_table_mean.append([allk_best_devacc, allk_mean, gen_mean])
            mean_table = wandb.Table(columns=["dev_acc", "allk_mean", "gen_mean"], data=wnb_table_mean)
            run.log({"mean_accs_allkbest": mean_table})
        else:
            wnb_table_data = []
            for key, value in test_accs_devbest.items():
                k_setting = key.split("/")[1]
                wnb_table_data.append([k_setting, value])
            # wandb.Tableを作成
            my_table = wandb.Table(columns=["k_setting", "accuracy"], data=wnb_table_data)
            run.log({"accs_robust_bestdev": my_table})

        if args.testk < 0 and epoch_id > 10:
            for i in best_test_accuracies.keys():
                print("Best accuracy for {}: {:.4f} achieved at epoch {}".format(i, best_test_accuracies[i], best_test_epochs[i]))
            # model_art = wandb.Artifact(model_art_name, type="model", description="args  :  " + str(dict(vars(args))), metadata={"epoch_id": dev_best_epoch}.update(wandb.config))
            # model_art.add_file(model_path)
            # run.log_artifact(model_art, aliases=[f"epoch_{dev_best_epoch}"])  # TODO  aliases=['latest','best-ap50']
        if args.sweep:
            del model
            del loss
            torch.cuda.empty_cache()

    # except (KeyboardInterrupt, RuntimeError) as e:
    #     print(e)


def eval_detail(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path

    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print("| num_concepts: {} |".format(concept_num))

    model_state_dict, old_args = torch.load(model_path, map_location=torch.device("cpu"))
    model = LM_QAGNN(
        old_args,
        old_args.encoder,
        k=old_args.k,
        n_ntype=4,
        n_etype=old_args.num_relation,
        n_concept=concept_num,
        concept_dim=old_args.gnn_dim,
        concept_in_dim=concept_dim,
        n_attention_head=old_args.att_head_num,
        fc_dim=old_args.fc_dim,
        n_fc_layer=old_args.fc_layer_num,
        p_emb=old_args.dropouti,
        p_gnn=old_args.dropoutg,
        p_fc=old_args.dropoutf,
        pretrained_concept_emb=cp_emb,
        freeze_ent_emb=old_args.freeze_ent_emb,
        init_range=old_args.init_range,
        encoder_config={},
    )
    model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    model.encoder.to(device0)
    model.decoder.to(device1)
    model.eval()

    statement_dic = {}
    for statement_path in (
        args.train_statements,
        args.dev_statements,
        args.test_statements,
    ):
        statement_dic.update(load_statement_dict(statement_path))

    #!use_contextualized = "lm" in old_args.ent_emb

    print("inhouse?", args.inhouse)

    print("args.train_statements", args.train_statements)
    print("args.dev_statements", args.dev_statements)
    print("args.test_statements", args.test_statements)
    print("args.train_adj", args.train_adj)
    print("args.dev_adj", args.dev_adj)
    print("args.test_adj", args.test_adj)

    dataset = LM_QAGNN_DataLoader(
        args,
        args.train_statements,
        args.train_adj,
        args.dev_statements,
        args.dev_adj,
        args.test_statements,
        args.test_adj,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=(device0, device1),
        model_name=old_args.encoder,
        max_node_num=old_args.max_node_num,
        max_seq_length=old_args.max_seq_len,
        is_inhouse=args.inhouse,
        inhouse_train_qids_path=args.inhouse_train_qids,
        subsample=args.subsample,
        use_cache=args.use_cache,
    )

    save_test_preds = args.save_model
    dev_acc = evaluate_accuracy(dataset.dev(), model)
    print("dev_acc {:7.4f}".format(dev_acc))
    if not save_test_preds:
        test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
    else:
        eval_set = dataset.test()
        total_acc = []
        count = 0
        dt = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        preds_path = os.path.join(args.save_dir, "test_preds_{}.csv".format(dt))
        with open(preds_path, "w") as f_preds:
            with torch.no_grad():
                for qids, labels, *input_data in tqdm(eval_set):
                    count += 1
                    (
                        logits,
                        _,
                        concept_ids,
                        node_type_ids,
                        edge_index,
                        edge_type,
                    ) = model(*input_data, detail=True)
                    predictions = logits.argmax(1)  # [bsize, ]
                    preds_ranked = (-logits).argsort(1)  # [bsize, n_choices]
                    for i, (
                        qid,
                        label,
                        pred,
                        _preds_ranked,
                        cids,
                        ntype,
                        edges,
                        etype,
                    ) in enumerate(
                        zip(
                            qids,
                            labels,
                            predictions,
                            preds_ranked,
                            concept_ids,
                            node_type_ids,
                            edge_index,
                            edge_type,
                        )
                    ):
                        acc = int(pred.item() == label.item())
                        print(
                            "{},{}".format(qid, chr(ord("A") + pred.item())),
                            file=f_preds,
                        )
                        f_preds.flush()
                        total_acc.append(acc)
        test_acc = float(sum(total_acc)) / len(total_acc)

        print("-" * 71)
        print("test_acc {:7.4f}".format(test_acc))
        print("-" * 71)


if __name__ == "__main__":
    main()
