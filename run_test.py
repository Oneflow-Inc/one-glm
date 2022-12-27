import time
from utils import Timers
from configure_data import prepare_tokenizer
import oneflow as flow
import oneflow.nn as nn
from oneflow.nn.parallel import DistributedDataParallel as ddp
from arguments import get_args
from model import GLMModel
import sys
import mpu
from pretrain_glm import get_train_val_test_data, train
sys.path.append(".")


def load_torch_model(model, path):
    import torch

    torch_params = torch.load(path, map_location="cpu")
    flow_params = {}
    for k in torch_params.keys():
        flow_params[k] = flow.Tensor(torch_params[k].numpy().astype("float32"))
    model.load_state_dict(flow_params, strict=False)
    print("load pretraining model succeed!")


def get_model(args):
    assert args.mode in ["eager", "graph"]

    model = GLMModel(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        embedding_dropout_prob=args.hidden_dropout,
        attention_dropout_prob=args.attention_dropout,
        output_dropout_prob=args.hidden_dropout,
        max_sequence_length=args.max_position_embeddings,
        max_memory_length=args.mem_length,
        checkpoint_activations=args.checkpoint_activations,
        checkpoint_num_layers=args.checkpoint_num_layers,
        parallel_output=True,
        relative_encoding=args.transformer_xl,
        block_position_encoding=args.block_lm and not args.masked_lm,
        output_predict=True,
        spell_length=None,
        spell_func=args.prompt_func,
        attention_scale=args.attention_scale,
    )

    if args.debug_loss:
        # load pretrain
        load_torch_model(model, path=args.debug_pretrain_model)
        model.eval()
    else:
        model.train()

    if args.mode == "graph":
        placement = flow.env.all_device_placement("cuda")
        model = model.to_global(placement=placement, sbp=flow.sbp.broadcast)
    elif args.mode == "eager":
        model.cuda()
        if flow.env.get_world_size() > 1:
            model = ddp(model)
    else:
        raise NotImplementedError

    optimizer = flow.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                betas=(args.adam_beta1, args.adam_beta2),
                                eps=args.adam_eps)
    # optimizer = flow.optim.SGD(
    #     model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0,
    # )
    lr_scheduler = flow.optim.lr_scheduler.StepLR(optimizer, step_size=100000)

    if args.mode == "eager":
        return model, optimizer, lr_scheduler
    if args.mode == "graph":
        graph_model = mpu.GLMGraph(args, model, optimizer, lr_scheduler)
        return graph_model, None, None
    else:
        raise NotImplementedError


def get_train_dataloader(args):
    tokenizer = prepare_tokenizer(args)
    train_data, val_data, test_data, = get_train_val_test_data(args, tokenizer)
    train_data_iterator = iter(train_data)
    return train_data_iterator


if __name__ == "__main__":
    args = get_args()
    model, optimizer, lr_scheduler = get_model(args)
    train_data_iterator = get_train_dataloader(args)
    timers = Timers()
    train(model, optimizer,
          lr_scheduler,
          (train_data_iterator, None),
          (None, None),
          timers, args, summary_writer=None)
