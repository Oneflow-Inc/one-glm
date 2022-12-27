import mpu
import deepspeed
import oneflow  as flow
# from apex.optimizers import FusedAdam as Adam
from oneflow.optim import Adam 
from oneflow import distributed as dist
from oneflow.nn.parallel import DistributedDataParallel as ddp
# from fp16 import FP16_Module, FP16_Optimizer, DynamicLossScaler
from fp16.loss_scaler import DynamicLossScaler
from learning_rates import AnnealingLR
from model import GLMModel
# from model import GLMForMultiTokenCloze, GLMForMultiTokenClozeFast, GLMForSingleTokenCloze, GLMForSequenceClassification
# from model import PyTorchDistributedDataParallel as TorchDDP, DistributedDataParallel as LocalDDP
from model.modeling_bert import BertForMultipleChoice, BertForSequenceClassification
from utils import print_rank_0, get_checkpoint_name, get_checkpoint_iteration


def load_pretrained(model, checkpoint_path, args, task_tokens=None):
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading pretrained model {}'.format(
            flow.env.get_rank(), checkpoint_name))
    # Load the checkpoint.
    sd = flow.load(checkpoint_name, map_location='cpu')
    if args.deepspeed:
        model = model.module
    if isinstance(model, TorchDDP):
        model = model.module
    if isinstance(model, FP16_Module):
        model = model.module
    if hasattr(model, "model"):
        model = model.model

    # Model.
    def extend_embedding_weights(state_weights, model_weights):
        original_length = state_weights.shape[0]
        assert original_length <= args.max_position_embeddings + 1
        new_weights = model_weights.clone()
        new_weights[:original_length] = state_weights
        return new_weights

    if args.block_lm:
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            position_weights = sd['module']["transformer.position_embeddings.weight"]
            if args.max_position_embeddings + 1 > position_weights.shape[0]:
                sd['module']["transformer.position_embeddings.weight"] = extend_embedding_weights(
                    position_weights, model.state_dict()["transformer.position_embeddings.weight"].data)
                print_rank_0(f"Extend position embedding to {args.max_position_embeddings + 1}")
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            block_position_weights = sd['module']["transformer.block_position_embeddings.weight"]
            if args.max_position_embeddings + 1 > block_position_weights.shape[0]:
                sd['module']["transformer.block_position_embeddings.weight"] = extend_embedding_weights(
                    block_position_weights,
                    model.state_dict()["transformer.block_position_embeddings.weight"].data)
                print_rank_0(f"Extend block position embedding to {args.max_position_embeddings + 1}")
    missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
    if missing_keys or unexpected_keys:
        print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
    if args.continuous_prompt and args.prompt_init:
        model.prompt_spell.init_embedding(model.word_embeddings.weight.data, task_tokens)

def check_mode(args,model):
    if args.mode == "graph":
        placement = flow.env.all_device_placement("cuda")
        model = model.to_global(placement=placement,
                                    sbp=flow.sbp.broadcast
                                    )
    elif args.mode == "eager":
        model.cuda()
        if flow.env.get_world_size() > 1:
            model = ddp(model)
    else:
        raise NotImplementedError
    
def get_model(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Build the model."""
    assert args.mode in ["eager", "graph"]
    print_rank_0('building GPT2 model ...')
    # print(f'{args.pretrained_bert=}')
    # False
    if args.pretrained_bert:
        if model_type == "multiple_choice":
            model = BertForMultipleChoice.from_pretrained(args.tokenizer_model_type,
                                                          cache_dir=args.cache_dir,
                                                          fp32_layernorm=args.fp32_layernorm,
                                                          fp32_embedding=args.fp32_embedding,
                                                          layernorm_epsilon=args.layernorm_epsilon)
        elif model_type == "classification":
            model = BertForSequenceClassification.from_pretrained(args.tokenizer_model_type,
                                                                  cache_dir=args.cache_dir,
                                                                  fp32_layernorm=args.fp32_layernorm,
                                                                  fp32_embedding=args.fp32_embedding,
                                                                  layernorm_epsilon=args.layernorm_epsilon,
                                                                  num_labels=num_labels)
        else:
            raise NotImplementedError
    else:
        output_predict, paralle_output = True, True
        if (model_type == "multiple_choice" or model_type == "classification") and not args.cloze_eval:
            output_predict = False
        if model_type is not None:
            paralle_output = False
        if spell_length is not None:
            print_rank_0(f"Continuous spell length {spell_length}")
        model = GLMModel(num_layers=args.num_layers,
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
                         parallel_output=paralle_output,
                         relative_encoding=args.transformer_xl,
                         block_position_encoding=args.block_lm and not args.masked_lm,
                         output_predict=output_predict,
                         spell_length=spell_length,
                         spell_func=args.prompt_func,
                         attention_scale=args.attention_scale)
        if args.freeze_transformer:
            model.freeze_transformer(tune_prefix_layers=args.tune_prefix_layers)
        if model_type is not None:
            if model_type == 'multiple_choice':
                if args.cloze_eval:
                    if multi_token:
                        if args.fast_decode:
                            model = GLMForMultiTokenClozeFast(model, length_penalty=args.length_penalty)
                        else:
                            model = GLMForMultiTokenCloze(model, length_penalty=args.length_penalty)
                    else:
                        model = GLMForSingleTokenCloze(model, take_softmax=args.adapet)
                else:
                    model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                         num_class=num_labels)
            elif model_type == 'classification':
                model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                     num_class=num_labels)
            elif model_type == 'generation':
                pass
            else:
                raise NotImplementedError(model_type)

    print(f'{(mpu.get_data_parallel_rank() == 0)=}')
    # True
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(flow.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    # if not args.deepspeed and (args.train_iters or args.epochs):
    #     if args.DDP_impl == 'torch':
    #         i = flow.cuda.current_device()
    #         model = TorchDDP(model, device_ids=[i], output_device=i,
    #                          process_group=mpu.get_data_parallel_group())
    #     elif args.DDP_impl == 'local':
    #         model = LocalDDP(model)
    #     else:
    #         print_rank_0("Skip DDP model")
    return model


# def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    # while isinstance(model, (LocalDDP, TorchDDP, FP16_Module)):
    #     model = model.module
    # param_groups = glm_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    # True
    # for param_group in param_groups:
    #     # print('## param_group', len(param_group['params']))
    #     for param in param_group['params']:
    #         if not hasattr(param, 'model_parallel'):
    #             param.model_parallel = False

    return param_groups


def get_optimizer(param_groups, args):
    """Set up the optimizer."""
    if args.cpu_optimizer:
        # Apex FusedAdam uses decoupled weight decay so use the same here
        if args.cpu_torch_adam:
            cpu_adam_optimizer = flow.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        if args.optimizer == 'adam':
            optimizer = Adam(param_groups,
                             lr=args.lr,
                             weight_decay=args.weight_decay,
                             betas=(args.adam_beta1, args.adam_beta2),
                             eps=args.adam_eps)
        elif args.optimizer == 'adafactor':
            from transformers import Adafactor
            optimizer = Adafactor(param_groups, lr=args.lr, relative_step=False, warmup_init=False)
        else:
            raise NotImplementedError

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if hasattr(args, "deepspeed") and args.deepspeed:
        raise NotImplementedError
        # fp16 wrapper is not required for DeepSpeed.
        # return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    if args.finetune:
        num_iters = num_iters // args.gradient_accumulation_steps
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters - warmup_iter,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio)

    return lr_scheduler


def load_torch_model(model, path):
    import torch
    torch_params = torch.load(path, map_location='cpu')

    flow_params = {}
    for k,v in torch_params.items():
        flow_params[k] = flow.Tensor(
            v.numpy().astype("float32"))
    model.load_state_dict(flow_params, strict=False)
    print("load pretraining model succeed!")

def setup_model_and_optimizer(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Setup model and optimizer."""

    model = get_model(args, model_type=model_type, multi_token=multi_token, num_labels=num_labels,
                      spell_length=spell_length)
    
    
    if args.debug_loss:
        # load pretrain
        load_torch_model(model, path=args.debug_pretrain_model)
        # load_torch_model(model,"/home/fengwen/datasets/mo.pt")    
        model.eval()
    else:
        model.train()
    print(f"/home/fengwen/datasets/mo.pt  is load!!")
    
    check_mode(args,model)

    # optimizer = flow.optim.SGD(
    #         model.parameters(),
    #         lr=args.lr,
    #         momentum=0.9,
    #         weight_decay=0.0,
    #     )
    optimizer = flow.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                betas=(args.adam_beta1, args.adam_beta2),
                                eps=args.adam_eps)
    lr_scheduler = flow.optim.lr_scheduler.StepLR(optimizer, step_size=100000) 

    if args.mode == "eager":
        return model, optimizer, lr_scheduler
    if args.mode == "graph":
        graph_model = mpu.GLMGraph(args, model, optimizer, lr_scheduler)
        return graph_model, None, None
    else:
        raise NotImplementedError


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    # if args.deepspeed:
    #     model.backward(loss)
    # else:
    #     # optimizer.zero_grad()
    #     if args.fp16:
    #         optimizer.backward(loss, update_master_grads=False)
    #     else:
    #         loss.backward()
    if args.mode == 'eager':
        loss.backward()
    if args.deepspeed or args.DDP_impl == 'torch':
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()
    else:
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False, fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0 and args.mode == 'eager':
            if not args.fp16 :
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss


def see_memory_usage(message, force=False):
    if not force:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print("Memory Allocated ", flow.cuda.memory_allocated() / (1024 * 1024 * 1024), "GigaBytes")
        print("Max Memory Allocated ", flow.cuda.max_memory_allocated() / (1024 * 1024 * 1024), "GigaBytes")
        print("Cache Allocated ", flow.cuda.memory_cached() / (1024 * 1024 * 1024), "GigaBytes")
        print("Max cache Allocated ", flow.cuda.max_memory_cached() / (1024 * 1024 * 1024), "GigaBytes")
        print(" ")
        # input("Press Any Key To Continue ..")

def train_step_graph(data_iterator, model, optimizer, lr_scheduler):
    placement = flow.env.all_device_placement("cuda")
    sbp = flow.sbp.split(0)
    tokens = tokens.to_global(placement=placement, sbp=sbp)
    position_ids = position_ids.to_global(placement=placement, sbp=sbp)
    attention_mask = attention_mask.to_global(
        placement=placement, sbp=sbp)
    labels = labels.to_global(placement=placement, sbp=sbp)
    loss_mask = loss_mask.to_global(placement=placement, sbp=sbp)
    loss = model(tokens, position_ids, attention_mask, labels, loss_mask)




def train_step(data_iterator, model, optimizer, lr_scheduler, args, timers, forward_step_func, mems=None,
               single_step=False):
    """Single training step."""
    mems = [] if mems is None else mems
    if args.mode == 'eager':
        optimizer.zero_grad()
        loss, mems, _ = forward_step_func(data_iterator, model, args, timers, mems)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        return loss,0,mems
    elif args.mode=="graph":
        loss, mems, _ = forward_step_func(data_iterator, model, args, timers, mems)
        return loss,0,mems
    else:
        raise ImportError

    lm_loss_total, count = 0.0, 0
    mems = [] if mems is None else mems
    if not args.deepspeed and args.mode == 'eager':
        optimizer.zero_grad()
    while True:
        skipped_iter, complete = 0, False
        # Forward model for one step.
        # timers('forward').start()
        lm_loss, mems, _ = forward_step_func(data_iterator, model, args, timers, mems)

        if args.mode=='graph':
            return lm_loss,skipped_iter,mems
            
        # timers('forward').stop()
        # print_rank_0("Forward step")
        if not args.deepspeed:
            lm_loss /= args.gradient_accumulation_steps

        reduced_loss = lm_loss.detach().clone().view(1)
        # flow.distributed.all_reduce(reduced_loss.data, group=mpu.get_data_parallel_group())
        reduced_loss.data = reduced_loss.data / (args.world_size / args.model_parallel_size)

        if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
            lm_loss_total += reduced_loss
            count += 1

            # Calculate gradients, reduce across processes, and clip.
            # timers('backward').start()
            backward_step(optimizer, model, lm_loss, args, timers)
            # timers('backward').stop()
            # print_rank_0("Backward step")
            # Update parameters.
            # timers('optimizer').start()
            if args.deepspeed:
                if model.is_gradient_accumulation_boundary():
                    model.step()
                    complete = True
                    if not (args.fp16 and optimizer.overflow):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
                else:
                    model.step()
            else:
                if count == args.gradient_accumulation_steps :
                    optimizer.step()
                    complete = True
                    # Update learning rate.
                    if not (args.fp16 and optimizer.overflow):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
            # print_rank_0("Optimizer step")
            # timers('optimizer').stop()
            if complete:
                break
        else:
            print_rank_0("Found NaN loss, skip backward")
            del lm_loss, reduced_loss
            mems = []
        if single_step:
            break
    # if args.deepspeed:
    #     lm_loss_total = lm_loss_total / count
    return lm_loss_total, skipped_iter, mems
