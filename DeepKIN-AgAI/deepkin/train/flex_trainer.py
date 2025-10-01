from __future__ import print_function, division

import gc
import math
import os
import random
import time
from datetime import timedelta
from typing import Tuple

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from deepkin.data.morpho_data import send_tuple_to, print_batch_tuple
from deepkin.train.flex_train_tools import create_model, create_optimizer_and_lr_scheduler, next_train_dataset, \
    train_args_init_b4mp, next_validation_dataset, init_global_data_b4mp
from deepkin.utils.arguments import FlexArguments
from deepkin.utils.misc_functions import time_now, time_after_eta


def timestr(tm):
    target = time_after_eta(tm)
    # Calculate the number of full days in the given time duration.
    day = tm // (24 * 3600)
    # Update the time variable to hold the remaining seconds after subtracting full days.
    tm = tm % (24 * 3600)

    # Calculate the number of full hours in the remaining time.
    hour = tm // 3600
    # Update the time variable to hold the remaining seconds after subtracting full hours.
    tm %= 3600

    # Calculate the number of full minutes in the remaining time.
    minutes = tm // 60
    # Update the time variable to hold the remaining seconds after subtracting full minutes.
    tm %= 60

    # The 'time' variable now represents the remaining seconds, which is the number of seconds.
    seconds = tm

    # Print the time duration in the format "d:h:m:s".
    return (f"{day:03.0f}d+" if (day>0)else"") + f"{hour:02.0f}h{minutes:02.0f}m{seconds:02.0f}s [{target}]"

def elapsed_eta(msps, iters, total_iters, val_steps, offset):
    # next_val_steps = (((int(iters-offset)//int(val_steps))+1)*int(val_steps))+int(offset)
    next_val_steps = int(math.ceil(math.ceil((iters - offset)/val_steps)*val_steps))+int(offset)
    return timestr(iters*1000.0/msps), timestr((total_iters - iters)*1000.0/msps), timestr((next_val_steps - iters)*1000.0/msps)

def save_model_state(filename:str, args: FlexArguments, model, optimizer, scaler, lr_scheduler, steps: int, best_valid_loss: float, msps: float):
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.save({'args': args.as_dict(),
                'model_state_dict': (model.module.state_dict() if args.use_ddp else model.state_dict()),
                'optimizer_state_dict': (optimizer.state_dict() if (optimizer is not None) else 'N/A'),
                'lr_scheduler_state_dict': (lr_scheduler.state_dict() if (lr_scheduler is not None) else 'N/A'),
                'scaler_state_dict': (scaler.state_dict() if (scaler is not None) else 'N/A'),
                'steps': steps, 'best_valid_loss': best_valid_loss, 'msps':msps},
               filename)

def load_model_state(args: FlexArguments, model, optimizer, scaler, lr_scheduler, map_location):
    # Load saved state
    if dist.get_rank() == 0:
        print(time_now(), 'Loading model state...', flush=True)
    kb_state_dict = torch.load(args.model_save_path, map_location=map_location)
    if args.use_ddp:
        model.module.load_state_dict(kb_state_dict['model_state_dict'])
    else:
        model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    gc.collect()

    if dist.get_rank() == 0:
        print(time_now(), 'Loading optimizer state...', flush=True)
    kb_state_dict = torch.load(args.model_save_path, map_location=torch.device('cpu'))

    if optimizer is not None:
        optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])

    if scaler is not None:
        scaler.load_state_dict(kb_state_dict['scaler_state_dict'])

    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
        # Adjust LR Scheduler to desired number of iters
        lr_scheduler.end_iter = args.num_iters

    best_valid_loss = kb_state_dict['best_valid_loss']
    total_steps = kb_state_dict['steps']
    if lr_scheduler is not None:
        total_steps = lr_scheduler.num_iters * int(args.accumulation_steps // dist.get_world_size())

    del kb_state_dict
    gc.collect()
    if os.path.exists(f'{args.model_save_path}_best_valid_loss.pt'):
        kb_state_dict = torch.load(f'{args.model_save_path}_best_valid_loss.pt', map_location=torch.device('cpu'))
        best_valid_loss = kb_state_dict['best_valid_loss']
        print(time_now(), f'Loaded best valid loss as: {best_valid_loss:.6f} !', flush=True)
        del kb_state_dict
        gc.collect()
    return best_valid_loss, total_steps

def validation_loop(args: FlexArguments, model, model_cache, device, the_optimizer, the_scaler, the_lr_scheduler, steps: int, best_valid_loss: float, validation_data_loader, msps: float):
    model.eval()
    model.zero_grad(set_to_none=True)
    if the_optimizer is not None:
        the_optimizer.zero_grad()
    if dist.get_rank() == 0:
        save_model_state(args.model_save_path, args, model, the_optimizer, the_scaler, the_lr_scheduler, steps, best_valid_loss, msps)
        print(time_now(),f"Saved model checkpoint at: {args.model_save_path} before validation", flush=True)
    validation_losses = [0.0 for _ in range(args.num_losses)]
    with torch.no_grad():
        for batch_idx, batch_data_item in enumerate(validation_data_loader):
            with torch.cuda.amp.autocast(enabled=args.enable_amp, dtype=(torch.bfloat16 if args.use_bfloat16 else torch.float16)):
                losses = model(batch_idx, args, model_cache, send_tuple_to(batch_data_item, device, args=args))
            for i in range(args.num_losses):
                validation_losses[i] += float(losses[i])
    for i in range(args.num_losses):
        validation_losses[i] /= float(len(validation_data_loader))
    total_loss = sum(validation_losses)
    if dist.get_rank() == 0:
        if (total_loss < best_valid_loss) and math.isfinite(total_loss):
            best_valid_loss = total_loss
            save_model_state(f'{args.model_save_path}_best_valid_loss.pt', args, model, the_optimizer, the_scaler, the_lr_scheduler, steps, best_valid_loss, msps)
            print(time_now(), f"Saved new best model checkpoint at: {args.model_save_path}_best_valid_loss.pt", flush=True)
    current_iters = int(steps) // int(args.accumulation_steps // dist.get_world_size())
    elta = elapsed_eta(msps, current_iters, args.num_iters, args.validation_steps, args.validation_offset)
    if dist.get_rank() == 0:
        print(time_now(),
              'Validation Iter:', "{}/{}".format(current_iters, args.num_iters),
              f'  VALID LOSS: {sum(validation_losses):.6f} [', '  '.join([f'{ls:.6f}' for ls in validation_losses]), ']',
              f'  BEST VALID LOSS: {best_valid_loss:.6f}',
              ("  LR: {:.6f}/{}".format(the_lr_scheduler.get_lr(), the_lr_scheduler.start_lr) if (the_lr_scheduler is not None) else ''),
              '  MSPS:', "{:.3f}".format(msps),
              '  Past:', elta[0],
              '  ETA:', elta[1],
              '  Next Valid. ETA:', elta[2], flush=True)
    return best_valid_loss

def train_loop(args: FlexArguments, device: torch.device, model, model_cache, the_optimizer, the_scaler, the_lr_scheduler,
               train_data_loader, validation_data_loader,
               total_steps: int, best_valid_loss:float,
               epoch:int, epoch_count:int, start_time:float) -> Tuple[int,float,int,int,float]:
    world_size = dist.get_world_size()
    current_iters = int(total_steps) // int(args.accumulation_steps // world_size)
    if (current_iters > 0) and (((current_iters - args.validation_offset) % args.validation_steps) == 0):
        if validation_data_loader is not None:
            model.zero_grad(set_to_none=True)
            best_valid_loss = validation_loop(args, model, model_cache, device, the_optimizer, the_scaler,
                                              the_lr_scheduler, total_steps, best_valid_loss, validation_data_loader,
                                              100)

    dist.barrier()
    model.train()
    model.zero_grad(set_to_none=True)

    aggregated_loss = [0.0 for _ in range(args.num_losses)]
    aggregated_count = 0.0

    if not args.use_iterable_dataset:
        num_data_items = torch.tensor(len(train_data_loader), device=device)
        dist.all_reduce(num_data_items, op=dist.ReduceOp.MIN)
        total_data_items = (int(num_data_items.cpu().item())//int(args.accumulation_steps)) * int(args.accumulation_steps)
        print(dist.get_rank(), f'Training on {total_data_items} batches out of {len(train_data_loader)}')
    else:
        total_data_items = 0

    start_steps = total_steps
    count_items = 0
    msps = -1

    # Train
    for batch_idx, batch_data_item in enumerate(train_data_loader):
        model.train()
        try:
            if batch_idx == 0:
                start_steps = total_steps
                start_time = time.time()
            with torch.cuda.amp.autocast(enabled=args.enable_amp, dtype=(torch.bfloat16 if args.use_bfloat16 else torch.float16)):
                losses = model(batch_idx, args, model_cache, send_tuple_to(batch_data_item, device, args=args))

            losses = [(loss/args.accumulation_steps) for loss in losses]

            # Backward step
            if the_optimizer is not None:
                if args.use_mtl_optimizer:
                    the_optimizer.backward(losses)
                else:
                    total_itr_loss = torch.stack(losses).sum()
                    if the_scaler is not None:
                        the_scaler.scale(total_itr_loss).backward()
                    else:
                        total_itr_loss.backward()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print('OOM Error while processing batch:', flush=True)
                print_batch_tuple(batch_data_item)
                torch.cuda.empty_cache()
                gc.collect()
                model.zero_grad(set_to_none=True)
                if the_optimizer is not None:
                    the_optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
        for i in range(len(losses)):
            aggregated_loss[i] += (float(losses[i])* args.accumulation_steps)
        aggregated_count += 1.0
        total_steps += 1
        count_items += 1
        epoch_count += 1
        if int(total_steps % (args.accumulation_steps//world_size)) == 0:
            current_iters = int(total_steps) // int(args.accumulation_steps//world_size)
            if the_lr_scheduler is not None:
                the_lr_scheduler.step()
            if the_optimizer is not None:
                if the_scaler is not None:
                    the_scaler.unscale_(the_optimizer)
                    if args.enable_grad_norm_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping_max_norm)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    the_scaler.step(the_optimizer)
                    the_scaler.update()
                    the_optimizer.zero_grad()
                else:
                    if args.enable_grad_norm_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping_max_norm)
                    the_optimizer.step()
                    the_optimizer.zero_grad()
            if args.empty_cache_at_gradient_step:
                torch.cuda.empty_cache()
                gc.collect()
            current_time = time.time()
            msps = 1000.0 * ((total_steps - start_steps) / (args.accumulation_steps // world_size)) / (current_time - start_time)
            if (dist.get_rank() == 0) and (current_iters > 0) and ((current_iters % args.train_log_steps) == 0):
                if args.empty_cache_at_log_step:
                    torch.cuda.empty_cache()
                    gc.collect()
                # Train log step on device 0
                current_losses = [(ls / aggregated_count) for ls in aggregated_loss]
                elta = elapsed_eta(msps, current_iters, args.num_iters, args.validation_steps if args.use_iterable_dataset else ((len(train_data_loader)*args.world_size)//args.accumulation_steps), args.validation_offset)
                print(time_now(),
                      'Train Iter:', "{}/{}".format(current_iters, args.num_iters),
                      f'  EP:{epoch} OBJ: {sum(current_losses):.6f} [', '  '.join([f'{ls:.6f}' for ls in current_losses]), ']',
                      ("  LR: {:.6f}/{}".format(the_lr_scheduler.get_lr(), the_lr_scheduler.start_lr) if (the_lr_scheduler is not None) else ''),
                      '  MSPS:', "{:.3f}".format(msps),
                      '  Past:', elta[0],
                      '  ETA:', elta[1],
                      '  Next Valid. ETA:', elta[2], flush=True)
            if (current_iters > 0) and (((current_iters - args.checkpoint_offset) % args.checkpoint_steps) == 0):
                model.eval()
                model.zero_grad(set_to_none=True)
                if the_optimizer is not None:
                    the_optimizer.zero_grad()
                if dist.get_rank() == 0:
                    save_model_state(args.model_save_path, args, model, the_optimizer, the_scaler, the_lr_scheduler, total_steps, best_valid_loss, msps)
                    print(time_now(), f"Saved model checkpoint at: {args.model_save_path}", flush=True)

            if (current_iters > 0) and (((current_iters - args.validation_offset) % args.validation_steps) == 0) and args.use_iterable_dataset:
                # Validation loop on device 0; others wait.
                torch.cuda.empty_cache()
                gc.collect()
                epoch_count = 0
                epoch += 1
                if validation_data_loader is not None:
                    model.zero_grad(set_to_none=True)
                    best_valid_loss = validation_loop(args, model, model_cache, device, the_optimizer, the_scaler, the_lr_scheduler,
                                                      total_steps, best_valid_loss,
                                                      validation_data_loader, msps)
                start_steps = total_steps
                start_time = time.time()

                dist.barrier()
                model.train()

            # End of training
            if current_iters >= args.num_iters:
                if dist.get_rank() == 0:
                    print(time_now(), f"END_OF_TRAINING", flush=True)
                break
        if (not args.use_iterable_dataset) and (batch_idx == (total_data_items - 1)):
            break

    if not args.use_iterable_dataset:
        # Validation loop on device 0; others wait.
        torch.cuda.empty_cache()
        gc.collect()
        epoch_count = 0
        epoch += 1
        if validation_data_loader is not None:
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                best_valid_loss = validation_loop(args, model, model_cache, device, the_optimizer, the_scaler, the_lr_scheduler,
                                                  total_steps, best_valid_loss,
                                                  validation_data_loader, msps)
        # start_steps = total_steps
        start_time = time.time()

        dist.barrier()
        model.train()

    return total_steps, best_valid_loss, epoch, epoch_count, start_time

def train_fn(rank: int, args: FlexArguments, global_data):
    print(time_now(), 'Called train_fn()', flush=True)
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(device_id=device, backend=f'{args.dist_backend}', init_method='env://', world_size=args.world_size, rank=rank, timeout=timedelta(hours=10, minutes=30))
    torch.backends.cudnn.benchmark = True
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)
    print('Using device: ', device, "from", dist.get_world_size(), 'processes', flush=True)

    dist.barrier()

    best_valid_loss: float = 999999.0
    model, model_cache = create_model(rank, device, args)
    if args.use_ddp and (not args.use_mtl_optimizer):# or ((dist.get_world_size() > 1) and (not args.use_mtl_optimizer)):
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    elif (dist.get_world_size() > 1) and (not args.use_mtl_optimizer):
        print('========================> IMPORTANT WARNING: Training multi-gpu model without DDP or MTL Optimizer: You need to have an option for multi-gpu training!')
    model.float()

    if dist.get_rank() == 0:
        print(f'---------------------------------- Model Size @ {time_now()} ----------------------------------------')
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(time_now(), 'Total params:', total_params, f'({(total_params / 1_000_000.0):.0f}M)',
              'Trainable params:', trainable_params, f'({(trainable_params / 1_000_000.0):.0f}M)',  flush=True)
        print('Saving model in:', args.model_save_path)
        print('---------------------------------------------------------------------------------------')

    optimizer, scaler, lr_scheduler = create_optimizer_and_lr_scheduler(rank, args, model, device)

    args.load_saved_model = args.load_saved_model and os.path.exists(args.model_save_path)

    dist.barrier()

    if (not args.load_saved_model) and (dist.get_world_size() > 1):
        if dist.get_rank() == 0:
            save_model_state(args.model_save_path, args, model, optimizer, scaler, lr_scheduler,
                             0, best_valid_loss, 0)
        dist.barrier()
        args.load_saved_model = True

    total_steps = 0
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if args.load_saved_model:
        best_valid_loss, total_steps = load_model_state(args, model, optimizer, scaler, lr_scheduler, map_location)

    current_iters = int(total_steps) // int(args.accumulation_steps)
    if lr_scheduler is not None:
        current_iters = lr_scheduler.num_iters

    offset = max(0, current_iters)
    args.checkpoint_offset = offset
    args.validation_offset = offset

    if dist.get_rank() == 0:
        print(f'------------------ Train Config @ {time_now()} --------------------')
        print('number_of_load_batches: ', args.number_of_load_batches)
        print('accumulation_steps: ', args.accumulation_steps)
        print('batch_size (sequences): ', args.batch_size)
        print('effective_batch_size (sequences): ', args.batch_size * args.accumulation_steps)
        print('batch_size (tokens): ', args.max_batch_tokens)
        print('effective_batch_size (tokens): ', args.max_batch_tokens * args.accumulation_steps)
        print('peak_lr: {:.8f}'.format(args.peak_lr))
        print('iters: ', current_iters)
        print('warmup_iter: ', args.warmup_iter)
        print('end_iters: ', args.num_iters)
        print(f'best_valid_loss: {best_valid_loss:.6f}')
        print('total_steps: ', total_steps)
        print('-----------------------------------------------------')

    if dist.get_rank() == 0:
        print(time_now(), f'Start training  at {current_iters}/{args.num_iters} iterations)', flush=True)

    validation_data_loader = None
    if dist.get_rank() == 0:
        validation_data_cache, validation_dataset, validation_data_loader = next_validation_dataset(rank, global_data, device, args,None, None, None)
    train_data_cache, train_dataset, train_data_loader = None, None, None
    epoch, epoch_count = 0, 0
    start_time = 0.0
    while current_iters < args.num_iters:
        if dist.get_rank() == 0:
            print(time_now(), 'Loading dataset...', flush=True)
        train_data_cache, train_dataset, train_data_loader = next_train_dataset(rank, global_data, device, args, train_data_cache, train_dataset, train_data_loader)
        if dist.get_rank() == 0:
            print(time_now(), 'Memory status: ', psutil.virtual_memory(), flush=True)

        dist.barrier()
        (total_steps, best_valid_loss,
         epoch, epoch_count, start_time) = train_loop(args, device, model, model_cache, optimizer, scaler, lr_scheduler,
                                                      train_data_loader, validation_data_loader, total_steps,
                                                      best_valid_loss, epoch, epoch_count, start_time)
        current_iters = int(total_steps) // int(args.accumulation_steps // dist.get_world_size())

def trainer_main():
    args: FlexArguments = FlexArguments().parse_args()
    args: FlexArguments = train_args_init_b4mp(args)
    print(time_now(),'Training Args:', sorted([f'{k}: {v}' for k,v in args.as_dict().items()]))
    args.world_size = args.gpus
    if args.gpus == 0:
        args.world_size = 1
    global_data = init_global_data_b4mp(args)
    mp.spawn(train_fn, nprocs=args.world_size, args=(args,global_data,))

if __name__ == '__main__':
    import logging, sys
    import logging.config
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{random.randint(6060,9595)}'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    print(time_now(), 'LAUNCH_TRAINER')
    logging.basicConfig(level='WARNING', stream=sys.stdout)
    logging.config.dictConfig({
        'version': 1,
        # Other configs ...
        'level': 'WARNING',
        'disable_existing_loggers': True
    })
    trainer_main()
