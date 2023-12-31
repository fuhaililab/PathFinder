"""
Main train classifier file
Jiarui Feng
"""
import random
from collections import OrderedDict
from json import dumps

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_utils
import train_utils
from args import get_classifier_args
from models.models import PathFinder, SimpleClassifier
from intra_network_generation import generate_intra_network
import os


def get_model(args, log):
    model = PathFinder(input_size=args.input_size,
                       hidden_size=args.hidden_size,
                       r=args.r,
                       N=args.num_layer,
                       head=args.head,
                       num_in_degree=args.num_in_degree,
                       num_out_degree=args.num_out_degree,
                       num_nodes=args.num_nodes,
                       num_paths=args.num_paths,
                       num_edges=args.num_edges,
                       max_path_len=args.max_length + 1,
                       gamma=args.gamma,
                       JK=args.JK,
                       drop_prob=args.drop_prob)

    predictor = SimpleClassifier(hidden_size=args.hidden_size,
                                 output_size=args.output_size)

    model = nn.DataParallel(model, args.gpu_ids)
    predictor = nn.DataParallel(predictor, args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, epoch = train_utils.load_model(model, args.load_path + f"_pathformer.pth.tar", args.gpu_ids)
        predictor, epoch = train_utils.load_model(predictor, args.load_path + f"_predictor.pth.tar", args.gpu_ids)

    else:
        epoch = 0

    return model, predictor, epoch

def train(epoch, model, predictor, path_list, path_index, path_edge_type, path_positions, prior_path_weight, optimizer, train_loader, device):
    with torch.enable_grad(), \
            tqdm(total=len(train_loader.dataset)) as progress_bar:
        for batch_x, batch_y, batch_in_deg, batch_out_deg, batch_edge_types, batch_node_index, batch_mask in train_loader:
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_in_deg = batch_in_deg.to(device)
            batch_out_deg = batch_out_deg.to(device)
            batch_edge_types = batch_edge_types.to(device)
            batch_node_index = batch_node_index.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            x, path_emb = model(batch_x, batch_mask, batch_in_deg, batch_out_deg, batch_edge_types,
                                batch_node_index, path_list, path_index, path_edge_type, path_positions)
            prediction = predictor(path_emb)
            prediction = torch.log_softmax(prediction, dim=-1)

            reg_loss = model.module.compute_reg_loss(prior_path_weight)
            weight_vector = torch.zeros([args.output_size], device=device)
            for i in range(args.output_size):
                n_samplei = torch.sum(batch_y == i).item()
                if n_samplei > 0:
                    weight_vector[i] = batch_size / (n_samplei * args.output_size)

            nll_loss = F.nll_loss(prediction, batch_y, weight=weight_vector)
            loss = nll_loss + args.reg_weight * reg_loss
            loss_val = nll_loss.item()

            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm)
            optimizer.step()

            progress_bar.update(batch_size)
            progress_bar.set_postfix(epoch=epoch,
                                     NLL=loss_val)

    return



def evaluate(model, predictor, path_list, path_index, path_edge_type, path_positions, prior_path_weight, data_loader, device, save_dir, epoch, test=False):
    nll_meter = train_utils.AverageMeter()
    metrics_meter = train_utils.MetricsMeter()
    model.eval()
    predictor.eval()
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for batch_x, batch_y, batch_in_deg, batch_out_deg, batch_edge_types, batch_node_index, batch_mask in data_loader:
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_in_deg = batch_in_deg.to(device)
            batch_out_deg = batch_out_deg.to(device)
            batch_edge_types = batch_edge_types.to(device)
            batch_node_index = batch_node_index.to(device)
            batch_mask = batch_mask.to(device)
            x, path_emb = model(batch_x, batch_mask, batch_in_deg, batch_out_deg, batch_edge_types,
                                batch_node_index, path_list, path_index, path_edge_type, path_positions)
            prediction = predictor(path_emb)
            prediction = torch.log_softmax(prediction, dim=-1)

            reg_loss = model.module.compute_reg_loss(prior_path_weight)
            nll_loss = F.nll_loss(prediction, batch_y)
            loss = nll_loss + args.reg_weight * reg_loss
            loss_val = nll_loss.item()
            nll_meter.update(loss_val, batch_size)
            metrics_meter.update(prediction.exp()[:, 1].cpu(), batch_y.cpu())
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)
        if not test:
            path_weight = model.module.return_path_weight().detach().cpu().squeeze().numpy()
            plt.figure()
            sns.heatmap(path_weight[np.newaxis, :])
            plt.savefig(f"{save_dir}/masker_heatmap_{epoch}.png")
            plt.close()

        model.train()
        predictor.train()
        metrics_result = metrics_meter.return_metrics()
        results_list = [
            ('Loss', nll_meter.avg),
            ('Accuracy', metrics_result["Accuracy"]),
            ('Recall', metrics_result["Recall"]),
            ('Precision', metrics_result["Precision"]),
            ('Specificity', metrics_result["Specificity"]),
            ('F1', metrics_result["F1"]),
            ("AUC", metrics_result["AUC"])
        ]
        results = OrderedDict(results_list)
        return results



def main(args):
    # Set up directory
    args.save_dir = train_utils.get_save_dir(args.save_dir, args.name, "train_classifier")

    # Get and process input data
    args, expression, edge_types, G, gene_list, label, \
    receptor_list, target_list, fold_change, prior_path_weight, \
    shortest_path_length, in_deg, out_deg, path_list, path_index, \
    path_edge_type, path_positions = data_utils.process_input_data(args)

    # Set up logging and devices
    log = train_utils.get_logger(args.save_dir, args.name)
    # tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = train_utils.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    learned_path_weight_list = []
    best_val_results = []
    best_test_results = []
    # multiple runs
    for run in range(args.runs):
        log.info(f"Start run {run + 1}")

        seed = train_utils.get_seed(args.seed)
        # Set random seed
        log.info(f'Using random seed {seed}...')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Get model
        log.info('Building model...')
        model, predictor, epoch = get_model(args, log)
        model = model.to(device)
        predictor = predictor.to(device)
        model.train()
        predictor.train()
        path_list = path_list.to(device)
        path_index = path_index.to(device)
        path_edge_type = path_edge_type.to(device)
        path_positions = path_positions.to(device)
        prior_path_weight = prior_path_weight.to(device)


        # Get saver
        saver = train_utils.CheckpointSaver(args.save_dir,
                                            max_checkpoints=args.max_checkpoints,
                                            metric_name=args.metric_name,
                                            maximize_metric=args.maximize_metric,
                                            log=log)

        # Get optimizer and scheduler
        model_parameter_list = []
        path_weight_parameter = []
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                if name == "module.path_weight":
                    path_weight_parameter.append(param)
                else:
                    model_parameter_list.append(param)

        if args.maximize_metric:
            mode = "max"
        else:
            mode = "min"
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
        scheduler = sched.ReduceLROnPlateau(optimizer,
                                            factor=0.5, patience=5, min_lr=1e-6, mode=mode)


        # Get data loader.
        index = [i for i in range(expression.shape[0])]
        train_index, val_index, test_index = train_utils.split_train_val_test(index,
                                                                              seed,
                                                                              args.val_ratio,
                                                                              args.test_ratio,
                                                                              label)

        train_dataset = train_utils.LoadClassifierDataset(expression[train_index], args.gs_path, gene_list, in_deg, out_deg,
                                                    shortest_path_length, edge_types, label[train_index])
        val_dataset = train_utils.LoadClassifierDataset(expression[val_index], args.gs_path, gene_list, in_deg, out_deg,
                                                  shortest_path_length, edge_types, label[val_index])
        test_dataset = train_utils.LoadClassifierDataset(expression[test_index], args.gs_path, gene_list, in_deg, out_deg,
                                                  shortest_path_length, edge_types, label[test_index])

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  collate_fn=train_utils.classifier_collate_fn)

        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                collate_fn=train_utils.classifier_collate_fn)

        test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                collate_fn=train_utils.classifier_collate_fn)


        # Train
        log.info('train the model on classification...')
        while epoch != args.num_epochs:
            epoch += 1
            log.info(f'Starting epoch {epoch}...')
            train(epoch, model, predictor, path_list, path_index, path_edge_type, path_positions, prior_path_weight,
                  optimizer, train_loader, device)

            # Evaluate and save checkpoint
            log.info(f'Evaluating after epoch {epoch}...')
            # ema.assign(model)
            val_results = evaluate(model, predictor, path_list, path_index, path_edge_type, path_positions, prior_path_weight,
                               val_loader, device, args.save_dir, epoch)
            model_dict = dict(zip([f"_pathformer_{run + 1}", f"_predictor_{run + 1}"], [model, predictor]))
            scheduler.step(val_results[args.metric_name])

            # Log to console
            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in val_results.items())
            log.info(f'Val {results_str}')

            if saver.is_best(val_results[args.metric_name]):
                test_results = evaluate(model, predictor, path_list, path_index, path_edge_type, path_positions, prior_path_weight,
                                   test_loader, device, args.save_dir, epoch, True)

                # Log to console
                results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in test_results.items())
                log.info(f'Test {results_str}')

            saver.save(epoch, model_dict, val_results[args.metric_name], device)

        best_val_results.append(saver.best_val)
        model, _ = train_utils.load_model(model, f"{args.save_dir}/best" + f"_pathformer_{run + 1}.pth.tar", args.gpu_ids)
        model.eval()
        test_results = evaluate(model, predictor, path_list, path_index, path_edge_type, path_positions, prior_path_weight,
                                test_loader, device, args.save_dir, epoch, True)
        best_test_results.append(test_results[args.metric_name])
        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in test_results.items())
        log.info(f'Best test {results_str}')
        path_weight = model.module.return_path_weight().detach().cpu().squeeze().numpy()
        learned_path_weight_list.append(path_weight[np.newaxis, :])


    log.info(f"Finish training, compute average results.")
    mean_val = np.mean(best_val_results)
    std_val = np.std(best_val_results)
    mean_test = np.mean(best_test_results)
    std_test = np.std(best_test_results)

    val_desc = '{:.3f} ± {:.3f}'.format(mean_val, std_val)
    test_desc = '{:.3f} ± {:.3f}'.format(mean_test, std_test)

    log.info(f"Validation result: {val_desc}")
    log.info(f"Test result: {test_desc}")

    np.savez(f"{args.save_dir}/save_result.npz", gene_list=gene_list,
             path_list=path_list.cpu().numpy(),
             path_index=path_index.cpu().numpy(),
             path_edge_type=path_edge_type.cpu().numpy(),
             path_positions=path_positions.cpu().numpy(),
             shortest_path_length=shortest_path_length.cpu().numpy(),
             path_weight=np.concatenate(learned_path_weight_list, axis=0),
             receptor_list=receptor_list,
             prior_path_weight=prior_path_weight.cpu().numpy()
             )
    return



if __name__ == "__main__":
    # Load args.
    args = get_classifier_args()
    main(args)
    generate_intra_network(args.save_dir, 200)
