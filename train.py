import os

import torch
from torch import nn, optim
import pandas as pd
from data import get_dataloaders
from metric import calc_nd
from models import DomainAdaptation, Gene, ShareAttn


def train(
        feat_dim,
        pred_len,
        hidden_dim,
        kernel_size,
        syn_type,
        syn_param,
        tradeoff,
        batch_size,
        num_epoch,
        lr,
        seed,
):
    print(f"Training with {syn_type} data")
    print(
        f"  feat_dim: {feat_dim}, pred_len: {pred_len}, hidden_dim: {hidden_dim}, kernel_size: {kernel_size}"
    )
    print(f"  batch_size: {batch_size}, num_epoch: {num_epoch}, lr: {lr}")
    print(f"  tradeoff: {tradeoff}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    os.makedirs(ckpt_dir := f"checkpoints/{syn_type}", exist_ok=True)

    src_trainloader, tgt_trainloader, tgt_validloader = get_dataloaders(
        syn_type, syn_param, feat_dim, pred_len, batch_size
    )

    # models
    shr_attn = ShareAttn(feat_dim, hidden_dim, kernel_size)

    src_gene, tgt_gene = (
        Gene(feat_dim, pred_len, shr_attn, hidden_dim, kernel_size),
        Gene(feat_dim, pred_len, shr_attn, hidden_dim, kernel_size),
    )
    dom_adapt = DomainAdaptation(feat_dim, hidden_dim)

    # optimizers
    # 这部分创建了三个优化器：att_optim用于共享注意力模型的参数，gen_optim用于生成器模型的参数，dis_optim用于域鉴别器模型的参数。损失函数使用了均方误差损失（nn.MSELoss）和二元交叉熵损失（nn.BCELoss）。
    mse, bce = nn.MSELoss(), nn.BCELoss()
    l1_loss = torch.nn.L1Loss()
    att_optim = optim.Adam(shr_attn.parameters(), lr=lr)
    gen_optim = optim.Adam(
        list(src_gene.enc.parameters())
        + list(src_gene.dec.parameters())
        + list(tgt_gene.enc.parameters())
        + list(tgt_gene.dec.parameters()),
        lr=lr,
    )
    dis_optim = optim.Adam(dom_adapt.parameters(), lr=lr)

    # training
    for model in [shr_attn, src_gene, tgt_gene, dom_adapt]:
        model.train()
        model.to(device)
    best_metric, best_epoch = torch.inf, None
    for epoch in range(num_epoch):
        seq_losses, dom_losses, refactor_losses, tot_losses = [], [], [], []
        # src_True = torch.zeros((feat_dim, pred_len))
        # tgt_True = torch.zeros((feat_dim, pred_len))
        for (src_data, src_true), (tgt_data, tgt_true) in zip(
                src_trainloader, tgt_trainloader
        ):
            src_data, src_true, tgt_data, tgt_true = (
                src_data.to(device),
                src_true.to(device),
                tgt_data.to(device),
                tgt_true.to(device),
            )

            gen_optim.zero_grad()
            dis_optim.zero_grad()
            att_optim.zero_grad()

            # reconstruction & prediction
            # 这部分代码通过生成器模型分别对源数据和目标数据进行序列重构和预测，预测结果src_pred、tgt_pred。得到V
            src_pred, (src_query, src_key, src_value) = src_gene(src_data)
            tgt_pred, (tgt_query, tgt_key, tgt_value) = tgt_gene(tgt_data)

            dom_q, dom_k = dom_adapt(src_query, tgt_key, tgt_value)

            refactor_loss = (
                    mse(src_data, src_pred[..., :-pred_len]).mean() + mse(tgt_data, tgt_pred[..., :-pred_len]).mean()
            )

            seq_loss = (
                    mse(src_true, src_pred[..., -pred_len:]).mean()
                    + mse(tgt_true, tgt_pred[..., -pred_len:]).mean()
            )

            dom_loss = (
                    (l1_loss(src_query, dom_q)
                     + l1_loss(src_key, dom_k)).mean()
                    + (l1_loss(tgt_query, dom_q)
                       + l1_loss(tgt_key, dom_k)).mean()
            )

            tot_loss = seq_loss + refactor_loss + tradeoff * dom_loss
            seq_losses.append(seq_loss.item())
            refactor_losses.append(refactor_loss.item())
            dom_losses.append(dom_loss.item())
            tot_losses.append(tot_loss.item())

            # backpropagation 进行反向传播和优化器的更新
            tot_loss.backward()
            gen_optim.step()
            dis_optim.step()
            att_optim.step()

        metrics = []
        for tgt_data, tgt_true in tgt_validloader:
            tgt_data, tgt_true = tgt_data.to(device), tgt_true.to(device)
            tgt_pred, (tgt_query, tgt_key, tgt_value) = tgt_gene(tgt_data)
            norm_dev = calc_nd(tgt_true, tgt_pred[..., -pred_len:])
            metrics.append(norm_dev.item())

        # 在每个epoch结束后，计算验证集上的指标，并保存具有最佳指标的模型参数
        if (sum(metrics) / len(metrics)) < best_metric:
            best_metric, best_epoch = sum(metrics) / len(metrics), epoch + 1
            torch.save(
                {
                    "shr_attn": shr_attn.state_dict(),
                    "src_gene": src_gene.state_dict(),
                    "tgt_gene": tgt_gene.state_dict(),
                    "dom_adaptation": dom_adapt.state_dict(),
                },
                f"{ckpt_dir}/epoch{best_epoch}.pt",
            )

        print(f"Epoch {epoch + 1:4d} /{num_epoch} {'=' * 30}")
        print(f"Metric: {sum(metrics) / len(metrics):.8f}")
        print(
            "Loss:"
            f" total {sum(tot_losses) / len(tot_losses):.4f}"
            f" seq {sum(seq_losses) / len(seq_losses):.4f}"
            f" dom {sum(dom_losses) / len(dom_losses):.4f}"
        )

    print(f"Best metric: {best_metric:.8f} at epoch {best_epoch}")
    os.system(f"cp {ckpt_dir}/epoch{best_epoch}.pt {ckpt_dir}/best.pt")


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--feat_dim", type=int, default=None, help="dimension of features")
    parser.add_argument("--pred_len", type=int, default=3, help="prediction length")
    parser.add_argument(
        "--hidden_dim",
        nargs="+",
        type=int,
        default=(64, 64),
        help="dimension of hidden layers in all MLP layers",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=(3, 5),
        help="kernel size of convolutional layers",
    )
    parser.add_argument(
        "--syn_type",
        type=str,
        default="source1",
        help="type of synthetic data (coldstart or fewshot)",
    )
    parser.add_argument(
        "--syn_param",
        type=int,
        default=1,
        help="parameter of synthesis",
    )
    parser.add_argument(
        "--tradeoff",
        type=float,
        default=1.0,
        help="tradeoff parameter of loss calculation",
    )
    # parser.add_argument("--batch_size", type=int, default=int(1e3))
    parser.add_argument("--batch_size", type=int, default=int(6))
    # parser.add_argument("--num_epoch", type=int, default=int(1e3))
    parser.add_argument("--num_epoch", type=int, default=int(10))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=718)
    args = parser.parse_args()

    # 读取CSV数据
    data = pd.read_csv("data/loadshuju/source1.csv")

    # 获取特征数量
    feat_dim = data.shape[1] - 1  # 减去1是因为第一列为日期/时间

    # 如果没有显式提供feat_dim参数，则使用从数据中获取的特征数量
    if args.feat_dim is None:
        args.feat_dim = feat_dim

    print("feat_dim:", args.feat_dim)

    if len(args.hidden_dim) == 1:
        args.hidden_dim = args.hidden_dim[0]
    else:
        args.hidden_dim = tuple(args.hidden_dim)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    train(**vars(args))


if __name__ == "__main__":
    main()
