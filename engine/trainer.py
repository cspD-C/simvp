import os

import numpy as np
import torch
from tqdm import tqdm

from core.checkpoint import BestCheckpointSaver, save_training_state
from core.metrics import metric
from core.utils import print_log


def train_one_epoch(
    model, data_loader, optimizer, scheduler, criterion, device, max_steps=0
):
    model.train()
    losses = []
    pbar = tqdm(data_loader)
    for step, (batch_x, batch_y) in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        pbar.set_description(f"train loss: {loss_value:.4f}")
        if max_steps > 0 and (step + 1) >= max_steps:
            break
    return float(np.average(losses))


def validate_one_epoch(model, data_loader, criterion, device, max_samples=1000):
    model.eval()
    preds_lst, trues_lst = [], []
    losses = []

    pbar = tqdm(data_loader)
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(pbar):
            if i * batch_x.shape[0] > max_samples:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)

            losses.append(float(loss.mean().detach().cpu()))
            pbar.set_description(f"vali loss: {losses[-1]:.4f}")
            preds_lst.append(pred_y.detach().cpu().numpy())
            trues_lst.append(batch_y.detach().cpu().numpy())

    mean_loss = float(np.average(losses))
    preds = np.concatenate(preds_lst, axis=0)
    trues = np.concatenate(trues_lst, axis=0)
    mse, mae, ssim, psnr = metric(
        preds,
        trues,
        data_loader.dataset.mean,
        data_loader.dataset.std,
        True,
    )
    stats = {"mse": mse, "mae": mae, "ssim": ssim, "psnr": psnr}
    model.train()
    return mean_loss, stats


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    args,
    checkpoints_dir,
):
    auto_stop_mode = args.epochs == 0
    if auto_stop_mode and val_loader is None:
        raise RuntimeError(
            "Early-stop mode requires validation data. "
            "Set --epochs > 0 or provide a validation split."
        )
    target_epochs = args.epochs if args.epochs > 0 else args.max_epochs

    best_saver = BestCheckpointSaver(
        mode="min", delta=args.early_stop_min_delta, verbose=True
    )
    best_ckpt_path = os.path.join(checkpoints_dir, "checkpoint.pth")
    no_improve_epochs = 0

    for epoch in range(target_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            max_steps=args.max_steps_per_epoch,
        )

        if epoch % args.log_step != 0:
            continue

        if val_loader is None:
            val_loader = train_loader
        val_loss, val_stats = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_samples=args.max_val_samples,
        )
        improved = best_saver.update(val_loss, model, best_ckpt_path)
        if improved:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        print_log(
            (
                f"Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} "
                f"Vali Loss: {val_loss:.4f} | "
                f"mse:{val_stats['mse']:.4f}, mae:{val_stats['mae']:.4f}, "
                f"ssim:{val_stats['ssim']:.4f}, psnr:{val_stats['psnr']:.4f}"
            )
        )

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_training_state(
                os.path.join(checkpoints_dir, f"epoch_{epoch + 1}.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                args=args,
                metrics={"train_loss": train_loss, "val_loss": val_loss},
            )

        if auto_stop_mode and (epoch + 1) >= args.min_epochs and no_improve_epochs >= args.early_stop_patience:
            print_log(
                (
                    f"Early stopping at epoch={epoch + 1} "
                    f"(patience={args.early_stop_patience}, "
                    f"min_delta={args.early_stop_min_delta})"
                )
            )
            break

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    return model


def test_model(model, test_loader, device, save_dir=None):
    model.eval()
    inputs_lst, trues_lst, preds_lst = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred_y = model(batch_x.to(device))
            inputs_lst.append(batch_x.detach().cpu().numpy())
            trues_lst.append(batch_y.detach().cpu().numpy())
            preds_lst.append(pred_y.detach().cpu().numpy())

    inputs = np.concatenate(inputs_lst, axis=0)
    trues = np.concatenate(trues_lst, axis=0)
    preds = np.concatenate(preds_lst, axis=0)

    mse, mae, ssim, psnr = metric(
        preds,
        trues,
        test_loader.dataset.mean,
        test_loader.dataset.std,
        True,
    )
    metrics = {"mse": mse, "mae": mae, "ssim": ssim, "psnr": psnr}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "inputs.npy"), inputs)
        np.save(os.path.join(save_dir, "trues.npy"), trues)
        np.save(os.path.join(save_dir, "preds.npy"), preds)

    return metrics
