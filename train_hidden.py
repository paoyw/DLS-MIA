from argparse import ArgumentParser, Namespace
import os

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm

import utils
from model import hidden


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--device', default='cpu')

    parser.add_argument('--root', default='data/COCO')
    parser.add_argument('--train_csv', default='data/COCO/train_0.csv')
    parser.add_argument('--val_csv', default='data/COCO/val.csv')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--val-batch-size', default=16, type=int)
    parser.add_argument('--num-workers', default=8, type=int)

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--message-length', default=32, type=int)
    parser.add_argument('--lambda_img', default=1, type=float)
    parser.add_argument('--lambda_msg', default=0.7, type=float)
    parser.add_argument('--lambda_adv', default=1e-3, type=float)

    parser.add_argument('--save-dir', default='log/')
    parser.add_argument('--val-step', default=10, type=int)

    return parser.parse_args()


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'train.csv'), 'w') as f:
        f.write('epoch,L_img,L_msg,L_g_adv,L_d_adv,psnr,acc,adv_acc\n')

    with open(os.path.join(args.save_dir, 'val.csv'), 'w') as f:
        f.write('epoch,psnr,acc\n')

    dataloaders = utils.get_dataloaders(
        root=args.root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    hidden_model = hidden.Hidden(
        hidden.HiddenEncoder(message_length=args.message_length),
        hidden.HiddenDecoder(message_length=args.message_length),
        hidden.HiddenNoisysLayer(size=256),
        hidden.HiddenAdversary(),
    )

    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    optimizer = Adam([{'params': hidden_model.encoder.parameters()},
                      {'params': hidden_model.decoder.parameters()}],
                     lr=args.lr)
    adv_optimizer = Adam(hidden_model.adversary.parameters(), lr=args.lr)

    hidden_model = hidden_model.to(args.device)
    step = 0
    for epoch in range(1, args.epochs + 1):
        hidden_model.train()
        pbar = tqdm(dataloaders['train'], bar_format='{l_bar}{r_bar}')
        for images in pbar:
            step += 1
            images = (images.to(args.device) - 0.5) / 0.5
            messages = torch.randint(0, 2,
                                     (images.shape[0], args.message_length),
                                     dtype=torch.float, device=args.device)

            optimizer.zero_grad()
            encoded_images = hidden_model.encoder(images, messages)

            noisy_images = hidden_model.noisy_layer(encoded_images,
                                                    size=256)

            decoded_messages = hidden_model.decoder(noisy_images)

            image_dist_loss = mse_fn(encoded_images, images)
            message_loss = mse_fn(decoded_messages, messages)

            preds = hidden_model.adversary(encoded_images)
            adv_loss = ce_fn(
                preds,
                torch.zeros(encoded_images.shape[0],
                            dtype=torch.long,
                            device=args.device)
            )

            loss = args.lambda_img * image_dist_loss \
                + args.lambda_msg * message_loss \
                + args.lambda_adv * adv_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            adv_optimizer.zero_grad()
            adv_imgs = torch.concat(
                (images, encoded_images.detach()), dim=0)
            labels = torch.concat(
                (torch.zeros(images.shape[0],
                             dtype=torch.long,
                             device=args.device),
                    torch.ones(encoded_images.shape[0],
                               dtype=torch.long,
                               device=args.device)),
                dim=0,
            )

            d_preds = hidden_model.adversary(adv_imgs)
            d_adv_loss = ce_fn(d_preds, labels)
            d_adv_loss.backward()
            adv_optimizer.step()

            if step % 500 == 0:
                save_image(images * 0.5 + 0.5, 'org.png')
                save_image(noisy_images * 0.5 + 0.5, 'noise.png')
                save_image(encoded_images * 0.5 + 0.5, 'enc.png')

            with torch.no_grad():
                psnr = utils.cal_psnr(images, encoded_images, mean=0, std=0.5)
                acc = utils.cal_acc(decoded_messages, messages)
                adv_acc = (d_preds.argmax(dim=-1) ==
                           labels).to(torch.float).mean()
            pbar.set_description(
                f'|epoch {epoch}|'
                f'|L_img: {image_dist_loss.cpu().item():.4f}|'
                f'|L_msg: {message_loss.cpu().item():.4f}|'
                f'|L_g_adv: {adv_loss.cpu().item():.4f}|'
                f'|L_d_adv: {d_adv_loss.cpu().item():.4f}|'
                f'|PSNR: {psnr.mean().cpu().item():.3f}|'
                f'|MSG ACC: {acc.mean().cpu().item():.3f}|'
                f'|ADV ACC: {adv_acc.cpu().item():.3f}|')
        with open(os.path.join(args.save_dir, 'train.csv'), 'a') as f:
            f.write(f'{epoch},'
                    f'{image_dist_loss.cpu().item()},'
                    f'{message_loss.cpu().item()},'
                    f'{adv_loss.cpu().item()},'
                    f'{d_adv_loss.cpu().item()},'
                    f'{psnr.mean().cpu().item()},'
                    f'{acc.mean().cpu().item()},'
                    f'{adv_acc.cpu().item()}\n')

        if epoch % args.val_step == 0:
            hidden_model.eval()
            with torch.no_grad():
                psnrs = []
                accs = []
                pbar = tqdm(dataloaders['val'], ncols=70)
                for images in pbar:
                    images = 2 * images.to(args.device) - 1
                    messages = torch.randint(0, 2,
                                             (images.shape[0],
                                              args.message_length),
                                             dtype=torch.float,
                                             device=args.device)

                    encoded_images = hidden_model.encoder(images, messages)

                    noisy_images = hidden_model.noisy_layer(encoded_images)

                    decoded_messages = hidden_model.decoder(noisy_images)
                    psnrs.extend(
                        utils.cal_psnr(images, encoded_images, mean=0.5, std=0.5).cpu().tolist())
                    accs.extend(
                        utils.cal_acc(messages, decoded_messages).cpu().tolist())
                    pbar.set_description(f'{np.mean(psnrs):.4f}|'
                                         f'{np.mean(accs):.4f}')
                print(f'VAL PSNR {np.mean(psnrs)}|VAL ACC {np.mean(accs)}')
                with open(os.path.join(args.save_dir, 'val.csv'), 'a') as f:
                    f.write(f'{np.mean(psnrs)},{np.mean(accs)}\n')
                torch.save(hidden_model.state_dict(), os.path.join(
                    args.save_dir, f'epoch{epoch}.ckpt'))


if __name__ == '__main__':
    main(get_args())
