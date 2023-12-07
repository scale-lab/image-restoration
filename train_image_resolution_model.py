import os
from data import DIV2K
from model.edsr import edsr
from model.wdsr import wdsr_b
from train import EdsrTrainer, WdsrTrainer
import argparse
import tensorflow.compat.v1 as tf

def main(model_name, downgrade, scale, batch_size=16, depth=16):
    div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
    div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)

    train_ds = div2k_train.dataset(batch_size=batch_size, random_transform=True)
    valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

    if model_name == 'edsr':
        trainer = EdsrTrainer(model=edsr(scale=scale, num_res_blocks=depth), 
                            checkpoint_dir=f'.ckpt/edsr-{depth}-{downgrade}-x{scale}')
    elif model_name == 'wdsr':
        trainer = WdsrTrainer(model=wdsr_b(scale=scale, num_res_blocks=depth), 
                            checkpoint_dir=f'.ckpt/edsr-{depth}-{downgrade}-x{scale}')
    else:
        NotImplementedError(f"Model {model_name} not implemented")

    # Train the model for 300,000 steps and evaluate model
    # every 1000 steps on the first 10 images of the DIV2K
    # validation set. Save a checkpoint only if evaluation
    # PSNR has improved.
    print(f"Train {model_name} model for 300,000 steps and evaluate model")
    trainer.train(train_ds,
                valid_ds.take(10),
                steps=300000, 
                evaluate_every=1000, 
                save_best_only=True)

    # Evaluate model on full validation set
    print("Evaluate model on full validation set")
    psnrv = trainer.evaluate(valid_ds)
    print(f'PSNR = {psnrv.numpy():3f}')

    # Save weights to separate location.
    print("Save weights to separate location")
    trainer.model.save_weights(f'weights/{model_name}-{depth}-{downgrade}-x{scale}/weights.h5')   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Restoration Model.')
    parser.add_argument('--scale', type=int, default=4,
                        help='super-resolution factor')
    parser.add_argument('--downgrade', type=str, default='bicubic',
                        help='downgrade type')
    parser.add_argument('--depth', type=int, default=16,
                        help='Number of residual blocks')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch Size for training')
    parser.add_argument('--model', type=str, default='edsr',
                        help='Model name, can be edsr or wdsr')
    args = parser.parse_args()

    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("WARNING: No GPU found, running on CPU")

    main(model_name=args.model,
         downgrade=args.downgrade, 
         scale=args.scale,
         batch_size=args.batch_size,
         depth=args.depth)
