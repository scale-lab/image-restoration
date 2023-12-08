from data import DIV2K
from model.edsr import edsr
from model.wdsr import wdsr_b
import argparse
import tensorflow.compat.v1 as tf
import datetime

# https://keras.io/examples/vision/edsr/
def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

def main(model_name, downgrade, scale, batch_size=16, epochs=100, depth=16):
    div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
    div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)

    train_ds = div2k_train.dataset(batch_size=batch_size, random_transform=True)
    valid_ds = div2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)

    if model_name == 'edsr':
        model = edsr(scale=scale, num_res_blocks=depth)
    elif model_name == 'wdsr':
        model = wdsr_b(scale=scale, num_res_blocks=depth)
    else:
        NotImplementedError(f"Model {model_name} not implemented")

    loss_object = tf.keras.losses.MeanAbsoluteError()

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                boundaries=[200000], values=[1e-4, 5e-5])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(loss=loss_object,
              optimizer=optimizer, 
              metrics=['MAE', PSNR])

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + '/checkpoints/',
        monitor='MAE',
        mode='min')
    
    model.fit(train_ds,
            epochs=epochs,
            steps_per_epoch=800 // batch_size,
            validation_data=valid_ds,
            validation_steps=100 // batch_size,
            callbacks=[tensorboard_callback,
                        model_checkpoint_callback])
    

    print("Evaluating model...")
    model.evaluate(valid_ds)

    print("Saving model...")
    model.save_weights(f'weights/{model_name}-{depth}-{downgrade}-x{scale}/weights.h5')

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
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--model', type=str, default='edsr',
                        help='Model name, can be edsr or wdsr')
    args = parser.parse_args()

    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("WARNING: No GPU found, running on CPU")

    main(model_name=args.model,
         downgrade=args.downgrade, 
         scale=args.scale,
         batch_size=args.batch_size,
         epochs=args.epochs,
         depth=args.depth)
