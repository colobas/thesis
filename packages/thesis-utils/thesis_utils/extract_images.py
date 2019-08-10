import os
import argparse
import pathlib
import tensorflow as tf

from imageio import imwrite

def save_images_from_event(fn, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    imwrite(output_fn, im)
                    count += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--outdir', type=str, default="./")
    args = parser.parse_args()

    save_images_from_event(args.filename, args.tag, args.outdir)
