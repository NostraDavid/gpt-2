#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import gpt_2.model as model
import gpt_2.sample as sample
import gpt_2.encoder as encoder


def interact_model(
    model_name: str = "124M",
    seed: int | None = None,
    nsamples: int = 1,
    batch_size: int = 1,
    length: int | None = None,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 1,
    models_dir: str = "models",
) -> None:
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, "hparams.json")) as f:
        hparams_dict = json.load(f)
        for k, v in hparams_dict.items():
            setattr(hparams, k, v)
    # Use getattr to avoid attribute errors if n_ctx is missing
    n_ctx = getattr(hparams, 'n_ctx', 1024)
    if length is None:
        length = n_ctx // 2
    elif length > n_ctx:
        raise ValueError(f"Can't get samples longer than window size: {n_ctx}")
    # TensorFlow 2.x compatibility
    import tensorflow.compat.v1 as tf1
    tf1.disable_v2_behavior()
    with tf1.Session(graph=tf1.Graph()) as sess:
        context = tf1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        saver = tf1.train.Saver()
        ckpt = tf1.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print("Prompt should not be empty!")
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(
                    output,
                    feed_dict={context: [context_tokens for _ in range(batch_size)]},
                )[:, len(context_tokens) :]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


if __name__ == "__main__":
    fire.Fire(interact_model)
