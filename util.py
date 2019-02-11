import time
import pandas as pd
import paddle.v2 as paddle

from model_utils.model import DeepSpeech2Model


def infer_batch(infer_data, data_generator, codec, batch_id, args):
    st1 = time.time()
    if args.use_gpu:
        paddle.init(
            use_gpu=True,
            rnn_use_batch=True,
            trainer_count=args.trainer_count)
    else:
        paddle.init(use_gpu=False, trainer_count=1)

    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.model_path,
        share_rnn_weights=args.share_rnn_weights)

    print("Processing batch {}".format(batch_id))
    st2 = time.time()
    # decoders only accept string encoded in utf-8
    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

    # ds2_model.logger.info("start inference ...")

    probs_split = ds2_model.infer_batch_probs(
        infer_data=infer_data,
        feeding_dict=data_generator.feeding)

    result_transcripts = ds2_model.decode_batch_greedy(
        probs_split=probs_split,
        vocab_list=vocab_list)

    result_encodings = ds2_model.get_encoded_strings(
        probs_split=probs_split,
        codec=codec
    )

    res = []
    for target, result, encode in zip(
        infer_data,
        result_transcripts,
        result_encodings
    ):
        res.append([
            target[3],
            target[1],
            result,
            encode
        ])

    cols = ['filepath', 'orig_script', 'infer_script', 'encode']
    df = pd.DataFrame(res, columns=cols)
    df.to_csv(
        'batch-infer-{}.csv'.format(batch_id + 1),
        index=False,
        header=False
    )
    et = time.time()

    total_time1 = (et - st1)
    total_time2 = (et - st2)
    duration = 10
    print("Time taken to process batch of duration {} is {}-{}".format(
        duration, total_time1, total_time2))
    return res

