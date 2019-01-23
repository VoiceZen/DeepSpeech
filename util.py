def infer_batch(ds2_model, infer_data, data_generator, codec, args):
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

    return res
