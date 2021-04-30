'''Driver script for deepspeech2 service'''
import argparse
import logging

from flask import Flask, request, abort, jsonify
from transcriber import DeepSpeechTransriber
from asr_utils import convert_audio

# Service app session
app = Flask(__name__)

# logger
logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
logger = logging.getLogger("flask")

@app.route("/transcribe", methods=['POST'])
@app.route("/ds2_transcribe", methods=['POST'])
def ds2_transcribe():
    '''Main function to handle the request for transcription'''
    audiodata = request.files['wav_file']
    dest_audio = convert_audio(audiodata)
    logger.info("processed audio is %s" % dest_audio)
    if not dest_audio:
        return abort(400, {'message': 'Unable to parse required audio'})
    rle, lm_transcript = transcriber.transcribe([dest_audio])
    print (lm_transcript)
    print (rle)

    response_dict = {}
    response_dict['rle'] = rle[0]
    response_dict['lm'] = lm_transcript[0].decode('utf-8')
    response_dict['greedy'] = "greedy"
    response_dict['language'] = "english"
    response_dict['dest_wav'] = dest_audio
    return jsonify(response_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ai_mode",
        default=False,
        type=lambda x: (str(x).lower() == 'true'),
        help="Import AI modules Default: %(default)s.")
    parser.add_argument(
        "--port",
        default=5050,
        type=int,
        help="port on which you want to run your service")
    parser.add_argument(
        "--config",
        default="../config/base_config.json",
        type=str,
        help="asr configuration file")
    args = parser.parse_args()

    # Load the deepspeech transcriber model and model
    transcriber = DeepSpeechTransriber()
    transcriber.load_model()

    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=True)
