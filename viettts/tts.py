import os
import numpy as np
from tqdm import tqdm
from loguru import logger
from hyperpyyaml import load_hyperpyyaml

from viettts.model import TTSModel
from viettts.frontend import TTSFrontEnd
from viettts.utils.file_utils import download_model, save_wav


class TTS:
    def __init__(
        self,
        model_dir,
        load_jit=False,
        load_onnx=False
    ):
        required_files = [
            "config.yaml", "speech_embedding.onnx", "speech_tokenizer.onnx", 
            "llm.pt", "flow.pt", "hift.pt"
        ]
        files_exist = os.path.exists(model_dir) and all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

        if not files_exist:
            logger.info(f'Downloading model from huggingface [dangvansam/viet-tts]')
            download_model(model_dir)

        with open(f'{model_dir}/config.yaml', 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = TTSFrontEnd(
            speech_embedding_model=f'{model_dir}/speech_embedding.onnx',
            speech_tokenizer_model=f'{model_dir}/speech_tokenizer.onnx'
        )
        self.model = TTSModel(
            llm=configs['llm'],
            flow=configs['flow'],
            hift=configs['hift']
        )
        self.model.load(
            llm_model=f'{model_dir}/llm.pt',
            flow_model=f'{model_dir}/flow.pt',
            hift_model=f'{model_dir}/hift.pt'
        )
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.fp16.zip'.format(model_dir),
                                '{}/llm.llm.fp16.zip'.format(model_dir),
                                '{}/flow.encoder.fp32.zip'.format(model_dir))
            logger.success('Loaded jit model from {}'.format(model_dir))
        
        if load_onnx:
            self.model.load_onnx('{}/flow.decoder.estimator.fp32.onnx'.format(model_dir))
            logger.success('Loaded onnx model from {}'.format(model_dir))

        logger.success('Loaded model from {}'.format(model_dir))
        self.model_dir = model_dir

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_tts(self, tts_text, prompt_speech_16k, stream=False, speed=1.0):
        for i in tqdm(self.frontend.preprocess_text(tts_text, split=True)):
            model_input = self.frontend.frontend_tts(i, prompt_speech_16k)
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                yield model_output

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k)
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            yield model_output

    def tts_to_wav(self, text, prompt_speech_16k, speed=1.0):
        wavs = []
        for output in self.inference_tts(text, prompt_speech_16k, stream=False, speed=speed):
            wavs.append(output['tts_speech'].squeeze(0).numpy())
        return np.concatenate(wavs, axis=0)
    
    def tts_to_file(self, text, prompt_speech_16k, speed, output_path):
        wav = self.tts_to_wav(text, prompt_speech_16k, speed)
        save_wav(wav, 22050, output_path)