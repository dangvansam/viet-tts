import io
import os
import queue
import random
import subprocess
import threading
import wave

import tempfile
import shutil
import requests
import numpy as np
from loguru import logger
from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel
from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from viettts.tts import TTS
from viettts.utils.file_utils import load_prompt_speech_from_file, load_voices


VOICE_DIR = 'samples'
VOICE_MAP = load_voices(VOICE_DIR)
MODEL_DIR = 'pretrained-models'

global tts_obj
tts_obj = None


app = FastAPI(
    title="VietTTS API",
    description="""
    VietTTS API (https://github.com/dangvansam/viet-tts)
    Vietnamese Text To Speech and Voice Clone
    License: Apache 2.0 - Author: <dangvansam dangvansam98@gmail.com>
    """
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    audio = wav_chunk_header()
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16)
        tts_audio = tts_audio.tobytes()
        audio += tts_audio
    yield audio


class OpenAITTSRequest(BaseModel):
    input: str
    model: str = "tts-1"
    voice: str = random.choice(list(VOICE_MAP))
    response_format: str = "wav"
    speed: float = 1.0

class TTSRequest(BaseModel):
    text: str
    voice: str = random.choice(list(VOICE_MAP))
    speed: float = 1.0

def wav_chunk_header(sample_rate=22050, bit_depth=16, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


@app.get("/", response_class=PlainTextResponse)
async def root():
    return 'VietTTS API'

@app.get("/health", response_class=PlainTextResponse)
async def health():
    return 'VietTTS API is running...'

@app.get("/voices")
@app.get("/v1/voices")
async def show_voices():
    return list(VOICE_MAP.keys()) 

@app.post("/audio/speech")
@app.post("/v1/audio/speech")
async def openai_api_tts(tts_request: OpenAITTSRequest):
    logger.info(f"Received TTS request: {tts_request.dict()}")
    
    if tts_request.voice.isdigit():
        voice_file = list(VOICE_MAP.values())[int(tts_request.voice)]
    else:
        voice_file = VOICE_MAP.get(tts_request.voice)

    if not voice_file:
        logger.error(f"Voice {tts_request.voice} not found")
        return PlainTextResponse(content="Voice not found", status_code=404)

    prompt_speech_16k = load_prompt_speech_from_file(
        filepath=voice_file,
        min_duration=3,
        max_duration=5
    )
    # prompt_speech_16k = fade_in_out_audio(prompt_speech_16k)

    def build_ffmpeg_args(response_format, input_format, sample_rate=24000):
        if input_format == 'WAV':
            ffmpeg_args = ["ffmpeg", "-loglevel", "error", "-f", "WAV", "-i", "-"]
        else:
            ffmpeg_args = ["ffmpeg", "-loglevel", "error", "-f", input_format, "-ar", sample_rate, "-ac", "1", "-i", "-"]
        if response_format == "mp3":
            ffmpeg_args.extend(["-f", "mp3", "-c:a", "libmp3lame", "-ab", "64k"])
        elif response_format == "opus":
            ffmpeg_args.extend(["-f", "ogg", "-c:a", "libopus"])
        elif response_format == "aac":
            ffmpeg_args.extend(["-f", "adts", "-c:a", "aac", "-ab", "64k"])
        elif response_format == "flac":
            ffmpeg_args.extend(["-f", "flac", "-c:a", "flac"])
        elif response_format == "wav":
            ffmpeg_args.extend(["-f", "wav", "-c:a", "pcm_s16le"])
        elif response_format == "pcm": # even though pcm is technically 'raw', we still use ffmpeg to adjust the speed
            ffmpeg_args.extend(["-f", "s16le", "-c:a", "pcm_s16le"])
        return ffmpeg_args

    def exception_check(exq: queue.Queue):
        try:
            e = exq.get_nowait()
        except queue.Empty:
            return
        raise e

    if tts_request.response_format == "mp3":
        media_type = "audio/mpeg"
    elif tts_request.response_format == "opus":
        media_type = "audio/ogg;codec=opus" # codecs?
    elif tts_request.response_format == "aac":
        media_type = "audio/aac"
    elif tts_request.response_format == "flac":
        media_type = "audio/x-flac"
    elif tts_request.response_format == "wav":
        media_type = "audio/wav"
    elif tts_request.response_format == "pcm":
        media_type = "audio/pcm;rate=24000"
    else:
        raise ValueError(f"Invalid response_format: '{tts_request.response_format}'", param='response_format')

    ffmpeg_args = None
    ffmpeg_args = build_ffmpeg_args(tts_request.response_format, input_format="f32le", sample_rate="24000")
    ffmpeg_args.extend(["-"])
    ffmpeg_proc = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    in_q = queue.Queue()
    ex_q = queue.Queue()

    def generator():
        # text -> in_q
        try:
            model_output = tts_obj.inference_tts(
                tts_text=tts_request.input,
                prompt_speech_16k=prompt_speech_16k,
                speed=tts_request.speed,
                stream=False
            )
            for chunk in model_output:
                exception_check(ex_q)
                chunk = chunk['tts_speech'].numpy().tobytes()
                in_q.put(chunk)
        except BrokenPipeError as e:
            logger.info("Client disconnected - 'Broken pipe'")
        except Exception as e:
            logger.error(f"Exception: {repr(e)}")
            raise e
        finally:
            in_q.put(None) # sentinel

    def out_writer(): 
        try:
            while True:
                chunk = in_q.get()
                if chunk is None: # sentinel
                    break
                ffmpeg_proc.stdin.write(chunk) # BrokenPipeError from here on client disconnect
        except Exception as e: # BrokenPipeError
            ex_q.put(e)  # we need to get this exception into the generation loop
            ffmpeg_proc.kill()
            return
        finally:
            ffmpeg_proc.stdin.close()
            
    generator_worker = threading.Thread(target=generator, daemon=True)
    generator_worker.start()
    out_writer_worker = threading.Thread(target=out_writer, daemon=True)
    out_writer_worker.start()

    async def cleanup():
        try:
            ffmpeg_proc.kill()
            # del generator_worker
            # del out_writer_worker
        except Exception as e:
            logger.error(f"Exception: {repr(e)}")

    return StreamingResponse(
        content=ffmpeg_proc.stdout,
        media_type=media_type,
        background=cleanup
    )

@app.post("/tts")
@app.post("/v1/tts")
async def tts(
    text: str = Form(...),
    voice: str = Form("0"),
    speed: float = Form(1.0),
    audio_url: str = Form(None),
    audio_file: UploadFile = File(None)
):
    logger.info(f"Received TTS request: text={text}, voice={voice}, speed={speed}, audio_url={audio_url}")
    voice_file = None

    # Case 1: Uploaded audio file
    if audio_file:
        temp_audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f'.{audio_file.filename.split(".")[-1]}'
        )
        try:
            with open(temp_audio_file.name, "wb") as temp_file:
                shutil.copyfileobj(audio_file.file, temp_file)
            voice_file = temp_audio_file.name
            logger.info(f"Using uploaded audio file as voice: {voice_file}")
        finally:
            audio_file.file.close()

    # Case 2: Audio URL
    elif audio_url:
        temp_audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f'.{audio_url.lower().split(".")[-1]}'
        )
        try:
            response = requests.get(audio_url, stream=True)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch audio from URL")
            with open(temp_audio_file.name, "wb") as temp_file:
                shutil.copyfileobj(response.raw, temp_file)
            voice_file = temp_audio_file.name
            logger.info(f"Using audio URL as voice: {voice_file}")
        finally:
            response.close()

    # Case 3: Predefined voice
    elif voice:
        if voice.isdigit():
            voice_file = list(VOICE_MAP.values())[int(voice)]
        else:
            voice_file = VOICE_MAP.get(voice)

        if not voice_file:
            logger.error(f"Voice {voice} not found")
            raise HTTPException(status_code=404, detail="Voice not found")
    
    else:
        voice_file = random.choice(list(VOICE_MAP.values()))

    # Error if no voice file is available
    if not voice_file or not os.path.exists(voice_file):
        raise HTTPException(status_code=400, detail="No valid voice file provided")

    prompt_speech_16k = load_prompt_speech_from_file(
        filepath=voice_file,
        min_duration=2,
        max_duration=4
    )

    temp_output_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    )

    try:
        model_output = tts_obj.inference_tts(
            tts_text=text,
            prompt_speech_16k=prompt_speech_16k,
            speed=speed,
            stream=False
        )

        raw_audio = b''.join(chunk['tts_speech'].numpy().tobytes() for chunk in model_output)
        ffmpeg_args = [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", "24000", "-ac", "1",
            "-i", "-", "-f", "mp3", "-c:a", "libmp3lame", "-ab", "64k", temp_output_file.name
        ]
        ffmpeg_proc = subprocess.run(
            ffmpeg_args,
            input=raw_audio,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if ffmpeg_proc.returncode != 0:
            logger.error(f"FFmpeg error: {ffmpeg_proc.stderr.decode()}")
            raise HTTPException(status_code=500, detail="Error during audio processing")

        if not os.path.exists(temp_output_file.name):
            logger.error(f"FFmpeg did not create the output file: {temp_output_file.name}")
            raise HTTPException(status_code=500, detail="FFmpeg failed to produce the output file")

        return FileResponse(
            path=temp_output_file.name,
            media_type="audio/mpeg",
            filename=temp_output_file.name.split("/")[-1]
        )

    finally:
        if audio_file or audio_url:
            if os.path.exists(temp_audio_file.name):
                os.unlink(temp_audio_file.name)


@app.on_event("startup")
async def startup():
    global tts_obj
    RunVar("_default_thread_limiter").set(CapacityLimiter(os.cpu_count()))
    tts_obj = TTS(model_dir=MODEL_DIR)