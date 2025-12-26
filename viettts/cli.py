import os
import sys
import time
import click
import subprocess
from loguru import logger
from rich.table import Table
from rich.console import Console
from viettts.tts import TTS
from viettts.utils.file_utils import load_prompt_speech_from_file, load_voices


AUDIO_DIR = 'samples'
MODEL_DIR = 'pretrained-models'

@click.command('server')
@click.option('-h', '--host', type=str, default='0.0.0.0', help="The host address to bind the server to. Default is '0.0.0.0'.")
@click.option('-p', '--port', type=int, default=8298, help="The port number to bind the server to. Default is 8298.")
@click.option('-w', '--workers', type=int, default=1, help="The number of worker processes to handle requests. Default is 1.")
def start_server(host: str, port: int, workers: int):
    """Start API server (OpenAI TTS API compatible).

    Usage: viettts server --host 0.0.0.0 --port 8298 -w 4
    """
    logger.info("Starting server")
    cmd = f'gunicorn viettts.server:app \
        -k uvicorn.workers.UvicornWorker \
        --bind {host}:{port} \
        --workers {workers} \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --timeout 300 \
        --keep-alive 75 \
        --graceful-timeout 60'

    subprocess.call(cmd, shell=True, stdout=sys.stdout)


@click.command('synthesis')
@click.option('-t', "--text", type=str, required=True, help="The input text to synthesize into speech.")
@click.option('-v', "--voice", type=str, default='1', help="The voice ID or file path to clone the voice from. Default is '1'.")
@click.option('-s', "--speed", type=float, default=1, help="The speed multiplier for the speech. Default is 1 (normal speed).")
@click.option('-o', "--output", type=str, default='output.wav', help="The file path to save the synthesized audio. Default is 'output.wav'.")
def synthesis(text: str, voice: str, speed: float, output: str):
    """Synthesis audio from text and save to file.

    Usage: viettts synthesis --text 'Xin chào VietTTS' --voice nu-nhe-nhang --voice 8 --speed 1.2 --output test_nu-nhe-nhang.wav
    """
    logger.info("Starting synthesis")
    st = time.perf_counter()
    if not text:
        logger.error('text must not empty')
        return
    
    if speed > 2 or speed < 0.5:
        logger.error(f'speed must in range 0.5-2.0')
        return

    if not os.path.exists(voice):
        voice_map = load_voices(AUDIO_DIR)
        if voice.isdigit():
            voice = list(voice_map.values())[int(voice)]
        else:
            voice = voice_map.get(voice)

    if not os.path.exists(voice):
        logger.error(f'voice is not available. Use --voice <voice-name/voice-id/local-file> or run `viettts show-voices` to get available voices.')
        return

    logger.info('Loading model')
    tts = TTS(model_dir=MODEL_DIR)
    
    logger.info('Loading voice')
    voice = load_prompt_speech_from_file(voice)
    
    logger.info('Processing')
    tts.tts_to_file(text, voice, speed, output)
    
    et = time.perf_counter()
    logger.success(f"Saved to: {output} [time cost={et-st:.2f}s]")


@click.command('show-voices')
def show_voice():
    """Print all available voices.

    Usage: viettts show-voices
    """
    voice_map = load_voices(AUDIO_DIR)
    console = Console()
    table = Table(show_header=True, header_style="green", show_lines=False)
    table.add_column("Voice ID", width=10)
    table.add_column("Voice Name", width=30)
    table.add_column("File", justify="left")
    
    for i, (voice_name, voice_path) in enumerate(voice_map.items()):
        table.add_row(str(i), voice_name, voice_path)

    console.print(table)


@click.group()
def cli():
    """
    VietTTS CLI v0.1.0
    
    Vietnamese Text To Speech and Voice Clone
    License: Apache 2.0 - Author: <dangvansam dangvansam98@gmail.com>
    """
    pass

cli.add_command(start_server)
cli.add_command(synthesis)
cli.add_command(show_voice)