from pathlib import Path
from runtime import configure_runtime

configure_runtime()

from kokoro import KModel, KPipeline
import gradio as gr
import os
import random
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
REPO_ID = 'hexgrad/Kokoro-82M'
DATA_DIR = Path(__file__).resolve().parent
models = {CUDA_AVAILABLE: KModel(repo_id=REPO_ID).to('cuda' if CUDA_AVAILABLE else 'cpu').eval()}
pipelines = {lang_code: KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

@torch.inference_mode()
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)


@torch.inference_mode()
def forward_cpu(ps, ref_s, speed):
    if False not in models:
        models[False] = KModel(repo_id=REPO_ID).eval()
    return models[False](ps, ref_s, speed)

def generate_first(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = forward_cpu(ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                audio = forward_cpu(ps, ref_s, speed)
            else:
                raise gr.Error(e)
        return (24000, audio.mul(32767).clamp(-32768, 32767).to(torch.int16).numpy()), ps
    return None, ''

# Arena API
def predict(text, voice='af_heart', speed=1):
    return generate_first(text, voice, speed, use_gpu=CUDA_AVAILABLE)[0]

def tokenize_first(text, voice='af_heart'):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = forward_cpu(ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Switching to CPU')
                audio = forward_cpu(ps, ref_s, speed)
            else:
                raise gr.Error(e)
        yield 24000, audio.mul(32767).clamp(-32768, 32767).to(torch.int16).numpy()

with (DATA_DIR / 'en.txt').open(encoding='utf-8') as r:
    random_quotes = [line.strip() for line in r]

def get_random_quote():
    return random.choice(random_quotes)

def get_gatsby():
    with (DATA_DIR / 'gatsby5k.md').open(encoding='utf-8') as r:
        return r.read().strip()

def get_frankenstein():
    with (DATA_DIR / 'frankenstein5k.md').open(encoding='utf-8') as r:
        return r.read().strip()

CHOICES = {
'🇺🇸 🚺 Heart ❤️': 'af_heart',
'🇺🇸 🚺 Bella 🔥': 'af_bella',
'🇺🇸 🚺 Nicole 🎧': 'af_nicole',
'🇺🇸 🚺 Aoede': 'af_aoede',
'🇺🇸 🚺 Kore': 'af_kore',
'🇺🇸 🚺 Sarah': 'af_sarah',
'🇺🇸 🚺 Nova': 'af_nova',
'🇺🇸 🚺 Sky': 'af_sky',
'🇺🇸 🚺 Alloy': 'af_alloy',
'🇺🇸 🚺 Jessica': 'af_jessica',
'🇺🇸 🚺 River': 'af_river',
'🇺🇸 🚹 Michael': 'am_michael',
'🇺🇸 🚹 Fenrir': 'am_fenrir',
'🇺🇸 🚹 Puck': 'am_puck',
'🇺🇸 🚹 Echo': 'am_echo',
'🇺🇸 🚹 Eric': 'am_eric',
'🇺🇸 🚹 Liam': 'am_liam',
'🇺🇸 🚹 Onyx': 'am_onyx',
'🇺🇸 🚹 Santa': 'am_santa',
'🇺🇸 🚹 Adam': 'am_adam',
'🇬🇧 🚺 Emma': 'bf_emma',
'🇬🇧 🚺 Isabella': 'bf_isabella',
'🇬🇧 🚺 Alice': 'bf_alice',
'🇬🇧 🚺 Lily': 'bf_lily',
'🇬🇧 🚹 George': 'bm_george',
'🇬🇧 🚹 Fable': 'bm_fable',
'🇬🇧 🚹 Lewis': 'bm_lewis',
'🇬🇧 🚹 Daniel': 'bm_daniel',
}
pipelines['a'].load_voice('af_heart')

TOKEN_NOTE = '''
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`

💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
'''

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    with gr.Accordion('Output Tokens', open=True):
        out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio, up to 510 context length.')
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button('Predict', variant='secondary', visible=False)

STREAM_NOTE = ['Streams every text chunk. Use Stop to cancel generation.']
STREAM_NOTE = '\n\n'.join(STREAM_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
    with gr.Accordion('Note', open=True):
        gr.Markdown(STREAM_NOTE)

API_OPEN = True
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text', info=f"Arbitrarily many characters supported")
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice', info='Quality and availability vary by language')
                use_gpu = gr.Dropdown(
                    [('NVIDIA CUDA', True), ('CPU', False)],
                    value=CUDA_AVAILABLE,
                    label='Hardware',
                    info=torch.cuda.get_device_name(0) if CUDA_AVAILABLE else 'CUDA unavailable',
                    interactive=CUDA_AVAILABLE
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            random_btn = gr.Button('🎲 Random Quote 💬', variant='secondary')
            with gr.Row():
                gatsby_btn = gr.Button('🥂 Gatsby 📕', variant='secondary')
                frankenstein_btn = gr.Button('💀 Frankenstein 📗', variant='secondary')
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ['Generate', 'Stream'])
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])
    generate_btn.click(fn=generate_first, inputs=[text, voice, speed, use_gpu], outputs=[out_audio, out_ps], concurrency_id='tts', concurrency_limit=1)
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])
    stream_event = stream_btn.click(fn=generate_all, inputs=[text, voice, speed, use_gpu], outputs=[out_stream], concurrency_id='tts', concurrency_limit=1)
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(fn=predict, inputs=[text, voice, speed], outputs=[out_audio], concurrency_id='tts', concurrency_limit=1)

if __name__ == '__main__':
    if CUDA_AVAILABLE:
        forward_gpu('həlˈO wˈɜɹld', pipelines['a'].load_voice('af_heart')[11], 1)
        print(f'CUDA ready: {torch.cuda.get_device_name(0)}; threads={torch.get_num_threads()}', flush=True)
    app.queue(api_open=API_OPEN, max_size=16).launch(
        server_name=os.getenv('KOKORO_HOST', '127.0.0.1'),
        server_port=int(os.getenv('KOKORO_PORT', '7860')),
        show_api=API_OPEN,
        inbrowser=os.getenv('KOKORO_OPEN_BROWSER', '0') == '1',
    )
