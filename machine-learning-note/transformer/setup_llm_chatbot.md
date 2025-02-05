# LLM Setup


## Setup Ollama

- [ollama/ollama](https://github.com/ollama/ollama.git)
- [download LLM models](https://ollama.com/search)

```bash
# curl http://127.0.0.1:11434/
Ollama is running

# display help message
# ollama help

# ollama list
NAME               ID              SIZE      MODIFIED     
deepseek-r1:14b    ea35dfe18182    9.0 GB    24 hours ago    
llama3.2:latest    a80c4f17acd5    2.0 GB    3 days ago

# ollama show llama3.2
  Model
    architecture        llama     
    parameters          3.2B      
    context length      131072    
    embedding length    3072      
    quantization        Q4_K_M    

  Parameters
    stop    "<|start_header_id|>"    
    stop    "<|end_header_id|>"      
    stop    "<|eot_id|>"             

  License
    LLAMA 3.2 COMMUNITY LICENSE AGREEMENT                 
    Llama 3.2 Version Release Date: September 25, 2024 

# ollama run llama3.2:latest
>>> nice to meet you
Nice to meet you too! I'm a large language model, so I don't have personal experiences or emotions like humans do, but I'm here to help and chat with 
you. How's your day going so far?

...

# ollama ps
NAME               ID              SIZE      PROCESSOR    UNTIL              
llama3.2:latest    a80c4f17acd5    4.0 GB    100% GPU     3 minutes from now 

# ollama stop llama3.2:latest

```

## Setup open-webui

- [open-webui/open-webui](https://github.com/open-webui/open-webui.git)

- run open-webui

```bash
# open-webui serve --help
                                                                                                                                   
 Usage: open-webui serve [OPTIONS]                                                                                                 
                                                                                                                                   
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --host        TEXT     [default: 0.0.0.0]                                                                                       │
│ --port        INTEGER  [default: 8080]                                                                                          │
│ --help                 Show this message and exit.                                                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


# open-webui serve

INFO  [open_webui.env] Embedding model set: sentence-transformers/all-MiniLM-L6-v2
INFO  [open_webui.env] 'WHISPER_MODEL' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_STT_OPENAI_API_BASE_URL' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_STT_OPENAI_API_KEY' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_STT_ENGINE' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_STT_MODEL' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_OPENAI_API_BASE_URL' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_OPENAI_API_KEY' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_API_KEY' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_ENGINE' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_MODEL' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_VOICE' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_SPLIT_ON' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_AZURE_SPEECH_REGION' loaded from the latest database entry
INFO  [open_webui.env] 'AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT' loaded from the latest database entry
WARNI [langchain_community.utils.user_agent] USER_AGENT environment variable not set, consider setting it to identify your requests.


Fetching 30 files: 100%|████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 99391.09it/s]
INFO:     Started server process [13277]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)

# try open-webui in browser with `http://127.0.0.1:8080` (you maynot use `0.0.0.0` to avoid `Connection is not secure` warnning, otherwise you would have trouble to call Mic or speaker)
```

- set local STT and TTS

![open-webui TTS, STT Settings](../images/open-webui_tts_stt_setting.png)

- deploy local TTS server
  - [TTS Server: matatonic/openedai-speech](https://github.com/matatonic/openedai-speech.git)
  - [TTS Model: rhasspy/piper-voices](https://hf-mirror.com/rhasspy/piper-voices)

- openedai-speech server dependencies

```
# requirement.txt
--index-url=https://mirrors.aliyun.com/pypi/simple
--find-links=https://mirrors.aliyun.com/pytorch-wheels/cu121
fastapi
uvicorn
loguru
pyyaml
torch==2.1.0+cu121
torchaudio==2.1.0+cu121
langdetect==1.0.9
numpy==1.22.0
TTS==0.22.0
piper-tts==1.2.0
# must come after `piper-tts`
onnxruntime-gpu==1.20.1
```

- tricks

```
# accelerate models fetches
export HF_ENDPOINT=https://hf-mirror.com
# set open-webui in offline mode
export HF_HUB_OFFLINE=1
# Ignore models from openai if you cannot use openai either
export ENABLE_OPENAI_API=0
```

- install stable-diffusion (you need VPN to run webui)
  - [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion.git)
  - [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui.git)
  - [where to find finetuned models: civitai.work](https://civitai.work/models)
  - [Civitai - how to use models](https://github.com/civitai/civitai/wiki/How-to-use-models)
  - [Flux 1.x Quick Guide](https://education.civitai.com/quickstart-guide-to-flux-1/)
