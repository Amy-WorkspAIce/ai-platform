# Install on Unraid

## Prerequisites

- Unraid 6.12+ (7.x recommended).
- **For GPU:** the **Nvidia Driver** plugin (Apps → search "Nvidia Driver"),
  installed and rebooted. Note your GPU's UUID on the plugin's settings page.

## Install

**From Community Applications** *(once published)*: Apps → search "Amy AI Platform" → Install.

**Manually, until then:**

1. Open the Unraid terminal and fetch the template:
   ```bash
   wget -O /boot/config/plugins/dockerMan/templates-user/my-amy-ai-platform.xml \
     https://raw.githubusercontent.com/Amy-WorkspAIce/ai-platform/main/deploy/unraid/amy-ai-platform.xml
   ```
2. Docker tab → **Add Container** → Template: `amy-ai-platform`.
3. Review the settings:
   - **Data:** `/mnt/user/appdata/amy-ai-platform`. Models are several GB each,
     so put appdata on an SSD/cache pool if you can.
   - **NVIDIA Visible Devices:** `all`, or your GPU's UUID.
   - **No GPU?** Under *Advanced View*, remove `--runtime=nvidia` from **Extra Parameters**.
4. **Apply.**

## First run

1. Open the container's **Logs**. You'll see:
   ```
   Your API key (shown once): amy-XXXXXXXX
   ```
   It's also saved at `/mnt/user/appdata/amy-ai-platform/secrets/api_key`.
2. Open the **WebUI** (`http://<server-ip>:8484`) and paste the key.
3. Go to **Models**. Amy shows your GPU and recommends models that fit it. Click
   **Install** on one. The first install on a new GPU spends a few extra minutes on
   "Optimizing for your GPU (first run only)" while CUDA compiles kernels for your card.
   This happens once and is cached in appdata.
4. Back in **Chat**, start talking. Your chats are saved in the sidebar.

**Images and audio:** install an image model (e.g. FLUX.2 [klein]) or an audio model
(Whisper, Kokoro) from **Models**. The first one of each kind also installs its engine
into appdata: about 7 GB for images (PyTorch) and about 2 GB for audio. This happens once.
Use the **Images** and **Audio** pages, or the mic and speaker buttons in chat.

> The microphone only works over HTTPS (or on `localhost`), a browser rule. Over plain
> `http://<server-ip>:8484`, uploading audio files works, and so does reading replies aloud.
> For dictation, put Amy behind a reverse proxy with HTTPS (e.g. SWAG or Nginx Proxy Manager).

## Verify the GPU is used

In the container console, run `ollama ps` while a chat is generating. The
`PROCESSOR` column should say `100% GPU`. You can also watch `nvidia-smi` in the
Unraid terminal.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Container won't start: `unknown or invalid runtime name: nvidia` | Install the Nvidia Driver plugin, or remove `--runtime=nvidia`. |
| Models run on CPU | Check `NVIDIA_VISIBLE_DEVICES`. Make sure no other container is holding all the VRAM. |
| First message after a fresh install takes minutes | A one-time CUDA compile. Installs through the Models page do it up front. Models pulled from the console will do it on first use. |
| Permission errors in appdata | Check PUID/PGID (99/100), then run `chown -R 99:100 /mnt/user/appdata/amy-ai-platform`. |
