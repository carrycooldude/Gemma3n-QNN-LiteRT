# Gemma3n-QNN-LiteRT Android Chat

A premium on-device LLM chat application for Android, powered by **Google LiteRT (formerly TensorFlow Lite)** and running the **Gemma-2B (3n)** model.

<p align="center">
  <img src="app/src/main/res/mipmap-xxxhdpi/ic_launcher.png" width="100" />
</p>

## 🚀 Features

*   **On-Device Inference**: Runs entirely offline using the Gemma-2B-it-int4 quantized model.
*   **LiteRT-LM Engine**: efficient LLM inference using Google's LiteRT GenAI stack.
*   **Hardware Acceleration**: Optimized for **GPU** (OpenCL) execution. *(Note: QNN/NPU support is experimentally implemented but currently disabled due to model file constraints).*
*   **Modern Premium UI**:
    *   Deep Blue & Soft Gray aesthetic.
    *   Floating "pill" input bar with keyboard auto-adjustment.
    *   Native Android `RecyclerView` with smooth streaming updates.
    *   Custom vector avatars and message bubbles.
*   **Streaming Responses**: Real-time text generation with typing indicators.

## 🛠️ Setup & Installation

### Prerequisites
*   Android Studio Ladybug (or newer).
*   Android Device with GPU support (Android 10+ recommended).
*   ~2GB free storage for the model.

### 1. Clone the Repository
```bash
git clone https://github.com/carrycooldude/Gemma3n-QNN-LiteRT.git
cd Gemma3n-QNN-LiteRT
```

### 2. Download the Model
Due to size and licensing, the model is not included in the repo.
1.  Download the **`gemma-3n-E2B-it-litert-lm.bin`** (or compatible LiteRT model) from:
    *   [**HuggingFace: google/gemma-3n-E2B-it-litert-lm**](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm)
2.  Rename the downloaded file to `model.bin`.

### 3. Build & Install
Open the project in Android Studio and build `assembleDebug`.
```bash
./gradlew assembleDebug
```

### 4. Push Model to Device
The app looks for the model at `/data/local/tmp/llm/model.bin`.
```bash
adb shell mkdir -p /data/local/tmp/llm/
adb push path/to/your/gemma-2b-it-gpu-int4.bin /data/local/tmp/llm/model.bin
```

## 📱 Usage
1.  Launch the app.
2.  Wait for initialization (GPU delegates loading).
3.  Start chatting! The keyboard will automatically resize the chat window.

## ⚠️ Notes on QNN (NPU)
This project contains code for **Qualcomm QNN (NPU)** acceleration via LiteRT delegates.
*   Currently, the `gemma-3n` model binaries compatible with the specific QNN HTP backend version are limited.
*   The app defaults to **GPU** for maximum stability.
*   To experiment with NPU, modify `LiteRTLMManager.kt` to prioritize `Backend.NPU`.

## 📜 License
Apache 2.0
