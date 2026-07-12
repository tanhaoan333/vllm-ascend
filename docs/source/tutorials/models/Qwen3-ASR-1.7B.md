# Qwen3-ASR-1.7B

## 1 Introduction

The released Qwen3-ASR-1.7B is a lightweight, high-performance automatic speech recognition (ASR) model developed by the Qwen Team. It delivers industry-leading recognition accuracy across Chinese/English multi-scene speech, Chinese dialects, multilingual and singing voice scenarios, with native support for long audio and streaming inference, and deep optimization for Ascend NPU hardware.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Please refer to [Supported Features List](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/support_matrix/supported_models.html) to get the model's supported feature matrix.

Please refer to [Feature Guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/index.html) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

`Qwen3-ASR-1.7B`(BF16 version): requires 1 Ascend 910B (with 1 x 64G NPUs).

`Qwen3-ASR-1.7B`(BF16 version): requires 1 Ascend 310P (with 1 x 48G NPUs).
[Download model weight](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3-ASR models.

**Docker Pull:**

```bash
docker pull quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
```

**Docker Run:**

=== "Atlas 800I A3"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run \
        --name vllm-ascend-env \
        --shm-size=128g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci8 \
        --device /dev/davinci9 \
        --device /dev/davinci10 \
        --device /dev/davinci11 \
        --device /dev/davinci12 \
        --device /dev/davinci13 \
        --device /dev/davinci14 \
        --device /dev/davinci15 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -it -d $IMAGE bash
    ```

    !!! note

        A3 has 8 NPUs with dual-die design (16 chips total: `/dev/davinci[0-15]`).
        If you are on a shared machine, map only the chips you need (e.g., `/dev/davinci[0-7]` for NPU 0-3).

=== "Atlas 800I A2"

    ```bash

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run \
        --name vllm-ascend-env \
        --shm-size=128g \
        --net=host \
        --privileged=true \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -it -d $IMAGE bash
    ```

The default workdir is `/workspace`. vLLM and vLLM-Ascend are installed as Python packages in site-packages.

**Installation Verification:**

After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend-env
```

Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source. Install vLLM from source first:

1. Clone and install vLLM:

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

2. Clone and install the vLLM-Ascend repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

!!! note

    If deploying a multi-node environment, set up the environment on each node.

Please install system dependencies.

```bash
# Used for audio processing.
apt-get update && apt-get install -y ffmpeg
# Check the installation.
ffmpeg -version
```

## 5 Online Service Deployment

PS:Because the model has fewer parameters, it doesn’t involve the PD separation scenario.

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios.

> The following command is an example configuration. Adjust the parameters based on your actual scenario.

**Atlas 800I A2/A3:**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve your_model_path \
    --served-model-name qwen3-asr \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-num-batched-tokens 2048 \
    --max-model-len 512 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_capture_sizes":[128,256,512,1024,2048],"cudagraph_mode": "FULL"}'  \
    --gpu-memory-utilization 0.95 \
    --port 8000 
```

!!! note

    - `ASCEND_RT_VISIBLE_DEVICES`: must be set to the NPU chip IDs allocated to your environment (e.g., `0,1,2,3` for 4 chips).
    - `--port`: adjust to avoid conflicts with other services running on the same machine.
    - `--no-enable-prefix-caching`: disabled by default as prefix caching effectiveness for this model on Ascend NPUs has not been fully characterized. You can try enabling it to evaluate the cache hit rate for your workload.

!!! tip

    For parameter details, refer to:

    - [vLLM CLI documentation](https://docs.vllm.ai/en/stable/cli/) — standard serve parameters (`--host`, `--port`, `--max-model-len`, etc.)
    - [Environment Variables](../../user_guide/configuration/env_vars.md) — Ascend-specific environment variables (`HCCL_*`, etc.)
    - [Additional Configuration](../../user_guide/configuration/additional_config.md) — `--additional-config` format and options

**Service Verification:**

If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=<pid>) INFO:     Started server process [<pid>]
(APIServer pid=<pid>) INFO:     Waiting for application startup.
(APIServer pid=<pid>) INFO:     Application startup complete.
```

**Atlas 300I A2 2UP**

```shell
vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

**Atlas 300I DUO**

```shell
vllm serve "Qwen/Qwen3-ASR-1.7B" \
  --tensor-parallel-size 1 \
  --gpu_memory_utilization 0.9 \
  --dtype float16 \
  --max_model_len 4096 \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,4]}' \
  --port 8000
```

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://localhost:8000/v1/chat/completions
    -H "Content-Type: application/json"
    -d '{
    "messages": [
    {"role": "user", "content": [
        {"type": "audio_url",
        "audio_url":
        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}}
    ]}
    ]
}'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using EvalScope

As an example, take the `aishell` dataset as a test dataset, and run accuracy evaluation of `Qwen3-ASR-1.7B` in online mode.

After all samples were processed, transcription quality was measured using:

- WER (Word Error Rate) for word-level recognition accuracy
- CER (Character Error Rate) for character-level recognition accuracy

### Remarks

This result reflects end-to-end serving performance, including audio preprocessing, request construction, API communication, inference, and response parsing. Actual performance may vary depending on hardware, concurrency, audio length, and deployment configuration.

Further benchmarking is recommended for latency distribution, concurrent throughput, long-audio scenarios, and system resource utilization.
