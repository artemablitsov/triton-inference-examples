docker run --gpus all -it --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/Source/Triton-examples/flux/model_repository:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  triton-server
