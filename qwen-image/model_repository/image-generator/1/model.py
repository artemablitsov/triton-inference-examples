import io
import torch
import numpy as np
import triton_python_backend_utils as pb_utils

from huggingface_hub import hf_hub_download
from diffusers import (
    DiffusionPipeline, 
    QwenImageTransformer2DModel, 
    GGUFQuantizationConfig
)

class TritonPythonModel:
    def initialize(self, args):
        """
        Вызывается Тритоном один раз при старте контейнера или загрузке модели.
        """
        self.base_model_id = "Qwen/Qwen-Image-2512"
        self.gguf_repo_id = "unsloth/Qwen-Image-2512-GGUF"
        self.gguf_filename = "qwen-image-2512-Q2_K.gguf"

        print("Triton: Скачивание/Поиск GGUF файла в кэше...")
        self.gguf_path = hf_hub_download(
            repo_id=self.gguf_repo_id, 
            filename=self.gguf_filename
        )

        print("Triton: Инициализация 2-битного Трансформера (может занять время)...")
        quant_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
        
        transformer = QwenImageTransformer2DModel.from_single_file(
            self.gguf_path,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            config=self.base_model_id,
            subfolder="transformer",
#            device='cuda',
            low_cpu_mem_usage=True
        )

        print("Triton: Сборка DiffusionPipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            self.base_model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
 #           device='cuda',
            low_cpu_mem_usage=True
        )
        
        print("Triton: Включение выгрузки неактивных модулей в RAM...")
        # Критически важно для работы на видеокартах до 24-32 Гб
        self.pipe.enable_model_cpu_offload()
        print("Triton: Модель Qwen-Image (GGUF 2-bit) успешно загружена и готова!")

    def execute(self, requests):
        """
        Вызывается при каждом HTTP/GRPC запросе к Тритону.
        """
        responses = []
        
        for request in requests:
            # Читаем входной промпт
            in_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt = in_tensor.as_numpy()[0].decode("utf-8")
           
            # Задаем дефолтные значения
            steps = 12
            cfg_scale = 5.0
            width = 512
            height = 512
            negative_prompt = " "
            seed = None

            # Читаем входной негативный промпт
            np_tensor = pb_utils.get_input_tensor_by_name(request, "negative_prompt")
            if np_tensor is not None:
                negative_prompt = np_tensor.as_numpy()[0].decode("utf-8") 

            # Читаем опциональные параметры (если клиент их передал)
            steps_tensor = pb_utils.get_input_tensor_by_name(request, "num_inference_steps")
            if steps_tensor is not None:
                steps = int(steps_tensor.as_numpy()[0])

            cfg_tensor = pb_utils.get_input_tensor_by_name(request, "guidance_scale")
            if cfg_tensor is not None:
                cfg_scale = float(cfg_tensor.as_numpy()[0])

            seed_tensor = pb_utils.get_input_tensor_by_name(request, "seed")
            if seed_tensor is not None:
                seed = int(seed_tensor.as_numpy()[0])

            # Подготавливаем генератор для фиксированного сида
            generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

            # Запускаем генерацию с новыми параметрами
            with torch.inference_mode():
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]
            
            # Конвертация в байты PNG
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            
            # Создаем выходной тензор
            out_tensor = pb_utils.Tensor(
                "image_bytes", 
                np.array([img_bytes], dtype=np.object_)
            )
            
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            
        return responses

    def finalize(self):
        """Очистка памяти при выгрузке модели Тритоном"""
        self.pipe = None
        torch.cuda.empty_cache()
