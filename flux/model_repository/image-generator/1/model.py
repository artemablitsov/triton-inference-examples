import io
import torch
import numpy as np
import triton_python_backend_utils as pb_utils

from diffusers import Flux2KleinPipeline

class TritonPythonModel:
    def initialize(self, args):
        """
        Вызывается Тритоном один раз при старте контейнера или загрузке модели.
        """

        print("Triton: Запуск flux.")
        self.pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        print("Triton: Модель Flux успешно загружена и готова!")

    def execute(self, requests):
        """
        Вызывается при каждом HTTP/GRPC запросе к Тритону.
        """
        responses = []
        
        for request in requests:
            # Читаем входной промпт
            in_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt = in_tensor.as_numpy()[0].decode("utf-8")
            print(f"Triton: Взял в работу промпт: {prompt}.")

            # Задаем дефолтные значения
            steps = 4
            cfg_scale = 5.0
            width = 2048
            height = 2048
            seed = None

            # Читаем опциональные параметры (если клиент их передал)
            width_tensor = pb_utils.get_input_tensor_by_name(request, "width")
            if width_tensor is not None:
                steps = int(width_tensor.as_numpy()[0])

            height_tensor = pb_utils.get_input_tensor_by_name(request, "height")
            if height_tensor is not None:
                steps = int(height_tensor.as_numpy()[0])

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
            print(f"Triton: Начинаю генерацию.")
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]
            print(f"Triton: Генерация закончена.")            

            print(f"Triton: Конвертирую данные.")            
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
            print(f"Triton: Промпт обработан успешно.")       
            
        return responses

    def finalize(self):
        """Очистка памяти при выгрузке модели Тритоном"""
        self.pipe = None
        torch.cuda.empty_cache()
