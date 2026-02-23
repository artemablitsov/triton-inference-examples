import numpy as np
import tritonclient.http as httpclient

timeout = 60*60*3

def generate_via_triton(prompt: str, output_path: str = "output.png"):
    client = httpclient.InferenceServerClient(
            url="localhost:8000",
            connection_timeout=timeout,
            network_timeout=timeout,
        )
    
    # Подготавливаем вход
    input_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
    inputs = [httpclient.InferInput("prompt", [1], "BYTES")]
    inputs[0].set_data_from_numpy(input_data)
    
    # Запрашиваем выход
    outputs = [httpclient.InferRequestedOutput("image_bytes")]
    
    print("Отправка запроса к Triton. Ждем генерацию...")
    # Таймаут нужен большой, т.к. генерация диффузии — долгий процесс
    response = client.infer(
        model_name="image-generator",
        inputs=inputs,
        outputs=outputs,
        timeout=timeout
    )
    
    # Сохраняем результат
    image_bytes = response.as_numpy("image_bytes")[0]
    with open(output_path, "wb") as f:
        f.write(image_bytes)
        
    print(f"Готово! Сохранено в {output_path}")

if __name__ == "__main__":
    generate_via_triton(
        "A hyper-realistic photo of a cute Shiba Inu drinking coffee in a Parisian cafe, cinematic lighting.",
        "shiba_triton.png"
    )
