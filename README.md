# Пример использования Triton inference server для инференса генеративок

Простой пример инференса генеративок qwen-image и flux на [Tritom inference server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/index.html)

## Установка.

### Пререквизиты

Для инференса должен быть установлен [Docker](https://docs.docker.com/engine/install/) и [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Генеративки прожорливые, GPU должна быть с минимум 12 Gb VRAM для flux, а qwen еще более прожорливый...

### Собственно установка

1. Делай раз:
```build.sh```
2. Делай два:
```Поправь в start.sh каталог с моделью на свой```
3. Делай два с половиной:
```Там еще в start.sh монтируется пользовательский кэш с huggingface, можно выключить, тогда при старте будет каждый раз скачиваться модель```
4. Делай три(стартуй или из каталога flux или qwen-image):
```start.sh```
5. Чуть ждем, как загрузятся модели и стартуем: 
```pip install -r ./client/requrements.txt && python ./client/client.py```
