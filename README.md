# lr4
Лабораторная работа #4<br />
Использование техники Transfer Learning<br />
Цель лабораторной работы : Обучить сверточную нейронную сеть с использованием
техники Transfer Learning для восстановления информации в цветовых каналах
изображения<br />
Задачи:<br />
1. Используя результаты лабораторных работ #1,2,3 выполнить:<br />
a. Обучить нейронную на 50к изображениях с использованием MobileNet V2,
предварительно обученной на imagenet, в качестве сети для выделения
признаков<br />
<br /><br />
Training: 67 117 фотографий случайно взятых из соцсетей https://www.kaggle.com/greg115/various-tagged-images<br />
Validation - 852 фотографий из личного архива <br />
<br />
Результат работы:
![alt text](https://github.com/NikitaNechaev1/lr4/blob/main/lr4-results/lr4_graph.png)
![alt text](https://github.com/NikitaNechaev1/lr4/blob/main/lr4-results/lr4_results.png)<br />

Лог:
2021-01-04 12:44:10.965189: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll<br />
2021-01-04 12:44:12.531661: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the follow<br />
ing CPU instructions in performance-critical operations:  AVX2<br />
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.<br />
2021-01-04 12:44:12.546034: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1e12ed561e0 initialized for platform Host (this does not guarantee that XLA will be used).<br />
Devices:<br />
2021-01-04 12:44:12.549170: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version<br />
2021-01-04 12:44:12.552349: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library nvcuda.dll<br />
2021-01-04 12:44:12.587629: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:<br />
pciBusID: 0000:07:00.0 name: GeForce RTX 2070 computeCapability: 7.5<br />
coreClock: 1.83GHz coreCount: 36 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 417.29GiB/s<br />
2021-01-04 12:44:12.592704: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll<br />
2021-01-04 12:44:12.600345: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll<br />
2021-01-04 12:44:12.602924: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll<br />
2021-01-04 12:44:12.608799: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll<br />
2021-01-04 12:44:12.612337: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll<br />
2021-01-04 12:44:12.621209: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll<br />
2021-01-04 12:44:12.625514: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll<br />
2021-01-04 12:44:12.628386: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll<br />
2021-01-04 12:44:12.630780: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0<br />
2021-01-04 12:44:13.242044: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:<br />
2021-01-04 12:44:13.244966: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0<br />
2021-01-04 12:44:13.246866: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N<br />
2021-01-04 12:44:13.248649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6226 MB memory) -> ph<br />
ysical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:07:00.0, compute capability: 7.5)<br />
2021-01-04 12:44:13.256862: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1e152923800 initialized for platform CUDA (this does not guarantee that XLA will be used).<br />
Devices:<br />
2021-01-04 12:44:13.260448: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce RTX 2070, Compute Capability 7.5<br />
2021-01-04 12:44:13.264212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:<br />
pciBusID: 0000:07:00.0 name: GeForce RTX 2070 computeCapability: 7.5<br />
coreClock: 1.83GHz coreCount: 36 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 417.29GiB/s<br />
2021-01-04 12:44:13.270191: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll<br />
2021-01-04 12:44:13.273073: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll<br />
2021-01-04 12:44:13.276081: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll<br />
2021-01-04 12:44:13.278754: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll<br />
2021-01-04 12:44:13.281186: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll<br />
2021-01-04 12:44:13.283653: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll<br />
2021-01-04 12:44:13.286238: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll<br />
2021-01-04 12:44:13.289287: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll<br />
2021-01-04 12:44:13.291810: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0<br />
2021-01-04 12:44:13.294260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:<br />
pciBusID: 0000:07:00.0 name: GeForce RTX 2070 computeCapability: 7.5<br />
coreClock: 1.83GHz coreCount: 36 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 417.29GiB/s<br />
2021-01-04 12:44:13.299234: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll<br />
2021-01-04 12:44:13.302340: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll<br />
2021-01-04 12:44:13.305383: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll<br />
2021-01-04 12:44:13.307856: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll<br />
2021-01-04 12:44:13.310343: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll<br />
2021-01-04 12:44:13.312939: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll<br />
2021-01-04 12:44:13.315698: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll<br />
2021-01-04 12:44:13.318170: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll<br />
2021-01-04 12:44:13.320577: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0<br />
2021-01-04 12:44:13.322580: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:<br />
2021-01-04 12:44:13.325791: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0<br />
2021-01-04 12:44:13.327440: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N<br />
2021-01-04 12:44:13.329203: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6226 MB memory) -> ph<br />
ysical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:07:00.0, compute capability: 7.5)<br />
2021-01-04 12:44:13.395834: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)<br />
Count of train images: 67117<br />
Count of validation images: 852<br />
Model: "model"<br />
_________________________________________________________________<br />
Layer (type)                 Output Shape              Param #<br />
=================================================================<br />
input_2 (InputLayer)         [(None, 224, 224, 1)]     0<br />
_________________________________________________________________<br />
tf.image.grayscale_to_rgb (T (None, 224, 224, 3)       0<br />
_________________________________________________________________<br />
mobilenetv2_1.00_224 (Functi (None, 7, 7, 1280)        2257984<br />
_________________________________________________________________<br />
conv2d_transpose (Conv2DTran (None, 28, 28, 128)       4096128<br />
_________________________________________________________________<br />
conv2d_transpose_1 (Conv2DTr (None, 56, 56, 128)       147584<br />
_________________________________________________________________<br />
conv2d_transpose_2 (Conv2DTr (None, 112, 112, 64)      73792<br />
_________________________________________________________________<br />
conv2d_transpose_3 (Conv2DTr (None, 224, 224, 2)       1154<br />
=================================================================<br />
Total params: 6,576,642<br />
Trainable params: 4,318,658<br />
Non-trainable params: 2,257,984<br />
_________________________________________________________________<br />
None<br />
2021-01-04 12:44:28.320060: I tensorflow/core/profiler/lib/profiler_session.cc:136] Profiler session initializing.<br />
2021-01-04 12:44:28.322279: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.<br />
2021-01-04 12:44:28.324168: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1365] Profiler found 1 GPUs<br />
2021-01-04 12:44:28.327108: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cupti64_110.dll<br />
2021-01-04 12:44:28.398156: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.<br />
2021-01-04 12:44:28.400293: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1487] CUPTI activity buffer flushed<br />
Epoch 1/50<br />
2021-01-04 12:44:31.209078: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll<br />
2021-01-04 12:44:32.131456: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0<br />
2021-01-04 12:44:32.176607: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0<br />

2021-01-04 12:44:32.207792: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll<br />
2021-01-04 12:44:32.648096: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll<br />
      1/Unknown - 6s 6s/step - loss: 0.25152021-01-04 12:44:35.011056: I tensorflow/core/profiler/lib/profiler_session.cc:136] Profiler session initializing.<br />
2021-01-04 12:44:35.014054: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.<br />
      2/Unknown - 7s 1s/step - loss: 0.24672021-01-04 12:44:35.530641: I tensorflow/core/profiler/lib/profiler_session.cc:71] Profiler session collecting data.<br />
2021-01-04 12:44:35.533967: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1487] CUPTI activity buffer flushed<br />
2021-01-04 12:44:35.735137: I tensorflow/core/profiler/internal/gpu/cupti_collector.cc:228]  GpuTracer has collected 536 callback api events and 424 activity events.<br />
2021-01-04 12:44:35.748512: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.<br />
2021-01-04 12:44:35.820685: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_0<br />
4_09_44_35<br />
2021-01-04 12:44:35.872462: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for trace.json.gz to E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35\DESKTOP-67BS2ML.trace.json.gz<br />
2021-01-04 12:44:35.917077: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35<br />
2021-01-04 12:44:35.951277: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for memory_profile.json.gz to E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35\DESKTOP-67BS2ML.memory_profile.json.gz<br />
2021-01-04 12:44:36.105923: I tensorflow/core/profiler/rpc/client/capture_profile.cc:251] Creating directory: E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35Dumped tool data for xplane.pb to E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35\DESKTOP-67BS2ML.xplane.pb<br />
Dumped tool data for overview_page.pb to E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35\DESKTOP-67BS2ML.overview_page.pb<br />
Dumped tool data for input_pipeline.pb to E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35\DESKTOP-67BS2ML.input_pipeline.pb<br />
Dumped tool data for tensorflow_stats.pb to E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35\DESKTOP-67BS2ML.tensorflow_stats.pb<br />
Dumped tool data for kernel_stats.pb to E:\Nicky\soroka_lab/log_lab3/20210104-124413\train\plugins\profile\2021_01_04_09_44_35\DESKTOP-67BS2ML.kernel_stats.pb<br />

1000/1000 [==============================] - 366s 360ms/step - loss: 0.0852 - val_loss: 0.0800<br />
Epoch 2/50<br />
1000/1000 [==============================] - 353s 354ms/step - loss: 0.0614 - val_loss: 0.0738<br />
Epoch 3/50<br />
1000/1000 [==============================] - 354s 354ms/step - loss: 0.0567 - val_loss: 0.0686<br />
Epoch 4/50<br />
1000/1000 [==============================] - 355s 355ms/step - loss: 0.0551 - val_loss: 0.0673<br />
Epoch 5/50<br />
1000/1000 [==============================] - 354s 354ms/step - loss: 0.0541 - val_loss: 0.0670<br />
Epoch 6/50<br />
1000/1000 [==============================] - 354s 354ms/step - loss: 0.0536 - val_loss: 0.0667<br />
Epoch 7/50<br />
1000/1000 [==============================] - 355s 355ms/step - loss: 0.0533 - val_loss: 0.0660<br />
Epoch 8/50<br />
1000/1000 [==============================] - 353s 354ms/step - loss: 0.0524 - val_loss: 0.0671<br />
Epoch 9/50<br />
1000/1000 [==============================] - 354s 354ms/step - loss: 0.0525 - val_loss: 0.0659<br />
Epoch 10/50<br />
1000/1000 [==============================] - 354s 354ms/step - loss: 0.0519 - val_loss: 0.0656<br />
Epoch 11/50<br />
1000/1000 [==============================] - 355s 355ms/step - loss: 0.0518 - val_loss: 0.0659<br />
Epoch 12/50<br />
1000/1000 [==============================] - 358s 358ms/step - loss: 0.0517 - val_loss: 0.0656<br />
Epoch 13/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0516 - val_loss: 0.0662<br />
Epoch 14/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0512 - val_loss: 0.0653<br />
Epoch 15/50<br />
1000/1000 [==============================] - 357s 358ms/step - loss: 0.0509 - val_loss: 0.0655<br />
Epoch 16/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0508 - val_loss: 0.0652<br />
Epoch 17/50<br />
1000/1000 [==============================] - 358s 358ms/step - loss: 0.0505 - val_loss: 0.0656<br />
Epoch 18/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0507 - val_loss: 0.0652<br />
Epoch 19/50<br />
1000/1000 [==============================] - 358s 358ms/step - loss: 0.0504 - val_loss: 0.0650<br />
Epoch 20/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0502 - val_loss: 0.0650<br />
Epoch 21/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0503 - val_loss: 0.0648<br />
Epoch 22/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0501 - val_loss: 0.0650<br />
Epoch 23/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0501 - val_loss: 0.0652<br />
Epoch 24/50<br />
1000/1000 [==============================] - 356s 356ms/step - loss: 0.0496 - val_loss: 0.0652<br />
Epoch 25/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0498 - val_loss: 0.0646<br />
Epoch 26/50<br />
1000/1000 [==============================] - 358s 358ms/step - loss: 0.0498 - val_loss: 0.0651<br />
Epoch 27/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0499 - val_loss: 0.0649<br />
Epoch 28/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0496 - val_loss: 0.0649<br />
Epoch 29/50<br />
1000/1000 [==============================] - 358s 358ms/step - loss: 0.0496 - val_loss: 0.0647<br />
Epoch 30/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0494 - val_loss: 0.0645<br />
Epoch 31/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0491 - val_loss: 0.0647<br />
Epoch 32/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0493 - val_loss: 0.0647<br />
Epoch 33/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0491 - val_loss: 0.0648<br />
Epoch 34/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0493 - val_loss: 0.0644<br />
Epoch 35/50<br />
1000/1000 [==============================] - 358s 358ms/step - loss: 0.0491 - val_loss: 0.0647<br />
Epoch 36/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0491 - val_loss: 0.0644<br />
Epoch 37/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0491 - val_loss: 0.0655<br />
Epoch 38/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0489 - val_loss: 0.0649<br />
Epoch 39/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0490 - val_loss: 0.0646<br />
Epoch 40/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0491 - val_loss: 0.0641<br />
Epoch 41/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0489 - val_loss: 0.0643<br />
Epoch 42/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0491 - val_loss: 0.0646<br />
Epoch 43/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0488 - val_loss: 0.0642<br />
Epoch 44/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0487 - val_loss: 0.0654<br />
Epoch 45/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0485 - val_loss: 0.0639<br />
Epoch 46/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0485 - val_loss: 0.0640<br />
Epoch 47/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0486 - val_loss: 0.0642<br />
Epoch 48/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0486 - val_loss: 0.0643<br />
Epoch 49/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0485 - val_loss: 0.0636<br />
Epoch 50/50<br />
1000/1000 [==============================] - 357s 357ms/step - loss: 0.0485 - val_loss: 0.0637<br />
