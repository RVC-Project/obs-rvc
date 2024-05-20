# Retrieval-based Voice Conversion OBS plugin

The voice changer you love, directly inside OBS via audio filters. No more Python scripts or virtual cables hassle.

Working in progress...

## Current State

The plugin works, with approximately the same performance as the PyTorch implementation. 

However, I need help on the following topics:

I thought TensorRT would work, but it doesn't. It seems the ContentVec / RMVPE / model contains several
operations that would force ONNXRuntime to do them in the CPU. At this point, I have no idea what to do next,
and I don't know whether I should continuously use ONNX or not. After all, the DLL dependency of ONNXRuntime
is not versioned correctly and it is a PITA combining it with other ML related plugins.

I might be better off using [huggingface/candle](https://github.com/huggingface/candle) or Burn or Luminal, both performance-wise
and ease of use. The ONNX is not delivering the performance I was looking for.

On the other hand, there are still a few bugs in this implementation. The voice is not fluent enough compared to the 
PyTorch + WASAPI implementation. I have done everything I can do, but it cannot be eliminated entirely.
Also, the RMS mix is not working as expected. As long as I enable that, some words and phrases will be cut off, and
when I give a rate like 0.5, the volume will go to a weird level. However, the RMS mix function is mathematically correct
compared to Python side. 

