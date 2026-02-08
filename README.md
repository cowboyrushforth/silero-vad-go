<h1 align="center">
  <br>
  silero-vad-go
  <br>
</h1>
<h4 align="center">A simple Golang (CGO + ONNX Runtime) speech detector powered by Silero VAD</h4>
<p align="center">
  <a href="https://pkg.go.dev/github.com/streamer45/silero-vad-go"><img src="https://pkg.go.dev/badge/github.com/streamer45/silero-vad-go.svg" alt="Go Reference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>
<br>

### Requirements

- [Golang](https://go.dev/doc/install) >= v1.21
- A C compiler (e.g. GCC)
- ONNX Runtime (v1.18.1)
- A [Silero VAD](https://github.com/snakers4/silero-vad) model (v5)

### Usage

The detector accepts mono PCM samples as `[]float32` in little-endian format.

```go
cfg := speech.DetectorConfig{
  ModelPath:  "/path/to/silero_vad.onnx",
  SampleRate: 16000,
  Threshold:  0.5,
}

sd, err := speech.NewDetector(cfg)
if err != nil {
  log.Fatal(err)
}
defer sd.Destroy()

// Batch detection.
segments, err := sd.Detect(samples)
if err != nil {
  log.Fatal(err)
}
_ = segments

// Streaming detection.
// DetectStream emits start events (SpeechEndAt == 0) and end events.
for _, chunk := range chunks {
  updates, err := sd.DetectStream(chunk)
  if err != nil {
    log.Fatal(err)
  }
  for _, seg := range updates {
    fmt.Printf("start=%.3f end=%.3f\n", seg.SpeechStartAt, seg.SpeechEndAt)
  }
}
```

### Examples

- `examples/stream_file`: stream a PCM file from disk and run VAD on each chunk.
  ```sh
  go run ./examples/stream_file -model ./testfiles/silero_vad.onnx -pcm ./testfiles/samples.pcm
  ```

### Development

In order to build and/or run this library, you need to export (or pass) some env variables to point to the ONNX runtime files.

#### Linux

```sh
LD_RUN_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
LIBRARY_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
C_INCLUDE_PATH="/usr/local/include/onnxruntime-linux-x64-1.18.1/include"
```

#### Darwin (MacOS)

```sh
LIBRARY_PATH="/usr/local/lib/onnxruntime-linux-x64-1.18.1/lib"
C_INCLUDE_PATH="/usr/local/include/onnxruntime-linux-x64-1.18.1/include"
sudo update_dyld_shared_cache
```

### License

MIT License - see [LICENSE](LICENSE) for full text

