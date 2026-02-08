package speech

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

var (
	sinkProb     float32
	sinkSegments int
)

func BenchmarkInfer(b *testing.B) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	sd, err := NewDetector(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := sd.Destroy(); err != nil {
			b.Fatal(err)
		}
	}()

	samples := readSamplesFromFileB(b, "../testfiles/samples.pcm")
	windowSize := windowSizeForSampleRate(cfg.SampleRate)
	if len(samples) < windowSize {
		b.Fatalf("not enough samples")
	}

	b.ReportAllocs()
	b.ResetTimer()

	index := 0
	for i := 0; i < b.N; i++ {
		if index+windowSize > len(samples) {
			index = 0
			if err := sd.Reset(); err != nil {
				b.Fatal(err)
			}
		}
		prob, err := sd.Infer(samples[index : index+windowSize])
		if err != nil {
			b.Fatal(err)
		}
		sinkProb = prob
		index += windowSize
	}
}

func BenchmarkDetect(b *testing.B) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	sd, err := NewDetector(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := sd.Destroy(); err != nil {
			b.Fatal(err)
		}
	}()

	samples := readSamplesFromFileB(b, "../testfiles/samples.pcm")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		segments, err := sd.Detect(samples)
		if err != nil {
			b.Fatal(err)
		}
		if err := sd.Reset(); err != nil {
			b.Fatal(err)
		}
		sinkSegments = len(segments)
	}
}

func BenchmarkDetectStream(b *testing.B) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	sd, err := NewDetector(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := sd.Destroy(); err != nil {
			b.Fatal(err)
		}
	}()

	samples := readSamplesFromFileB(b, "../testfiles/samples.pcm")
	chunkSize := 1000

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := sd.Reset(); err != nil {
			b.Fatal(err)
		}
		total := 0
		for offset := 0; offset < len(samples); offset += chunkSize {
			end := offset + chunkSize
			if end > len(samples) {
				end = len(samples)
			}
			segments, err := sd.DetectStream(samples[offset:end])
			if err != nil {
				b.Fatal(err)
			}
			total += len(segments)
		}
		sinkSegments = total
	}
}

func readSamplesFromFileB(b *testing.B, path string) []float32 {
	b.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatal(err)
	}

	samples := make([]float32, 0, len(data)/4)
	for i := 0; i < len(data); i += 4 {
		samples = append(samples, math.Float32frombits(binary.LittleEndian.Uint32(data[i:i+4])))
	}
	return samples
}
