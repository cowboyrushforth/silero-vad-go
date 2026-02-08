package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"

	"github.com/streamer45/silero-vad-go/speech"
)

func main() {
	var (
		modelPath  string
		pcmPath    string
		sampleRate int
		threshold  float64
		chunkSize  int
	)

	flag.StringVar(&modelPath, "model", "", "path to silero_vad.onnx")
	flag.StringVar(&pcmPath, "pcm", "", "path to float32 LE PCM file")
	flag.IntVar(&sampleRate, "rate", 16000, "sample rate (8000 or 16000)")
	flag.Float64Var(&threshold, "threshold", 0.5, "speech probability threshold")
	flag.IntVar(&chunkSize, "chunk", 1600, "chunk size in frames")
	flag.Parse()

	if modelPath == "" || pcmPath == "" {
		log.Fatal("both -model and -pcm are required")
	}

	cfg := speech.DetectorConfig{
		ModelPath:  modelPath,
		SampleRate: sampleRate,
		Threshold:  float32(threshold),
	}

	sd, err := speech.NewDetector(cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := sd.Destroy(); err != nil {
			log.Fatal(err)
		}
	}()

	file, err := os.Open(pcmPath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	bytesPerSample := 4
	buf := make([]byte, chunkSize*bytesPerSample)

	for {
		n, err := io.ReadFull(reader, buf)
		if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
			log.Fatal(err)
		}

		if n == 0 {
			break
		}

		if n%bytesPerSample != 0 {
			n -= n % bytesPerSample
		}
		samples := bytesToFloat32(buf[:n])

		segments, err := sd.DetectStream(samples)
		if err != nil {
			log.Fatal(err)
		}

		for _, seg := range segments {
			if seg.SpeechEndAt == 0 {
				fmt.Printf("speech start: %.3fs\n", seg.SpeechStartAt)
				continue
			}
			fmt.Printf("speech end: %.3fs (start %.3fs)\n", seg.SpeechEndAt, seg.SpeechStartAt)
		}

		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
	}
}

func bytesToFloat32(data []byte) []float32 {
	samples := make([]float32, 0, len(data)/4)
	for i := 0; i+4 <= len(data); i += 4 {
		samples = append(samples, math.Float32frombits(binary.LittleEndian.Uint32(data[i:i+4])))
	}
	return samples
}
