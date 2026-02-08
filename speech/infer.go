package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"unsafe"
)

func (sd *Detector) Infer(samples []float32) (float32, error) {
	if sd == nil {
		return 0, fmt.Errorf("invalid nil detector")
	}

	if sd.windowSize == 0 {
		sd.windowSize = windowSizeForSampleRate(sd.cfg.SampleRate)
	}
	if len(samples) != sd.windowSize {
		return 0, fmt.Errorf("invalid samples length: expected %d, got %d", sd.windowSize, len(samples))
	}

	expectedInputLen := contextLen + sd.windowSize
	if len(sd.inputBuf) != expectedInputLen {
		sd.inputBuf = make([]float32, expectedInputLen)
		sd.pcmInputDims = [2]C.int64_t{1, C.int64_t(expectedInputLen)}
	}
	if sd.stateDims[0] == 0 {
		sd.stateDims = [3]C.int64_t{2, 1, 128}
	}
	if sd.rateInputDims[0] == 0 {
		sd.rateInputDims = [1]C.int64_t{1}
	}
	if sd.rateValue == 0 {
		sd.rateValue = C.int64_t(sd.cfg.SampleRate)
	}

	copy(sd.inputBuf[contextLen:], samples)

	// Create tensors
	var pcmValue *C.OrtValue
	status := C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&sd.inputBuf[0]),
		C.size_t(len(sd.inputBuf)*4), &sd.pcmInputDims[0], C.size_t(len(sd.pcmInputDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pcmValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, pcmValue)

	var stateValue *C.OrtValue
	status = C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&sd.state[0]),
		C.size_t(stateLen*4), &sd.stateDims[0], C.size_t(len(sd.stateDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &stateValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, stateValue)

	var rateValue *C.OrtValue
	status = C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&sd.rateValue),
		C.size_t(unsafe.Sizeof(sd.rateValue)), &sd.rateInputDims[0], C.size_t(len(sd.rateInputDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &rateValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, rateValue)

	// Run inference
	inputs := []*C.OrtValue{pcmValue, stateValue, rateValue}
	outputs := []*C.OrtValue{nil, nil}

	inputNames := []*C.char{
		sd.cStrings["input"],
		sd.cStrings["state"],
		sd.cStrings["sr"],
	}
	outputNames := []*C.char{
		sd.cStrings["output"],
		sd.cStrings["stateN"],
	}
	status = C.OrtApiRun(sd.api, sd.session, nil, &inputNames[0], &inputs[0], C.size_t(len(inputNames)),
		&outputNames[0], C.size_t(len(outputNames)), &outputs[0])
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to run: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	// Get output values from tensor data
	var prob unsafe.Pointer
	var stateN unsafe.Pointer

	status = C.OrtApiGetTensorMutableData(sd.api, outputs[0], &prob)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiGetTensorMutableData(sd.api, outputs[1], &stateN)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	C.memcpy(unsafe.Pointer(&sd.state[0]), stateN, stateLen*4)

	C.OrtApiReleaseValue(sd.api, outputs[0])
	C.OrtApiReleaseValue(sd.api, outputs[1])

	copy(sd.inputBuf[:contextLen], sd.inputBuf[sd.windowSize:])

	// Return speech probability
	return *(*float32)(prob), nil
}
