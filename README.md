# Video Analysis Pipeline

## Overview of `main.cc`

The video analysis pipeline consists of four main components, each running in its own thread:

1. **Read**: Reads frames from the video input
2. **FD (Face Detection)**: Determines appropriate bounding box coordinates for faces
3. **FR (Face Recognition)**: Recognizes the label of detected persons, draws bounding boxes and labels on the frame, and stores it in the display queue
4. **Display**: Responsible for displaying the result frame

The `main` function orchestrates these components, leveraging 4 threads to run them concurrently:

- Read Thread
- Face Detection (FD) Thread
- Face Recognition (FR) Thread
- Display Thread

This multi-threaded approach allows for efficient processing and real-time analysis of video frames.

# Performance Metrics

## Read Function
| Metric | Average Time |
|--------|--------------|
| Queue Check | 0.47585 µs |
| Read | 223228 µs |
| Mutex Lock | 26.0273 µs |
| Queue Push | 7.46446 µs |
| Mutex Unlock | 11.6281 µs |
| **Total Latency** | **223.28 ms** |

## Face Detection
| Metric | Average Time |
|--------|--------------|
| Queue | 41.64 µs |
| Resize | 3670.93 µs |
| Preprocess | 27816.73 µs |
| DPU Execution | 4087.88 µs |
| Postprocess | 40576.43 µs |
| Store Results | 1561.88 µs |
| **Total Latency** | **77.4389 ms** |

## Face Recognition
| Metric | Average Time |
|--------|--------------|
| Preprocess | 3092.98 µs |
| DPU Execution | 10091.48 µs |
| Postprocess | 1226.02 µs |
| Drawing | 199.84 µs |
| Wait for Result | 208333.38 µs |
| Crop Faces | 803.72 µs |
| Process Faces | 14892.62 µs |
| Add to Display Queue | 9.47 µs |
| **Total Latency** | **224.18 ms** |

## Display Function
| Metric | Average Time |
|--------|--------------|
| Queue Check | 2.96026 µs |
| Frame Processing | 151.60206 µs |
| ImShow | 2882.30639 µs |
| WaitKey | 616515.64 µs |
| **Total Latency** | **619.55 ms** |

**Average FPS**: 2.97334 (Note: This average may not be representative due to outliers)
