/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//rev1

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <numeric>
#include <deque>
#include <condition_variable>
#include <cnpy.h>


// Header file OpenCV for image processing
#include <opencv2/opencv.hpp>

#include "common.h"

using namespace std;
using namespace cv;

const int TNUM = 4; //쓰래드 돌릴 갯수, FD하나만 할 때는 1 (원래는 6개라 6이었음)

//For calculating FPS
const int FPS_QUEUE_SIZE = 30;  // Number of frames to average for FPS calculation
std::deque<double> fpsQueue;
std::chrono::time_point<std::chrono::high_resolution_clock> lastFrameTime;

// input video
VideoCapture video;

// flags for each thread
bool is_reading = true;
array<bool, TNUM> is_running;
bool is_displaying = true;

// comparison algorithm for priority_queue
class Compare {
 public:
  bool operator()(const pair<int, Mat>& n1, const pair<int, Mat>& n2) const {
    return n1.first > n2.first;
  }
};

queue<pair<int, Mat>> read_queue;  // read queue
priority_queue<pair<int, Mat>, vector<pair<int, Mat>>, Compare>
    display_queue;        // display queue
mutex mtx_read_queue;     // mutex of read queue
mutex mtx_display_queue;  // mutex of display queue
int read_index = 0;       // frame index of input video
int display_index = 0;    // frame index to display

GraphInfo shapes;

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
//비디오를 읽어서 frame을 저장하는 함수
void Read(bool& is_reading) {
    std::chrono::high_resolution_clock::time_point start_time, end_time, frame_start_time;
    std::chrono::duration<double, std::micro> duration;
    
    double total_read_time = 0.0;
    double total_queue_check_time = 0.0;
    double total_queue_push_time = 0.0;
    double total_mutex_lock_time = 0.0;
    double total_mutex_unlock_time = 0.0;
    double total_sleep_time = 0.0;
    int frame_count = 0;
    int sleep_count = 0;

    while (is_reading) {
        frame_start_time = std::chrono::high_resolution_clock::now();
        double frame_queue_check_time = 0.0;
        double frame_read_time = 0.0;
        double frame_mutex_lock_time = 0.0;
        double frame_queue_push_time = 0.0;
        double frame_mutex_unlock_time = 0.0;
        double frame_sleep_time = 0.0;

        Mat img;

        // Measure queue size check time
        start_time = std::chrono::high_resolution_clock::now();
        bool queue_not_full = (read_queue.size() < 30);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
        frame_queue_check_time = duration.count();
        total_queue_check_time += frame_queue_check_time;

        if (queue_not_full) {
            // Measure video read time
            start_time = std::chrono::high_resolution_clock::now();
            bool read_success = video.read(img);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
            frame_read_time = duration.count();
            total_read_time += frame_read_time;

            if (!read_success) {
                cout << "Video end." << endl;
                is_reading = false;
                break;
            }

            // Measure mutex lock time
            start_time = std::chrono::high_resolution_clock::now();
            mtx_read_queue.lock();
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
            frame_mutex_lock_time = duration.count();
            total_mutex_lock_time += frame_mutex_lock_time;

            // Measure queue push time
            start_time = std::chrono::high_resolution_clock::now();
            read_queue.push(make_pair(read_index++, img));
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
            frame_queue_push_time = duration.count();
            total_queue_push_time += frame_queue_push_time;

            // Measure mutex unlock time
            start_time = std::chrono::high_resolution_clock::now();
            mtx_read_queue.unlock();
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
            frame_mutex_unlock_time = duration.count();
            total_mutex_unlock_time += frame_mutex_unlock_time;

            frame_count++;

            // Print timing information for this frame
            auto frame_end_time = std::chrono::high_resolution_clock::now();
            auto frame_total_duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(frame_end_time - frame_start_time);
            cout << "Read Frame " << frame_count << " timings:" << endl;
            cout << "  Queue check: " << frame_queue_check_time << " µs" << endl;
            cout << "  Read: " << frame_read_time << " µs" << endl;
            cout << "  Mutex lock: " << frame_mutex_lock_time << " µs" << endl;
            cout << "  Queue push: " << frame_queue_push_time << " µs" << endl;
            cout << "  Mutex unlock: " << frame_mutex_unlock_time << " µs" << endl;
            cout << "  Total frame time: " << frame_total_duration.count() << " µs" << endl;
        } else {
            // Measure sleep time
            start_time = std::chrono::high_resolution_clock::now();
            usleep(20);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
            frame_sleep_time = duration.count();
            total_sleep_time += frame_sleep_time;
            sleep_count++;

            // Print sleep information
            cout << "Sleep duration: " << frame_sleep_time << " µs" << endl;
        }
    }

    // Print final statistics
    cout << "Read function final statistics:" << endl;
    cout << "  Total frames processed: " << frame_count << endl;
    cout << "  Average queue check time: " << total_queue_check_time / frame_count << " µs" << endl;
    cout << "  Average read time: " << total_read_time / frame_count << " µs" << endl;
    cout << "  Average mutex lock time: " << total_mutex_lock_time / frame_count << " µs" << endl;
    cout << "  Average queue push time: " << total_queue_push_time / frame_count << " µs" << endl;
    cout << "  Average mutex unlock time: " << total_mutex_unlock_time / frame_count << " µs" << endl;
    cout << "  Average sleep time: " << (sleep_count > 0 ? total_sleep_time / sleep_count : 0) << " µs" << endl;
    cout << "  Total sleep count: " << sleep_count << endl;
}

/**
 * @brief Display frames in display queue
 *
 * @param is_displaying - status flag of Display thread
 *
 * @return none
 */
void Display(bool& is_displaying) {
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double, std::micro> duration;
    
    double total_window_creation_time = 0.0;
    double total_queue_check_time = 0.0;
    double total_frame_processing_time = 0.0;
    double total_imshow_time = 0.0;
    double total_waitkey_time = 0.0;
    int frame_count = 0;

    // Measure window creation time
    start_time = std::chrono::high_resolution_clock::now();
    cv::namedWindow("Video Analysis", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Analysis", 1280, 720);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
    total_window_creation_time = duration.count();

    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    while (is_displaying) {
        start_time = std::chrono::high_resolution_clock::now();
        mtx_display_queue.lock();
        bool queue_empty = display_queue.empty();
        mtx_display_queue.unlock();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
        total_queue_check_time += duration.count();
        
        if (!queue_empty) {
            start_time = std::chrono::high_resolution_clock::now();
            mtx_display_queue.lock();
            auto current_frame = display_queue.top();
            display_queue.pop();
            mtx_display_queue.unlock();

            auto currentTime = std::chrono::high_resolution_clock::now();
            double fps = 1.0 / std::chrono::duration<double>(currentTime - lastFrameTime).count();
            lastFrameTime = currentTime;

            cv::putText(current_frame.second, 
                        "FPS: " + std::to_string(int(fps)), 
                        cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 
                        1, 
                        cv::Scalar(0, 255, 0), 
                        2);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
            total_frame_processing_time += duration.count();

            start_time = std::chrono::high_resolution_clock::now();
            cv::imshow("Video Analysis", current_frame.second);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
            total_imshow_time += duration.count();
            
            start_time = std::chrono::high_resolution_clock::now();
            int key = cv::waitKey(30);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end_time - start_time);
            total_waitkey_time += duration.count();

            if (key == 'q' || key == 27) {  // 'q' or ESC key
                is_displaying = false;
                is_reading = false;
                for (auto& running : is_running) {
                    running = false;
                }
                break;
            }

            frame_count++;

            // Print timing information for this frame
            cout << "Display Frame " << frame_count << " timings:" << endl;
            cout << "  Queue check: " << total_queue_check_time / frame_count << " µs" << endl;
            cout << "  Frame processing: " << total_frame_processing_time / frame_count << " µs" << endl;
            cout << "  ImShow: " << total_imshow_time / frame_count << " µs" << endl;
            cout << "  WaitKey: " << total_waitkey_time / frame_count << " µs" << endl;
            cout << "  FPS: " << fps << endl;
        } else {
            if (std::any_of(is_running.begin(), is_running.end(), [](bool v) { return v; }) || is_reading) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                is_displaying = false;
                break;
            }
        }
    }

    // Print final statistics
    cout << "Display function final statistics:" << endl;
    cout << "  Window creation time: " << total_window_creation_time << " µs" << endl;
    cout << "  Total frames displayed: " << frame_count << endl;
    cout << "  Average queue check time: " << total_queue_check_time / frame_count << " µs" << endl;
    cout << "  Average frame processing time: " << total_frame_processing_time / frame_count << " µs" << endl;
    cout << "  Average ImShow time: " << total_imshow_time / frame_count << " µs" << endl;
    cout << "  Average WaitKey time: " << total_waitkey_time / frame_count << " µs" << endl;

    cv::destroyAllWindows();
}

//runFacedetect 구조체 저장
struct FaceDetectResult {
    struct BoundingBox {
        float x, y, width, height, score;
    };
    int width, height;
    std::vector<BoundingBox> rects;
};

struct SharedDetectionResults {
    std::vector<FaceDetectResult> results;
    cv::Mat frame;
    int frame_index;
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    bool finished = false;
};

SharedDetectionResults sharedResults;

void NormalizeInputData(const unsigned char* input, int rows, int cols, int channels,
                        int stride, const std::vector<float>& mean,
                        const std::vector<float>& scale, int8_t* data) {
  for (int h = 0; h < rows; ++h) {
    for (int w = 0; w < cols; ++w) {
      for (int c = 0; c < channels; ++c) {
        //각 pixel에 대해 normalization 수행 (평균빼기, scale 곱하기)
        float value = (input[h * stride + w * channels + c] * 1.0f - mean[c]) * scale[c];
        //결과값을 반올림하고, -128~127사이로 clipping함
        int rounded_value = std::round(value);
        rounded_value = std::max(-128, std::min(127, rounded_value));
        //최종 값을 int8_t로 변환해서 data에 저장//즉 InputData배열의 특정 위치에 저장
        data[h * cols * channels + w * channels + c] = (int8_t)rounded_value;
      }
    }
  }
}

void setImageBGR(const std::vector<cv::Mat>& imgs, 
                 vart::Runner* runner,
                 int8_t* inputData,
                 const std::vector<float>& mean,
                 const std::vector<float>& scale) {
    auto inputTensors = runner->get_input_tensors();   //DPU runner로부터 입력 tensor 정보 가져오기
    CHECK_GT(inputTensors.size(), 0u);  //입력 tensor가 최소 하나 이상 있는지 확인
    
    //입력 tensor shape 가져온 후 저장
    auto in_dims = inputTensors[0]->get_shape();
    int batchSize = in_dims[0];
    int height = in_dims[1];
    int width = in_dims[2];
    int channels = in_dims[3];
    
    CHECK_EQ(imgs.size(), batchSize); //입력 layer 수와 batch 크기가 일치하는지 확인
    CHECK_LE(imgs.size(), batchSize); //입력 layer 수가 최대 입력 batch 크기 초과하지 않는지 확인
    
    float input_scale = get_input_scale(inputTensors[0]);   //input scale가져오기
    //RGB channel별 실제 scale 계산
    std::vector<float> real_scale{scale[0] * input_scale,
                                  scale[1] * input_scale,
                                  scale[2] * input_scale};

    //각 입력 이미지에 대해 처리
    for (size_t i = 0; i < imgs.size(); i++) {
        //batchData는 inputData배열의 특정 위치를 가리키는 포인터
        int8_t* batchData = inputData + i * height * width * channels; 
        
        NormalizeInputData(imgs[i].data, height, width, channels, imgs[i].step,
                           mean, real_scale, batchData);
    }
}

// Softmax function implementation
// cls = classes
void softmax(const int8_t* input, float scale, unsigned int cls, unsigned int group, float* output) {
  for (unsigned int i = 0; i < group; ++i) {
    float sum = 0.f;
    for (unsigned int j = 0; j < cls; ++j) {
      output[j] = exp(input[j] * scale);
      sum += output[j];
    }
    for (unsigned int j = 0; j < cls; ++j) {
      output[j] /= sum;
    }
    input += cls;
    output += cls;
  }
}

// FilterBox function implementation
// bb_out_scale : boundingbox 출력 scale(fixed_point -> floating point변환에 사용)
// bbout : boundingbox 좌표 data(int8 type 배열)
// w, h, : 출력 feature map의 width, height
// pred : 각 위치의 신뢰도 점수 (float type의 배열)
std::vector<std::vector<float>> FilterBox(const float bb_out_scale, const float det_threshold, int8_t* bbout, int w, int h, float* pred) {
  std::vector<std::vector<float>> boxes;
  //출력 feature map의 각 위치(i,j)에 대해서 반복
  //각 위치에서 1. 해당 위치의 신뢰도 점수를 확인(pred[position*2+1])
  //2. 신뢰도 점수가 임계값(det_threshold)보다 높으면 bounding box 생성
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int position = i * w + j;
      if (pred[position * 2 + 1] > det_threshold) {
        std::vector<float> box;
        //bounding box 좌표 계산
        // 모델 출력값에 scale을 곱하고, 격자 위치(i,j)를 더함
        //신뢰도 점수를 box 정보에 추가
        box.push_back(bbout[position * 4 + 0] * bb_out_scale + j * 4);
        box.push_back(bbout[position * 4 + 1] * bb_out_scale + i * 4);
        box.push_back(bbout[position * 4 + 2] * bb_out_scale + j * 4);
        box.push_back(bbout[position * 4 + 3] * bb_out_scale + i * 4);
        box.push_back(pred[position * 2 + 1]);
        boxes.push_back(box);
      }
    }
  }
  //임계값을 넘는 모든 유효한 bounding box list 반환
  // 각 box는 [x1, y1, x2, y2, score]형태의 vector로 표현됨
  return boxes; 
}

static float overlap(float x1, float x2, float x3, float x4) {
    float left = std::max(x1, x3);
    float right = std::min(x2, x4);
    return std::max(right - left, 0.0f);
}

static float box_c(const std::vector<float>& box1, const std::vector<float>& box2) {
    float center_x1 = (box1[0] + box1[2]) / 2;
    float center_y1 = (box1[1] + box1[3]) / 2;
    float center_x2 = (box2[0] + box2[2]) / 2;
    float center_y2 = (box2[1] + box2[3]) / 2;
    float dx = center_x1 - center_x2;
    float dy = center_y1 - center_y2;
    return dx * dx + dy * dy;
}

static float cal_iou(const std::vector<float>& box1, const std::vector<float>& box2) {
    // box1, box2: [x1, y1, x2, y2] 형식
    float w = overlap(box1[0], box1[2], box2[0], box2[2]);
    float h = overlap(box1[1], box1[3], box2[1], box2[3]);

    float inter_area = w * h;
    float box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = box1_area + box2_area - inter_area;
    
    float iou = inter_area / union_area;

    // 중심점 간 거리에 기반한 페널티 항 계산
    float c = box_c(box1, box2);
    if (c == 0) return iou;

    float d = (box1[0] - box2[0]) * (box1[0] - box2[0]) +
              (box1[1] - box2[1]) * (box1[1] - box2[1]);
    float u = std::pow(d / c, 0.6f);

    return std::max(iou - u, 0.0f);  // IoU가 음수가 되지 않도록 보장
}

// ApplyNMS function implementation
void applyNMS(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores,
              const float nms_threshold, const float conf_threshold, std::vector<size_t>& res,
              int max_output) { //boxes는 2차원 백터
  const size_t count = boxes.size();  //전체 box의 수
  //각 box의 점수와 index를 pair로 저장하는 vector (score, size)순서
  std::vector<std::pair<float, size_t>> order; 
  for (size_t i = 0; i < count; ++i) {
    // 박스 크기 계산 (너비 * 높이)
    float box_size = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
    order.push_back({box_size, i}); //order변수에 box_size넣음
  }
  std::sort(order.begin(), order.end(), //order 변수에 있는 box크기 순으로 내림차순 정렬 
            [](const std::pair<float, size_t>& ls, const std::pair<float, size_t>& rs) {
              return ls.first > rs.first;
            });
            
  std::vector<bool> exist_box(count, true); //exist_box선언, 모든 박스가 처음엔 존재한다고 표시

  for (size_t _i = 0; _i < count && res.size() < max_output; ++_i) { //정렬된 순서대로 각 box에 대해
    size_t i = order[_i].second;
    if (!exist_box[i]) continue;  //이미 false로 설정된 boxs는 건너 뜀
    if (scores[i] < conf_threshold) { //점수가 conf_threshold값 미만이면 false로 바꿈
      exist_box[i] = false;
      continue;
    }
    res.push_back(i); //현재 박스를 'res' list에 추가
    for (size_t _j = _i + 1; _j < count; ++_j) {
      size_t j = order[_j].second;
      if (!exist_box[j]) continue;
      float ovr = cal_iou(boxes[j], boxes[i]);  //두 box간의 IoU를 계산
      if (ovr >= nms_threshold) exist_box[j] = false; //IoU가 NMS 임계값 이상이면 해당 box 억제
    }
    //결과적으로 NMS과정을 통과한 box들의 index list만 남음
  }
}

void RunFaceDetect(vart::Runner* runner, bool& is_running) {
  std::cout << "Face Detection Thread Started\n";

  // Get tensors
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();

  // Get scales
  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);

  // Set up sizes
  int outSize = out_dims[1] * out_dims[2] * out_dims[3];
  int inSize = in_dims[1] * in_dims[2] * in_dims[3];
  int inHeight = in_dims[1];
  int inWidth = in_dims[2];
  int batchSize = in_dims[0];

  // Allocate memory for input images and results
  int8_t* imageInputs = new int8_t[inSize * batchSize];
  int8_t* FCResult = new int8_t[batchSize * outSize];

  // Allocate memory for prediction scores
  float* pred = new (std::nothrow) float[out_dims[1] * out_dims[2] * 2];
  if (!pred) {
    std::cerr << "Failed to allocate memory for pred" << std::endl;
    delete[] imageInputs;
    delete[] FCResult;
    return;
  }

  // Set up mean and scale values for image preprocessing
  std::vector<float> mean = {128.0f, 128.0f, 128.0f};
  std::vector<float> scale = {1.0f, 1.0f, 1.0f};

  // Timing variables
  auto total_start_time = std::chrono::high_resolution_clock::now();
  int frame_count = 0;
  double total_latency = 0.0;
  
  // Main processing loop
  while (is_running) {
    auto frame_start_time = std::chrono::high_resolution_clock::now();
    
    // Get an image from read queue
    int index;
    cv::Mat img;
    
    auto queue_start = std::chrono::high_resolution_clock::now();
    mtx_read_queue.lock();
    if (read_queue.empty()) {
      mtx_read_queue.unlock();
      if (is_reading) {
        continue;
      } else {
        is_running = false;
        break;
      }
    } else {
      index = read_queue.front().first;
      img = read_queue.front().second;
      read_queue.pop();
      mtx_read_queue.unlock();
    }
    auto queue_end = std::chrono::high_resolution_clock::now();
    auto queue_duration = std::chrono::duration_cast<std::chrono::microseconds>(queue_end - queue_start);

    // Resize image to match model input dimensions
    auto resize_start = std::chrono::high_resolution_clock::now();
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(inWidth, inHeight));
    auto resize_end = std::chrono::high_resolution_clock::now();
    auto resize_duration = std::chrono::duration_cast<std::chrono::microseconds>(resize_end - resize_start);

    // Prepare input data (Preprocessing)
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    setImageBGR({resized}, runner, imageInputs, mean, scale);
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start);

    // Set up input and output tensors for model execution
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(imageInputs, inputTensors[0]));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(FCResult, outputTensors[0]));
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    // Run DPU
    auto dpu_start = std::chrono::high_resolution_clock::now();
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    auto status = runner->wait(job_id.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";
    auto dpu_end = std::chrono::high_resolution_clock::now();
    auto dpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpu_end - dpu_start);

    // Set detection and NMS thresholds
    const float det_threshold = 0.9f;
    const float nms_threshold = 0.3f;
    
    // Calculate prediction scores
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < out_dims[1] * out_dims[2]; ++i) {
      pred[i * 2] = 1.0f - (FCResult[i * 2] * output_scale);  // background score
      pred[i * 2 + 1] = FCResult[i * 2 + 1] * output_scale;   // face score
    }    

    // Apply filtering to get potential face bounding boxes
    std::vector<std::vector<float>> boxes = FilterBox(
        output_scale,  // bb_out_scale
        det_threshold,
        FCResult,      // bbout
        out_dims[2],   // width
        out_dims[1],   // height
        pred           // pred
    );

    // Extract scores from boxes
    std::vector<float> scores;
    for (auto& box : boxes) {
      scores.push_back(box[4]);
    }

    // Apply Non-Maximum Suppression (NMS)
    std::vector<size_t> res_k;
    int max_faces = 1; // Adjust as needed
    applyNMS(boxes, scores, nms_threshold, det_threshold, res_k, max_faces);

    // Create FaceDetectResult structure with detected face rectangles
    FaceDetectResult result{inWidth, inHeight};
    for (auto& k : res_k) {
      result.rects.push_back(FaceDetectResult::BoundingBox{
          boxes[k][0] / inWidth,
          boxes[k][1] / inHeight,
          (boxes[k][2] - boxes[k][0]) / inWidth,
          (boxes[k][3] - boxes[k][1]) / inHeight,
          boxes[k][4]
      });
    }
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    auto postprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start);

    // Store the result in the shared structure
    auto store_start = std::chrono::high_resolution_clock::now();
    {
      std::lock_guard<std::mutex> lock(sharedResults.mutex);
      sharedResults.results.push_back(result);
      sharedResults.frame = img.clone();
      sharedResults.frame_index = index;
      sharedResults.cv.notify_one();
    }
    auto store_end = std::chrono::high_resolution_clock::now();
    auto store_duration = std::chrono::duration_cast<std::chrono::microseconds>(store_end - store_start);

    auto frame_end_time = std::chrono::high_resolution_clock::now();
    auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end_time - frame_start_time);
    
    frame_count++;
    total_latency += frame_duration.count();

    std::cout << "Face Detection latency for frame " << frame_count << ": " << frame_duration.count() << " ms" << std::endl;
    std::cout << "  Queue: " << queue_duration.count() << " µs" << std::endl;
    std::cout << "  Resize: " << resize_duration.count() << " µs" << std::endl;
    std::cout << "  Preprocess: " << preprocess_duration.count() << " µs" << std::endl;
    std::cout << "  DPU execution: " << dpu_duration.count() << " µs" << std::endl;
    std::cout << "  Postprocess: " << postprocess_duration.count() << " µs" << std::endl;
    std::cout << "  Store results: " << store_duration.count() << " µs" << std::endl;
  }

  auto total_end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
  double average_latency = total_latency / frame_count;
  
  std::cout << "Total Face Detection frames processed: " << frame_count << std::endl;
  std::cout << "Face Detection - Average latency: " << average_latency << " ms" << std::endl;
  std::cout << "Face Detection - Total runtime: " << total_duration.count() << " ms" << std::endl;

  // Clean up allocated memory
  delete[] imageInputs;
  delete[] FCResult;
  delete[] pred;

  std::cout << "Face Detection Thread Finished\n";
}

/* Run Facerecog */
float feature_norm(const float *feature) {
  float sum = 0;
  for (int i = 0; i < 512; ++i) {
    sum += feature[i] * feature[i];
  }
  return 1.f / sqrt(sum);
}

float feature_dot(const float *f1, const float *f2) {
  float dot = 0;
  for (int i = 0; i < 512; ++i) {
    dot += f1[i] * f2[i];
  }
  return (float)dot;
}

float cosine_similarity(const float *feature1, const float *feature2) {
  float norm1 = feature_norm(feature1);
  float norm2 = feature_norm(feature2);
  return feature_dot(feature1, feature2) * norm1 * norm2;
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<std::string>> load_embeddings(std::string embeddings_npzpath) {
  // std::map<std::string, std::vector<float>> embeddings_map;
  cnpy::npz_t npy_map = cnpy::npz_load(embeddings_npzpath); // using npz_t = std::map<std::string, NpyArray>;

  std::vector<std::vector<float>> embedding_arr;
  std::vector<float> embedding_norm_arr;
  std::vector<std::string> embedding_class_arr;

  for (auto &pair : npy_map) {
    std::string fname = pair.first;
    cnpy::NpyArray value_arr = pair.second;
    int value_size = value_arr.num_vals; // 512

    const float* value_ptr = value_arr.data<float>();
    std::vector<float> value(value_ptr, value_ptr + value_size);
    embedding_arr.push_back(value);
    embedding_norm_arr.push_back(feature_norm(value_ptr));
    embedding_class_arr.push_back(fname.substr(0, fname.rfind('/')));
  }
  // embedding_arr : {[e0,e1, ..., e512],[e0,e1, ..., e512], ...}
  // embedding_norm_arr : {e0, e1, ...}
  // embedding_class_arr : {c1, c1, ...} // c1 like "{class_name}/{fname}"
  return std::make_tuple(embedding_arr, embedding_norm_arr, embedding_class_arr);
}

// NormalizeInputDataRGB function
void NormalizeInputDataRGB(const uint8_t* input, int rows, int cols,
                           int channels, int stride,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale, int8_t* data) {
  for (int h = 0; h < rows; ++h) {
    for (int w = 0; w < cols; ++w) {
      for (int c = 0; c < channels; ++c) {
        int value = std::round(((input[h * stride + w * channels + c] * 1.0f - mean[c]) * scale[c]));
        value = std::max(-128, std::min(127, value));
        data[h * cols * channels + w * channels + (2 - c)] = (int8_t)value;
      }
    }
  }
}

void runFacerecog(vart::Runner* runner, bool& is_running) {
  std::cout << "Face Recognition Thread Started\n";

  // Load pre-computed embeddings
  auto embeddings_start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<float>> embedding_arr;
  std::vector<float> embedding_norm_arr;
  std::vector<std::string> embedding_class_arr;
  auto embeddings_npzpath = "/usr/share/vitis_ai_library/models/InceptionResnetV1/embeddings_xmodel.npz";
  std::tie(embedding_arr, embedding_norm_arr, embedding_class_arr) = load_embeddings(embeddings_npzpath);
  auto embeddings_end = std::chrono::high_resolution_clock::now();
  auto embeddings_duration = std::chrono::duration_cast<std::chrono::milliseconds>(embeddings_end - embeddings_start);
  std::cout << "Embeddings loading time: " << embeddings_duration.count() << " ms" << std::endl;

  // Get model input and output details 
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();
  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);
  int outSize = out_dims[1];
  int inSize = in_dims[1] * in_dims[2] * in_dims[3];
  int inHeight = in_dims[1];
  int inWidth = in_dims[2];
  int batchSize = in_dims[0];

  // Define preprocessing parameters
  std::vector<float> mean = {128.0f, 128.0f, 128.0f};
  std::vector<float> scale = {0.0078125f, 0.0078125f, 0.0078125f};

  // Timing variables
  auto total_start_time = std::chrono::high_resolution_clock::now();
  int frame_count = 0;
  double total_latency = 0.0;

  while (is_running) {
    auto frame_start_time = std::chrono::high_resolution_clock::now();

    FaceDetectResult faceDetectResult;
    cv::Mat frame;
    int frame_index;
    bool have_result = false;

    // Acquire lock and wait for results or finish signal
    auto wait_start = std::chrono::high_resolution_clock::now();
    {
      std::unique_lock<std::mutex> lock(sharedResults.mutex);
      sharedResults.cv.wait(lock, [&]{
        return !sharedResults.results.empty() || sharedResults.finished;
      });

      // Check if we're finished and no more results
      if (sharedResults.finished && sharedResults.results.empty()) {
        break;
      }

      // If we have results, get one result from the queue
      if (!sharedResults.results.empty()) {
        faceDetectResult = sharedResults.results.front();
        sharedResults.results.erase(sharedResults.results.begin());
        frame = sharedResults.frame.clone();
        frame_index = sharedResults.frame_index;
        have_result = true;
      }
    }
    auto wait_end = std::chrono::high_resolution_clock::now();
    auto wait_duration = std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start);

    if (have_result) {
      // Crop faces
      auto crop_start = std::chrono::high_resolution_clock::now();
      std::vector<cv::Mat> croppedFaces;
      for (const auto &r : faceDetectResult.rects) {
        cv::Mat cropped_img = frame(cv::Rect(r.x * frame.cols, r.y * frame.rows,
                                             r.width * frame.cols, r.height * frame.rows));
        cv::Mat resized_img;
        cv::resize(cropped_img, resized_img, cv::Size(inWidth, inHeight));
        croppedFaces.push_back(resized_img);
      }
      auto crop_end = std::chrono::high_resolution_clock::now();
      auto crop_duration = std::chrono::duration_cast<std::chrono::microseconds>(crop_end - crop_start);

      auto process_start = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < croppedFaces.size(); ++i) {
        std::vector<int8_t> imageInputs(inSize * batchSize, 0);
        std::vector<int8_t> FCResult(batchSize * outSize, 0);

        // Convert image to input format
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        for (int h = 0; h < inHeight; h++) {
          for (int w = 0; w < inWidth; w++) {
            for (int c = 0; c < 3; c++) {
              int index = h * inWidth * 3 + w * 3 + c;
              imageInputs[index] = static_cast<int8_t>(
                (croppedFaces[i].at<cv::Vec3b>(h, w)[c] - mean[c]) * input_scale);
            }
          }
        }
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start);

        // Prepare input and output tensors
        std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
        std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(imageInputs.data(), inputTensors[0]));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(FCResult.data(), outputTensors[0]));
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.push_back(outputs[0].get());

        // Execute the DPU
        auto dpu_start = std::chrono::high_resolution_clock::now();
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        auto status = runner->wait(job_id.first, -1);
        if (status != 0) {
          std::cerr << "DPU execution failed with status " << status << std::endl;
          continue;
        }
        auto dpu_end = std::chrono::high_resolution_clock::now();
        auto dpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpu_end - dpu_start);

        // Post-processing and face recognition 
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        std::vector<float> float_output(outSize);
        for (int j = 0; j < outSize; ++j) {
          float_output[j] = static_cast<float>(FCResult[j]) * output_scale;
        }

        float max_similarity = -1.0f;
        int max_similarity_index = -1;
        for (size_t e = 0; e < embedding_arr.size(); ++e) {
          float similarity = cosine_similarity(float_output.data(), embedding_arr[e].data());
          if (similarity > max_similarity) {
            max_similarity = similarity;
            max_similarity_index = e;
          }
        }

        std::string recognized_label = "Unknown";
        if (max_similarity_index != -1 && max_similarity > 0.6) {
          recognized_label = embedding_class_arr[max_similarity_index];
        }
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        auto postprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start);

        // Draw bounding box and label on the frame
        auto draw_start = std::chrono::high_resolution_clock::now();
        const auto& r = faceDetectResult.rects[i];
        cv::rectangle(frame, cv::Rect(r.x * frame.cols, r.y * frame.rows,
                                      r.width * frame.cols, r.height * frame.rows),
                      cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, recognized_label,
                    cv::Point(r.x * frame.cols, r.y * frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        auto draw_end = std::chrono::high_resolution_clock::now();
        auto draw_duration = std::chrono::duration_cast<std::chrono::microseconds>(draw_end - draw_start);

        std::cout << "Face " << i+1 << " Recognition timings:" << std::endl;
        std::cout << "  Preprocess: " << preprocess_duration.count() << " µs" << std::endl;
        std::cout << "  DPU execution: " << dpu_duration.count() << " µs" << std::endl;
        std::cout << "  Postprocess: " << postprocess_duration.count() << " µs" << std::endl;
        std::cout << "  Drawing: " << draw_duration.count() << " µs" << std::endl;
        std::cout << "  Recognized as: " << recognized_label << " (Similarity: " << max_similarity << ")" << std::endl;
      }
      auto process_end = std::chrono::high_resolution_clock::now();
      auto process_duration = std::chrono::duration_cast<std::chrono::microseconds>(process_end - process_start);

      // Put processed frame into display queue
      auto display_start = std::chrono::high_resolution_clock::now();
      mtx_display_queue.lock();
      display_queue.push(std::make_pair(frame_index, frame));
      mtx_display_queue.unlock();
      auto display_end = std::chrono::high_resolution_clock::now();
      auto display_duration = std::chrono::duration_cast<std::chrono::microseconds>(display_end - display_start);

      auto frame_end_time = std::chrono::high_resolution_clock::now();
      auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end_time - frame_start_time);
    
      frame_count++;
      total_latency += frame_duration.count();

      std::cout << "Face Recognition latency for frame " << frame_count << ": " << frame_duration.count() << " ms" << std::endl;
      std::cout << "  Wait for result: " << wait_duration.count() << " µs" << std::endl;
      std::cout << "  Crop faces: " << crop_duration.count() << " µs" << std::endl;
      std::cout << "  Process faces: " << process_duration.count() << " µs" << std::endl;
      std::cout << "  Add to display queue: " << display_duration.count() << " µs" << std::endl;
    }
  }

  auto total_end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
  double average_latency = total_latency / frame_count;

  std::cout << "Face Recognition - Total frames processed:" << frame_count << std::flush;
  std::cout << "Face Recognition - Average latency: " << average_latency << " ms" << std::flush;
  std::cout << "Face Recognition - Total runtime: " << total_duration.count() << " ms" << std::flush;
  
  std::cout << "Face Recognition Thread Finished\n";
}

/**
 * @brief Entry for runing SSD neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char** argv) {
  // Check args
  if (argc != 4) {
    cout << "Usage of video analysis demo: ./video_analysis [video_file] "
            "[fd_model_file]" "[fr_model_file]"
         << endl;
    return -1;
  }

  //모델 path
  argv[1] = "/home/root/VART/video_fd_fr/all5_720p.mp4";
  argv[2] = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.xmodel";
  argv[3] = "/usr/share/vitis_ai_library/models/InceptionResnetV1/InceptionResnetV1.xmodel";

  // Initialize video capture
  string file_name = argv[1];
  cout << "Detect video: " << file_name << endl;
  video.open(file_name);
  if (!video.isOpened()) {
    cout << "Failed to open video: " << file_name;
    return -1;
  }

  // Load the face detection and face recognition models
  auto graph_fd = xir::Graph::deserialize(argv[2]);
  auto graph_fr = xir::Graph::deserialize(argv[3]);

  // Extract the DPU subgraphs from the loaded models
  auto subgraph_fd = get_dpu_subgraph(graph_fd.get());
  auto subgraph_fr = get_dpu_subgraph(graph_fr.get());

  CHECK_EQ(subgraph_fd.size(), 1u) //DPU subgraph가 정확히 하나인지 확인
      << "Facedetect should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph_fd[0]->get_name(); //추출된 subgraph의 이름을 로그에 출력
  CHECK_EQ(subgraph_fr.size(), 1u) //DPU subgraph가 정확히 하나인지 확인
      << "Facerecog should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph_fr[0]->get_name(); //추출된 subgraph의 이름을 로그에 출력

  // create runner
  auto runner_fd = vart::Runner::create_runner(subgraph_fd[0], "run"); //추출된 subgraph를 실행할 runner객체 생성
  auto runner_fr = vart::Runner::create_runner(subgraph_fr[0], "run"); //추출된 subgraph를 실행할 runner객체 생성
  //auto runner = vart::Runner::create_runner(subgraph[0], "run");
  //auto runner1 = vart::Runner::create_runner(subgraph[0], "run");
  //auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  //auto runner3 = vart::Runner::create_runner(subgraph[0], "run");
  //auto runner4 = vart::Runner::create_runner(subgraph[0], "run");
  //auto runner5 = vart::Runner::create_runner(subgraph[0], "run");

  // Get input/output tensor shapes for face detection
  auto inputTensors_fd = runner_fd->get_input_tensors();
  auto outputTensors_fd = runner_fd->get_output_tensors();
  int inputCnt_fd = inputTensors_fd.size();
  int outputCnt_fd = outputTensors_fd.size();
  TensorShape inshapes_fd[inputCnt_fd];
  TensorShape outshapes_fd[outputCnt_fd];

  // Get input/output tensor shapes for face recognition
  auto inputTensors_fr = runner_fr->get_input_tensors();
  auto outputTensors_fr = runner_fr->get_output_tensors();
  int inputCnt_fr = inputTensors_fr.size();
  int outputCnt_fr = outputTensors_fr.size();
  TensorShape inshapes_fr[inputCnt_fr];
  TensorShape outshapes_fr[outputCnt_fr];

  // Set up shapes for face detection
  shapes.inTensorList = inshapes_fd;
  shapes.outTensorList = outshapes_fr;
  getTensorShape(runner_fd.get(), &shapes, inputCnt_fd, outputCnt_fd);

  // Run tasks for FD, FR
  //vector<thread> threads(TNUM); // TNUM(6)개의 스레드를 저장할 벡터 생성
  vector<thread> threads;
  is_running.fill(true);  // 모든 스레드의 실행 상태를 true로 초기화
  // 6개의 SSD 처리 스레드 생성 및 시작
  // 각 스레드는 서로 다른 runner 인스턴스를 사용하여 병렬 처리
  //threads[0] = thread(RunFaceDetect, runner.get(), ref(is_running[0]));
  //threads[1] = thread(RunFaceDetect, runner1.get(), ref(is_running[1]));
  //threads[2] = thread(RunFaceDetect, runner2.get(), ref(is_running[2]));
  //threads[3] = thread(RunFaceDetect, runner3.get(), ref(is_running[3]));
  //threads[4] = thread(RunFaceDetect, runner4.get(), ref(is_running[4]));
  //threads[5] = thread(RunFaceDetect, runner5.get(), ref(is_running[5]));
  threads.push_back(thread(RunFaceDetect, runner_fd.get(), ref(is_running[0])));
  threads.push_back(thread(runFacerecog, runner_fr.get(), ref(is_running[1])));

  //threads.push_back(thread(RunFaceDetect, runner_fd.get(), ref(is_running[2])));
  //threads.push_back(thread(runFacerecog, runner_fr.get(), ref(is_running[3])));
  // 비디오 프레임 읽기 스레드 생성 및 시작
  threads.push_back(thread(Read, ref(is_reading)));
  // 결과 표시 스레드 생성 및 시작
  //'Display'함수가 다른 thread와 병렬로 실행됨.
  //'is_displaying'변수를 Display함수에 전달 가능
  threads.push_back(thread(Display, ref(is_displaying)));
  

  // 모든 스레드가 완료될 때까지 대기
  // TNUM(6)개의 SSD 스레드 + 읽기 스레드 + 표시 스레드 = 총 8개 스레드 
  for (int i = 0; i < 2 + TNUM; ++i) {
    threads[i].join();
  }

  // 비디오 리소스 해제
  video.release();
  return 0;
}
