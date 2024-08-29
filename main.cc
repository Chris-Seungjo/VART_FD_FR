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
//rev2
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cnpy.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

GraphInfo shapes;

const string baseImagePath = "../images/";
const string wordsPath = "./";

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */

 //얼굴 검출 결과를 저장할 구조체 정의
 struct FaceDetectResult {
   struct BoundingBox {
     float x;
     float y;
     float width;
     float height;
     float score;
   };
   int width;
   int height;
   std::vector<BoundingBox> rects;
 };

void ListImages(string const& path, vector<string>& images) {
  images.clear();
  struct dirent* entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const& path, vector<string>& kinds) {
  kinds.clear();
  ifstream fkinds(path);
  if (fkinds.fail()) {
    fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
    exit(1);
  }
  string kind;
  while (getline(fkinds, kind)) {
    kinds.push_back(kind);
  }

  fkinds.close();
}

/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const int8_t* data, size_t size, float* result,
                    float scale) {
  assert(data && result);
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp((float)data[i] * scale);
    sum += result[i];
  }
  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float* d, int size, int k, vector<string>& vkinds) {
  assert(d && size > 0 && k > 0);
  priority_queue<pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(pair<float, int>(d[i], i));
  }

  for (auto i = 0; i < k; ++i) {
    pair<float, int> ki = q.top();
    printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
           vkinds[ki.second].c_str());
    q.pop();
  }
}

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

/**
 * @brief Run DPU Task for ResNet50
 *
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return none
 */

std::vector<FaceDetectResult> runFacedetect(vart::Runner* runner) {
  std::vector<FaceDetectResult> allResults; //FD 결과인 bounding box 위치를 저장할 vector 선언
  vector<string> images;  //이미지 파일 목록을 저장할 벡터
  ListImages(baseImagePath, images); //지정된 경로에서 이미지 파일 목록을 가져옴
  if (images.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return std::vector<FaceDetectResult>();
  }

  //preprocessing을 위한 mean, scale 설정
  std::vector<float> mean = {128.0f, 128.0f, 128.0f};
  std::vector<float> scale = {1.0f, 1.0f, 1.0f};

  //DPU runner로부터 입력, 출력 tensor 정보 가져오기
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();
  //입력 및 출력 scale가져오기
  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);
  //입출력 텐서의 크기 및 형태 정보 설정
  int outSize = out_dims[1] * out_dims[2] * out_dims[3]; //height*width*channels
  int inSize = in_dims[1] * in_dims[2] * in_dims[3];
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int batchSize = in_dims[0];

   // Tensor 정보 설정
  std::cout << "Output Tensor Shape: ";
  for (const auto& dim : out_dims) {
      std::cout << dim << " ";  //1 96 160 4
  }
  std::cout << std::endl;

  std::cout << "Input Tensor Shape: ";
  for (const auto& dim : in_dims) {
      std::cout << dim << " ";  //1 360 640 3
  }
  std::cout << std::endl;

  std::cout << "Input Scale: " << input_scale << std::endl; //1
  std::cout << "Output Scale: " << output_scale << std::endl; //1
  std::cout << "Output Size: " << outSize << std::endl; //61440
  std::cout << "out_dims[0] is " << out_dims[0] << std::endl; //1
  std::cout << "out_dims[1] is " << out_dims[1] << std::endl; //96
  std::cout << "out_dims[2] is " << out_dims[2] << std::endl; //160
  std::cout << "out_dims[3] is " << out_dims[3] << std::endl; //4
  std::cout << "Input Size: " << inSize << std::endl; //691200
  std::cout << "Input Height: " << inHeight << std::endl; //360
  std::cout << "Input Width: " << inWidth << std::endl; //640
  std::cout << "Batch Size: " << batchSize << std::endl;  //1

   //DPU 실행을 위한 입출력 buffer 준비
  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  //입력 이미지, 결과를 저장할 buffer 할당
  int8_t* imageInputs = new int8_t[inSize * batchSize]; //setImageBGR함수의 결과가 들어갈 배열 //resnet50참고해서 만듬
  int8_t* FCResult = new int8_t[batchSize * outSize]; //Inference 결과가 들어갈 배열

  //모든 이미지에 대해 처리 수행
  for (unsigned int n = 0; n < images.size(); n += batchSize) {
    unsigned int runSize = std::min((unsigned int)batchSize, (unsigned int)(images.size() - n));
    
    //이미지 로드 및 resize
    vector<cv::Mat> imageList;
    for (unsigned int i = 0; i < runSize; i++) {
      cv::Mat image = cv::imread(baseImagePath + images[n + i]);
      if (image.empty()) {
        cerr << "Error: Unable to load image " << images[n + i] << endl;
        continue;
      }
      cv::Mat resized;
      cv::resize(image, resized, cv::Size(inWidth, inHeight));
      imageList.push_back(image);
    }

    //이미지 데이터를 입력 형식에 맞게 변환
    setImageBGR(imageList, runner, imageInputs, mean, scale);
    
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(inputTensors[0]->get_name(), in_dims, xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(imageInputs, batchTensors.back().get()));

    batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[0]->get_name(), out_dims, xir::DataType{xir::DataType::XINT, 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(FCResult, batchTensors.back().get()));

    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    //DPU 실행
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    auto status = runner->wait(job_id.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";

    //Post-processing
    const float det_threshold = 0.9f;
    const float nms_threshold = 0.3f;
    
    
    const auto out_tensors00_channel_ = outputTensors[0]->get_shape()[3]; //=4 //channel => 위에 out_dim[3] 참고 (x,y, width, height)
    const auto out_tensors01_channel_ = outputTensors[1]->get_shape()[3]; //=2 
    auto conv_idx = 0;
    auto bbox_idx = 1;

    std::cout << "output_tensors[0] channel is " << out_tensors00_channel_ << std::endl;
    std::cout << "output_tensors[1] channel is " << out_tensors01_channel_ << std::endl;

    if (out_tensors00_channel_ == 2 && out_tensors01_channel_ == 4) {
      conv_idx = 0;
      bbox_idx = 1;
    } else if (out_tensors00_channel_ == 4 && out_tensors01_channel_ == 2) {
      conv_idx = 1;
      bbox_idx = 0;
    } else {
      std::cout << "Output tensors channel is unexpected. "
                << "output_tensors[0] channel is " << out_tensors00_channel_
                << " output_tensors[1] channel is " << out_tensors01_channel_
                << std::endl;
    }

    std::cout << "conv_idx is set to " << conv_idx << " and bbox_idx is set to " << bbox_idx << std::endl;

    //컨볼루션 출력 관련 정보 설정
    const auto conv_out_size_ = (out_dims[1] * out_dims[2] * out_dims[3])/batchSize;
    const auto conv_out_addr_ = FCResult + conv_idx * conv_out_size_; //conv_out 데이터 시작 위치(배열)
    const auto conv_out_scale_ = std::exp2f(-1.0f * (float)outputTensors[conv_idx]->get_attr<int>("fix_point"));  //: 양자화된 값을 부동소수점으로 변환할 때 사용하는 방법
    const auto conv_out_channel_ = outputTensors[conv_idx]->get_shape()[3]; //output tensor channel
    //conv_out_scale 원본코드 std::exp2f(-1.0f * (float)tensor.fixpos); 
    //tensor의 고정소수점 위치를 가져온 후, -1을 곱하고 float 으로 변환, 2의 거듭제곱 계산 
    //: 양자화된 값을 부동소수점으로 변환할 때 사용하는 방법

    std::cout << "conv_out_size_ is " << conv_out_size_ << std::endl;
    std::cout << "conv_out_addr_ is " << (void*)conv_out_addr_ << std::endl;
    std::cout << "conv_out_scale_ is " << conv_out_scale_ << std::endl;
    std::cout << "conv_out_channel_ is " << conv_out_channel_ << std::endl;

    //Softmax 연산 수행
    std::vector<float> conf(conv_out_size_);
    softmax(conv_out_addr_, conv_out_scale_, conv_out_channel_, 
            conv_out_size_ / conv_out_channel_, conf.data()); //conf.data()가 output

    //Bounding box 출력 정보 설정
    int8_t* bbout = FCResult + bbox_idx * conv_out_size_;
    const auto bb_out_width = out_dims[2];
    const auto bb_out_height = out_dims[1];
    const auto bb_out_scale = std::exp2f(-1.0f * (float)outputTensors[bbox_idx]->get_attr<int>("fix_point"));

    //BoundingBox 필터링
    std::vector<std::vector<float>> boxes = FilterBox(bb_out_scale, det_threshold, 
                                                      bbout, bb_out_width, bb_out_height, conf.data());

    //score를 넣음
    std::vector<float> scores;
    for (auto& box : boxes) { //모든 박스에 대해 반복
        scores.push_back(box[4]); //score에 box[4]원소를 넣음
    }

    //NMS(Non-Maximum Suppression) 적용
    std::vector<size_t> res_k;
    int max_faces = 1; // 최대 1개 박스로 제한
    applyNMS(boxes, scores, nms_threshold, det_threshold, res_k, max_faces);

    //최종 결과 생성
    FaceDetectResult result{inWidth, inHeight};
    //박스 좌표 조정
    //원본 좌표 정규화, 좌표표현방식 (x1, y1, x2,y2) -> (x1, y1, width, height)
    for (auto& k : res_k) {
      result.rects.push_back(FaceDetectResult::BoundingBox{
        boxes[k][0] / inWidth,  //x
        boxes[k][1] / inHeight, //y
        (boxes[k][2] - boxes[k][0]) / inWidth, //x2-x1 Width
        (boxes[k][3] - boxes[k][1]) / inHeight, //y2-y1 Height
        boxes[k][4] //Score
      });
    }

    //bounding box 위치 'allResults'에 저장
    allResults.push_back(result);

    // Draw bounding boxes, save the image, and print box positions
    for (unsigned int i = 0; i < runSize; i++) {
      cv::Mat image = imageList[i];
      cv::Mat resized;
      cv::resize(image, resized, cv::Size(result.width, result.height));

      std::cout << "Detected faces in image " << images[n + i] << ":" << std::endl;
      for (const auto &r : result.rects) {
        cv::rectangle(resized,
                      cv::Rect{cv::Point(r.x * resized.cols, r.y * resized.rows),
                               cv::Size{(int)(r.width * resized.cols),
                                        (int)(r.height * resized.rows)}},
                      cv::Scalar(0, 255, 0), 2);

        // Print box position and score
        std::cout << "  Box: x=" << r.x << ", y=" << r.y 
                  << ", width=" << r.width << ", height=" << r.height 
                  << ", score=" << r.score << std::endl;
      }

      std::string output_filename = "output_" + images[n + i];
      cv::imwrite(output_filename, resized);
      std::cout << "Saved result image: " << output_filename << std::endl;
    }

  }

  return allResults;

  delete[] imageInputs;
  delete[] FCResult;
}

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

void runFacerecog(vart::Runner* runner, const std::vector<FaceDetectResult>& faceDetectResults) {
  std::cout << "\n";
  std::cout << "/////////////////////////////\n";
  std::cout << "///Face Recognition Start///\n";
  std::cout << "/////////////////////////////\n";

  // Initialization and setup
  vector<string> images;
  ListImages(baseImagePath, images);  // Load image file names

  std::vector<cv::Mat> croppedFaces;  // Store cropped face images

  // Load pre-computed embeddings
  std::vector<std::vector<float>> embedding_arr;
  std::vector<float> embedding_norm_arr;
  std::vector<std::string> embedding_class_arr;
    auto embeddings_npzpath = "/usr/share/vitis_ai_library/models/InceptionResnetV1/embeddings_xmodel.npz";
  std::tie(embedding_arr, embedding_norm_arr, embedding_class_arr) = load_embeddings(embeddings_npzpath);

   // Get model input and output details
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();
  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);
  int outSize = out_dims[1];  // Changed from out_dims[1] * out_dims[2] * out_dims[3]
  int inSize = in_dims[1] * in_dims[2] * in_dims[3];
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int batchSize = in_dims[0];

  // Print model details for debugging
  std::cout << "Rec_Output Tensor Shape: ";
  for (const auto& dim : out_dims) {
      std::cout << dim << " ";
  }
  std::cout << std::endl;

  std::cout << "Rec_Input Tensor Shape: ";
  for (const auto& dim : in_dims) {
      std::cout << dim << " ";
  }
  std::cout << std::endl;

  std::cout << "Rec_Input Scale: " << input_scale << std::endl;
  std::cout << "Rec_Output Scale: " << output_scale << std::endl;
  std::cout << "Rec_Output Size: " << outSize << std::endl;
  std::cout << "Rec_in_dims[0] is " << in_dims[0] << std::endl;
  std::cout << "Rec_in_dims[1] is " << in_dims[1] << std::endl;
  std::cout << "Rec_in_dims[2] is " << in_dims[2] << std::endl;
  std::cout << "Rec_in_dims[3] is " << in_dims[3] << std::endl;
  std::cout << "Rec_out_dims[0] is " << out_dims[0] << std::endl;
  std::cout << "Rec_out_dims[1] is " << out_dims[1] << std::endl;
  std::cout << "Rec_out_dims[2] is " << out_dims[2] << std::endl;
  std::cout << "Rec_out_dims[3] is " << out_dims[3] << std::endl;
  std::cout << "Rec_Input Size: " << inSize << std::endl;
  std::cout << "Rec_Input Height: " << inHeight << std::endl;
  std::cout << "Rec_Input Width: " << inWidth << std::endl;
  std::cout << "Rec_Batch Size: " << batchSize << std::endl;

  std::cout << "faceDetectResults.size() : " << faceDetectResults.size() << std::endl;
  std::cout << "\n";

  std::cout << "Face Detection Results:\n";

  // Process each detected face
  for (size_t i = 0; i < faceDetectResults.size(); ++i) {
    std::cout << "Image: " << images[i] << std::endl;
    const auto& result = faceDetectResults[i];
    // Print face detection results
    for (const auto& rect : result.rects) {
      std::cout << "  Bounding Box: "
                << "x=" << rect.x << ", "
                << "y=" << rect.y << ", "
                << "width=" << rect.width << ", "
                << "height=" << rect.height << ", "
                << "score=" << rect.score << std::endl;
    }

    // Load and process the original image
    cv::Mat image = cv::imread(baseImagePath + images[i]);
    if (image.empty()) {
      std::cerr << "Error: Unable to read image " << images[i] << std::endl;
      continue;
    }

    // Crop and save detected faces
    for (const auto &r : result.rects) {
      cv::Mat resized;
      cv::resize(image, resized, cv::Size(result.width, result.height));

      std::cout << "Detected faces in image " << images[i] << ":" << std::endl;

      cv::Mat cropped_img = resized(cv::Rect(r.x * resized.cols, 
                                            r.y * resized.rows,
                                            r.width * resized.cols,
                                            r.height * resized.rows));
      
      std::string crop_filename = "cropped_face_" + images[i] + "_" + 
                                  std::to_string(croppedFaces.size()) + ".jpg";
      cv::imwrite(crop_filename, cropped_img);
      std::cout << "Saved result image: " << crop_filename << std::endl;

      croppedFaces.push_back(cropped_img);
    }
  }

  // Define preprocessing parameters
  std::vector<float> mean = {128.0f, 128.0f, 128.0f};
  std::vector<float> scale = {0.0078125f, 0.0078125f, 0.0078125f};

  // Process each cropped face
  for (size_t i = 0; i < croppedFaces.size(); ++i) {
    std::cout << "inSize: " << inSize << ", batchSize: " << batchSize << ", outSize: " << outSize << std::endl;
    std::cout << "Attempting to allocate imageInputs with size: " << (inSize * batchSize) << std::endl;
    std::cout << "Attempting to allocate FCResult with size: " << (batchSize * outSize) << std::endl;

    std::vector<int8_t> imageInputs;
    std::vector<int8_t> FCResult;

    try {
        imageInputs.resize(inSize * batchSize, 0);
        FCResult.resize(batchSize * outSize, 0);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        continue;  // Skip to the next iteration of the loop
    }

    std::cout << "imageInputs size: " << imageInputs.size() << std::endl;
    std::cout << "FCResult size: " << FCResult.size() << std::endl;

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

    if (imageInputs.empty() || FCResult.empty()) {
        std::cerr << "Failed to allocate memory for imageInputs or FCResult" << std::endl;
        continue;
    }

    // Preprocess the cropped face image
    cv::Mat& cropped_img = croppedFaces[i];
    cv::Mat resized_img;
    cv::resize(cropped_img, resized_img, cv::Size(inWidth, inHeight));

    // Convert image to input format
    for (int h = 0; h < inHeight; h++) {
      for (int w = 0; w < inWidth; w++) {
        for (int c = 0; c < 3; c++) {
          int index = h * inWidth * 3 + w * 3 + c;
          if (index < inSize * batchSize) {
            imageInputs[index] = static_cast<int8_t>(
              (resized_img.at<cv::Vec3b>(h, w)[c] - mean[c]) * input_scale);
          }
        }
      }
    }

    std::cout << "imageInputs size: " << imageInputs.size() << std::endl;
    if (imageInputs.size() != 1 * 160 * 160 * 3) {
        std::cerr << "Warning: imageInputs size mismatch!" << std::endl;
        continue;
    }

    auto input_tensor = xir::Tensor::create(inputTensors[0]->get_name(), in_dims, xir::DataType{xir::DataType::XINT, 8u});
    if (!input_tensor) {
        std::cerr << "Failed to create input tensor" << std::endl;
        continue;
    }
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(std::move(input_tensor)));

    try {
        std::cout << "Creating input CpuFlatTensorBuffer..." << std::endl;
        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(imageInputs.data(), batchTensors.back().get()));
        std::cout << "Input CpuFlatTensorBuffer created successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception creating CpuFlatTensorBuffer for input: " << e.what() << std::endl;
        continue;
    }

    auto created_tensor = batchTensors.back();
    auto shape = created_tensor->get_shape();
    std::cout << "Created input tensor shape: ";
    for (const auto& dim : shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    auto output_tensor = xir::Tensor::create(outputTensors[0]->get_name(), out_dims, xir::DataType{xir::DataType::XINT, 8u});
    if (!output_tensor) {
        std::cerr << "Failed to create output tensor" << std::endl;
        continue;
    }
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(std::move(output_tensor)));

    try {
        std::cout << "Creating output CpuFlatTensorBuffer..." << std::endl;
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(FCResult.data(), batchTensors.back().get()));
        std::cout << "Output CpuFlatTensorBuffer created successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception creating CpuFlatTensorBuffer for output: " << e.what() << std::endl;
        continue;
    }

    created_tensor = batchTensors.back();
    shape = created_tensor->get_shape();
    std::cout << "Created output tensor shape: ";
    for (const auto& dim : shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    if (inputs.empty() || outputs.empty() || inputs.back()->data().first == 0 || outputs.back()->data().first == 0) {
        std::cerr << "Failed to create tensor buffer: data pointer is null" << std::endl;
        std::cout << "inputs.empty(): " << inputs.empty() << std::endl;
        std::cout << "outputs.empty(): " << outputs.empty() << std::endl;
        if (!inputs.empty()) std::cout << "inputs.back()->data().first: " << inputs.back()->data().first << std::endl;
        if (!outputs.empty()) std::cout << "outputs.back()->data().first: " << outputs.back()->data().first << std::endl;
        continue;
    }

    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    if (inputsPtr.empty() || outputsPtr.empty() || 
        inputsPtr[0]->data().first == 0 || outputsPtr[0]->data().first == 0) {
        std::cerr << "Invalid input or output pointers" << std::endl;
        continue;
    }

     // Execute the DPU (Deep Processing Unit) asynchronously
    try {
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        
        std::cout << "Input tensor shape: ";
        for (const auto& dim : inputsPtr[0]->get_tensor()->get_shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Output tensor shape: ";
        for (const auto& dim : outputsPtr[0]->get_tensor()->get_shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "First 10 values of input data: ";
        int8_t* input_data = reinterpret_cast<int8_t*>(inputsPtr[0]->data().first);
        for (int j = 0; j < 10; ++j) {
            std::cout << static_cast<int>(input_data[j]) << " ";
        }
        std::cout << std::endl;

        auto status = runner->wait(job_id.first, -1);
        if (status == 0) {
            std::cout << "DPU execution successful. FCResult values:" << std::endl;
            for (int j = 0; j < std::min(static_cast<int>(FCResult.size()), 20); ++j) {
                std::cout << static_cast<int>(FCResult[j]) << " ";
            }
            std::cout << "... (showing first 20 values)" << std::endl;
        } else {
            std::cerr << "DPU execution failed with status " << status << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during DPU execution: " << e.what() << std::endl;
        continue;
    }

    // Post-processing: Convert int8_t output to float
    std::vector<float> float_output(outSize);
    for (int j = 0; j < outSize; ++j) {
      float_output[j] = static_cast<float>(FCResult[j]) * output_scale;
    }

    // Find the best matching embedding
    float max_similarity = -1.0f;
    int max_similarity_index = -1;
    for (size_t e = 0; e < embedding_arr.size(); ++e) {
      float similarity = cosine_similarity(float_output.data(), embedding_arr[e].data());
      if (similarity > max_similarity) {
        max_similarity = similarity;
        max_similarity_index = e;
        std::cout << "max_similarity_value " << max_similarity << std::endl;
        std::cout << "max_similarity_index: " << max_similarity_index << std::endl;
      }
    }

    // Get the label for the best match
    std::string recognized_label = "Unknown";
    if (max_similarity_index != -1 && max_similarity > 0.6) { // You can adjust this threshold
      recognized_label = embedding_class_arr[max_similarity_index];
      std::cout << "recognized_label: " << recognized_label << std::endl;
    }

    // Draw bounding box and label on the original image
    cv::Mat original_image = cv::imread(baseImagePath + images[i]);
    if (!original_image.empty()) {
      const auto& result = faceDetectResults[i];
      for (const auto& r : result.rects) {
        int x = r.x * original_image.cols;
        int y = r.y * original_image.rows;
        int width = r.width * original_image.cols;
        int height = r.height * original_image.rows;

        cv::rectangle(original_image, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 2);
        cv::putText(original_image, recognized_label,
                        cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
      }

      // Save the image with bounding box and label
      std::string output_filename = "output_" + images[i];
      cv::imwrite(output_filename, original_image);
      std::cout << "Saved output image with detection: " << output_filename << std::endl;
    }

    std::cout << "Recognized face: " << recognized_label << " (Similarity: " << max_similarity << ")" << std::endl;
  
  }
}

/**
 * @brief Entry for runing Densebox640_360 neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy Densebox640_360 on DPU platform.
 *
 */

//graph -> graph_fd
//subgraph -> subgraph_fd
//runner -> runner_fd
//inputTensors -> inputTensors_fd
//outputTensors -> outputTensors_fd
int main(int argc, char* argv[]) {
  // Check args
  if (argc != 3) { //command line 인수가 2개 (프로그램 이름, 모델 파일)이 아니면 사용법 출력, 프로그램 종료
    cout << "Usage of facedetect demo: ./fd_fr [fd_model_file] [fr_model_file]" << endl;
    return -1;
  }

  //모델 path
  argv[1] = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.xmodel";
  argv[2] = "/usr/share/vitis_ai_library/models/InceptionResnetV1/InceptionResnetV1.xmodel";

  auto graph_fd = xir::Graph::deserialize(argv[1]);
  auto graph_fr = xir::Graph::deserialize(argv[2]);

  auto subgraph_fd = get_dpu_subgraph(graph_fd.get());
  auto subgraph_fr = get_dpu_subgraph(graph_fr.get());

  CHECK_EQ(subgraph_fd.size(), 1u) //DPU subgraph가 정확히 하나인지 확인 
      << "Facedetect should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph_fd[0]->get_name(); //추출된 subgraph의 이름을 로그에 출력
  CHECK_EQ(subgraph_fr.size(), 1u) //DPU subgraph가 정확히 하나인지 확인 
      << "Facerecog should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph_fr[0]->get_name(); //추출된 subgraph의 이름을 로그에 출력
  
  /*create runner*/
  auto runner_fd = vart::Runner::create_runner(subgraph_fd[0], "run"); //추출된 subgraph를 실행할 runner객체 생성
  auto runner_fr = vart::Runner::create_runner(subgraph_fr[0], "run"); //추출된 subgraph를 실행할 runner객체 생성
  // ai::XdpuRunner* runner = new ai::XdpuRunner("./");
  
  // runner에서 입,출력 텐서들 가져옴
  auto inputTensors_fd = runner_fd->get_input_tensors();
  auto outputTensors_fd = runner_fd->get_output_tensors();
  auto inputTensors_fr = runner_fr->get_input_tensors();
  auto outputTensors_fr = runner_fr->get_output_tensors();

  //입,출력 텐서 갯수 저장
  int inputCnt_fd = inputTensors_fd.size();
  int outputCnt_fd = outputTensors_fd.size();
  int inputCnt_fr = inputTensors_fr.size();
  int outputCnt_fr = outputTensors_fr.size();
  
  // 입, 출력 tnesor 형태 저장할 배열 생성, shape 구조체에 연결
  TensorShape inshapes_fd[inputCnt_fd]; //TensorShape -> common.h
  TensorShape outshapes_fd[outputCnt_fd];
  TensorShape inshapes_fr[inputCnt_fr]; //TensorShape -> common.h
  TensorShape outshapes_fr[outputCnt_fr];

  //Run FD
  shapes.inTensorList = inshapes_fd; //shapes -> Graphinfo -> common.h
  shapes.outTensorList = outshapes_fd;
  getTensorShape(runner_fd.get(), &shapes, inputCnt_fd, outputCnt_fd); //getTensorShape -> common.h
  /*run with batch*/
  /*얼굴 검출 실행*/
  auto faceDetectResults = runFacedetect(runner_fd.get());

  //Run FR
  shapes.inTensorList = inshapes_fr; //shapes -> Graphinfo -> common.h
  shapes.outTensorList = outshapes_fr;
  getTensorShape(runner_fr.get(), &shapes, inputCnt_fr, outputCnt_fr); //getTensorShape -> common.h
  /*run with batch*/
  runFacerecog(runner_fr.get(), faceDetectResults);

  return 0;
}
