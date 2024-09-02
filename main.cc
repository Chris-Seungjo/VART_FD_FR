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
#include <chrono>

#include "common.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

GraphInfo shapes;

struct FaceDetectResult {
    struct BoundingBox {
        float x, y, width, height, score;
    };
    int width, height;
    vector<BoundingBox> rects;
};

// Function prototypes
void NormalizeInputData(const unsigned char* input, int rows, int cols, int channels,
                        int stride, const vector<float>& mean,
                        const vector<float>& scale, int8_t* data);
void setImageBGR(const vector<Mat>& imgs, vart::Runner* runner, int8_t* inputData,
                 const vector<float>& mean, const vector<float>& scale);
void softmax(const int8_t* input, float scale, unsigned int cls, unsigned int group, float* output);
vector<vector<float>> FilterBox(const float bb_out_scale, const float det_threshold,
                                int8_t* bbout, int w, int h, float* pred);
float cal_iou(const vector<float>& box1, const vector<float>& box2);
void applyNMS(const vector<vector<float>>& boxes, const vector<float>& scores,
              const float nms_threshold, const float conf_threshold, vector<size_t>& res,
              int max_output);
FaceDetectResult runFacedetect(vart::Runner* runner, const Mat& frame);
tuple<vector<vector<float>>, vector<float>, vector<string>> load_embeddings(string embeddings_npzpath);
void NormalizeInputDataRGB(const uint8_t* input, int rows, int cols, int channels, int stride,
                           const vector<float>& mean, const vector<float>& scale, int8_t* data);
float feature_norm(const float *feature);
float feature_dot(const float *f1, const float *f2);
float cosine_similarity(const float *feature1, const float *feature2);
void runFacerecog(vart::Runner* runner, const Mat& frame, const FaceDetectResult& faceDetectResult);


// Implement all the functions (NormalizeInputData, setImageBGR, softmax, FilterBox, cal_iou, applyNMS, 
// runFacedetect, load_embeddings, NormalizeInputDataRGB, runFacerecog) here...
// These functions should be the same as in the multi-threaded version.

void NormalizeInputData(const unsigned char* input, int rows, int cols, int channels,
                        int stride, const vector<float>& mean,
                        const vector<float>& scale, int8_t* data) {
    for (int h = 0; h < rows; ++h) {
        for (int w = 0; w < cols; ++w) {
            for (int c = 0; c < channels; ++c) {
                float value = (input[h * stride + w * channels + c] * 1.0f - mean[c]) * scale[c];
                int rounded_value = round(value);
                rounded_value = max(-128, min(127, rounded_value));
                data[h * cols * channels + w * channels + c] = (int8_t)rounded_value;
            }
        }
    }
}

void setImageBGR(const vector<Mat>& imgs, 
                 vart::Runner* runner,
                 int8_t* inputData,
                 const vector<float>& mean,
                 const vector<float>& scale) {
    auto inputTensors = runner->get_input_tensors();
    CHECK_GT(inputTensors.size(), 0u);
    
    auto in_dims = inputTensors[0]->get_shape();
    int batchSize = in_dims[0];
    int height = in_dims[1];
    int width = in_dims[2];
    int channels = in_dims[3];
    
    CHECK_EQ(imgs.size(), batchSize);
    CHECK_LE(imgs.size(), batchSize);
    
    float input_scale = get_input_scale(inputTensors[0]);
    vector<float> real_scale{scale[0] * input_scale,
                             scale[1] * input_scale,
                             scale[2] * input_scale};

    for (size_t i = 0; i < imgs.size(); i++) {
        int8_t* batchData = inputData + i * height * width * channels;
        NormalizeInputData(imgs[i].data, height, width, channels, imgs[i].step,
                           mean, real_scale, batchData);
    }
}

// ... (Include all other function implementations here)

FaceDetectResult runFacedetect(vart::Runner* runner, const Mat& frame) {
    auto outputTensors = runner->get_output_tensors();
    auto inputTensors = runner->get_input_tensors();
    auto out_dims = outputTensors[0]->get_shape();
    auto in_dims = inputTensors[0]->get_shape();

    auto input_scale = get_input_scale(inputTensors[0]);
    auto output_scale = get_output_scale(outputTensors[0]);

    int outSize = out_dims[1] * out_dims[2] * out_dims[3];
    int inSize = in_dims[1] * in_dims[2] * in_dims[3];
    int inHeight = in_dims[1];
    int inWidth = in_dims[2];
    int batchSize = in_dims[0];

    vector<int8_t> imageInputs(inSize * batchSize);
    vector<int8_t> FCResult(batchSize * outSize);

    vector<float> mean = {128.0f, 128.0f, 128.0f};
    vector<float> scale = {1.0f, 1.0f, 1.0f};

    Mat resized;
    cv::resize(frame, resized, Size(inWidth, inHeight));

    setImageBGR({resized}, runner, imageInputs.data(), mean, scale);

    vector<unique_ptr<vart::TensorBuffer>> inputs, outputs;
    vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    
    inputs.push_back(make_unique<CpuFlatTensorBuffer>(imageInputs.data(), inputTensors[0]));
    outputs.push_back(make_unique<CpuFlatTensorBuffer>(FCResult.data(), outputTensors[0]));
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    auto status = runner->wait(job_id.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";

    const float det_threshold = 0.9f;
    const float nms_threshold = 0.3f;
    
    vector<float> pred(out_dims[1] * out_dims[2] * 2);
    for (int i = 0; i < out_dims[1] * out_dims[2]; ++i) {
        pred[i * 2] = 1.0f - (FCResult[i * 2] * output_scale);
        pred[i * 2 + 1] = FCResult[i * 2 + 1] * output_scale;
    }    

    vector<vector<float>> boxes = FilterBox(
        output_scale,
        det_threshold,
        FCResult.data(),
        out_dims[2],
        out_dims[1],
        pred.data()
    );

    vector<float> scores;
    for (auto& box : boxes) {
        scores.push_back(box[4]);
    }

    vector<size_t> res_k;
    int max_faces = 1;
    applyNMS(boxes, scores, nms_threshold, det_threshold, res_k, max_faces);

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

    return result;
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
    return dot;
}

float cosine_similarity(const float *feature1, const float *feature2) {
    float norm1 = feature_norm(feature1);
    float norm2 = feature_norm(feature2);
    return feature_dot(feature1, feature2) * norm1 * norm2;
}

tuple<vector<vector<float>>, vector<float>, vector<string>> load_embeddings(string embeddings_npzpath) {
    cnpy::npz_t npy_map = cnpy::npz_load(embeddings_npzpath);

    vector<vector<float>> embedding_arr;
    vector<float> embedding_norm_arr;
    vector<string> embedding_class_arr;

    for (auto &pair : npy_map) {
        string fname = pair.first;
        cnpy::NpyArray value_arr = pair.second;
        int value_size = value_arr.num_vals;

        const float* value_ptr = value_arr.data<float>();
        vector<float> value(value_ptr, value_ptr + value_size);
        embedding_arr.push_back(value);
        embedding_norm_arr.push_back(feature_norm(value_ptr));
        embedding_class_arr.push_back(fname.substr(0, fname.rfind('/')));
    }
    return make_tuple(embedding_arr, embedding_norm_arr, embedding_class_arr);
}

void NormalizeInputDataRGB(const uint8_t* input, int rows, int cols,
                           int channels, int stride,
                           const vector<float>& mean,
                           const vector<float>& scale, int8_t* data) {
    for (int h = 0; h < rows; ++h) {
        for (int w = 0; w < cols; ++w) {
            for (int c = 0; c < channels; ++c) {
                int value = round(((input[h * stride + w * channels + c] * 1.0f - mean[c]) * scale[c]));
                value = max(-128, min(127, value));
                data[h * cols * channels + w * channels + (2 - c)] = (int8_t)value;
            }
        }
    }
}

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

vector<vector<float>> FilterBox(const float bb_out_scale, const float det_threshold, int8_t* bbout, int w, int h, float* pred) {
    vector<vector<float>> boxes;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int position = i * w + j;
            if (pred[position * 2 + 1] > det_threshold) {
                vector<float> box;
                box.push_back(bbout[position * 4 + 0] * bb_out_scale + j * 4);
                box.push_back(bbout[position * 4 + 1] * bb_out_scale + i * 4);
                box.push_back(bbout[position * 4 + 2] * bb_out_scale + j * 4);
                box.push_back(bbout[position * 4 + 3] * bb_out_scale + i * 4);
                box.push_back(pred[position * 2 + 1]);
                boxes.push_back(box);
            }
        }
    }
    return boxes;
}

float overlap(float x1, float x2, float x3, float x4) {
    float left = max(x1, x3);
    float right = min(x2, x4);
    return max(right - left, 0.0f);
}

float box_c(const vector<float>& box1, const vector<float>& box2) {
    float center_x1 = (box1[0] + box1[2]) / 2;
    float center_y1 = (box1[1] + box1[3]) / 2;
    float center_x2 = (box2[0] + box2[2]) / 2;
    float center_y2 = (box2[1] + box2[3]) / 2;
    float dx = center_x1 - center_x2;
    float dy = center_y1 - center_y2;
    return dx * dx + dy * dy;
}

float cal_iou(const vector<float>& box1, const vector<float>& box2) {
    float w = overlap(box1[0], box1[2], box2[0], box2[2]);
    float h = overlap(box1[1], box1[3], box2[1], box2[3]);

    float inter_area = w * h;
    float box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = box1_area + box2_area - inter_area;
    
    float iou = inter_area / union_area;

    float c = box_c(box1, box2);
    if (c == 0) return iou;

    float d = (box1[0] - box2[0]) * (box1[0] - box2[0]) +
              (box1[1] - box2[1]) * (box1[1] - box2[1]);
    float u = pow(d / c, 0.6f);

    return max(iou - u, 0.0f);
}

void applyNMS(const vector<vector<float>>& boxes, const vector<float>& scores,
              const float nms_threshold, const float conf_threshold, vector<size_t>& res,
              int max_output) {
    const size_t count = boxes.size();
    vector<pair<float, size_t>> order;
    for (size_t i = 0; i < count; ++i) {
        float box_size = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
        order.push_back({box_size, i});
    }
    sort(order.begin(), order.end(),
         [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
           return ls.first > rs.first;
         });
            
    vector<bool> exist_box(count, true);

    for (size_t _i = 0; _i < count && res.size() < max_output; ++_i) {
        size_t i = order[_i].second;
        if (!exist_box[i]) continue;
        if (scores[i] < conf_threshold) {
            exist_box[i] = false;
            continue;
        }
        res.push_back(i);
        for (size_t _j = _i + 1; _j < count; ++_j) {
            size_t j = order[_j].second;
            if (!exist_box[j]) continue;
            float ovr = cal_iou(boxes[j], boxes[i]);
            if (ovr >= nms_threshold) exist_box[j] = false;
        }
    }
}

void runFacerecog(vart::Runner* runner, const Mat& frame, const FaceDetectResult& faceDetectResult) {
    vector<vector<float>> embedding_arr;
    vector<float> embedding_norm_arr;
    vector<string> embedding_class_arr;
    auto embeddings_npzpath = "/usr/share/vitis_ai_library/models/InceptionResnetV1/embeddings_xmodel.npz";
    tie(embedding_arr, embedding_norm_arr, embedding_class_arr) = load_embeddings(embeddings_npzpath);

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

    vector<float> mean = {128.0f, 128.0f, 128.0f};
    vector<float> scale = {0.0078125f, 0.0078125f, 0.0078125f};

    for (const auto &r : faceDetectResult.rects) {
        Mat cropped_img = frame(Rect(r.x * frame.cols, r.y * frame.rows,
                                     r.width * frame.cols, r.height * frame.rows));
        Mat resized_img;
        cv::resize(cropped_img, resized_img, Size(inWidth, inHeight));

        vector<int8_t> imageInputs(inSize * batchSize, 0);
        vector<int8_t> FCResult(batchSize * outSize, 0);

        NormalizeInputDataRGB(resized_img.data, inHeight, inWidth, 3, resized_img.step,
                              mean, scale, imageInputs.data());

        vector<unique_ptr<vart::TensorBuffer>> inputs, outputs;
        vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

        inputs.push_back(make_unique<CpuFlatTensorBuffer>(imageInputs.data(), inputTensors[0]));
        outputs.push_back(make_unique<CpuFlatTensorBuffer>(FCResult.data(), outputTensors[0]));
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.push_back(outputs[0].get());

        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        auto status = runner->wait(job_id.first, -1);
        if (status != 0) {
            cerr << "DPU execution failed with status " << status << endl;
            continue;
        }

        vector<float> float_output(outSize);
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

        string recognized_label = "Unknown";
        if (max_similarity_index != -1 && max_similarity > 0.6) {
            recognized_label = embedding_class_arr[max_similarity_index];
        }

        rectangle(frame, Rect(r.x * frame.cols, r.y * frame.rows,
                                       r.width * frame.cols, r.height * frame.rows),
                  Scalar(0, 255, 0), 2);
        putText(frame, recognized_label,
                Point(r.x * frame.cols, r.y * frame.rows - 10),
                FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);

        cout << "Recognized face: " << recognized_label << " (Similarity: " << max_similarity << ")" << endl;
    }
}

auto measureTime(const std::function<void()>& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <video_file> <fd_model_file> <fr_model_file>" << endl;
        return -1;
    }

    string video_path = argv[1];
    string fd_model_path = argv[2];
    string fr_model_path = argv[3];

    VideoCapture video(video_path);
    if (!video.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    auto graph_fd = xir::Graph::deserialize(fd_model_path);
    auto graph_fr = xir::Graph::deserialize(fr_model_path);

    auto subgraph_fd = get_dpu_subgraph(graph_fd.get());
    auto subgraph_fr = get_dpu_subgraph(graph_fr.get());

    CHECK_EQ(subgraph_fd.size(), 1u) << "Face detection should have one and only one DPU subgraph.";
    CHECK_EQ(subgraph_fr.size(), 1u) << "Face recognition should have one and only one DPU subgraph.";

    auto runner_fd = vart::Runner::create_runner(subgraph_fd[0], "run");
    auto runner_fr = vart::Runner::create_runner(subgraph_fr[0], "run");

    Mat frame;
    int frame_count = 0;
    
    while (video.read(frame)) {
        cout << "Processing frame " << frame_count++ << endl;

        
        if (!video.read(frame)) {
            std::cout << "End of video stream" << std::endl;
            return;
        }
        
        FaceDetectResult faceDetectResult = runFacedetect(runner_fd.get(), frame);
        runFacerecog(runner_fr.get(), frame, faceDetectResult);

        imshow("Video Processing", frame);
        if (waitKey(30) >= 0) break;
    }

    video.release();
    destroyAllWindows();

    return 0;
}
