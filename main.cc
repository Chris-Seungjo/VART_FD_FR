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

#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

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
void runFacerecog(vart::Runner* runner, const Mat& frame, const FaceDetectResult& faceDetectResult,
                  const vector<vector<float>>& embedding_arr,
                  const vector<float>& embedding_norm_arr,
                  const vector<string>& embedding_class_arr);

int initFramebufferOutput(const char* device) {
    int fb_fd = open(device, O_RDWR);
    if (fb_fd < 0) {
        perror("Failed to open framebuffer");
        return -1;
    }

    struct fb_var_screeninfo vinfo;
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo)) {
        perror("Failed to get framebuffer info");
        close(fb_fd);
        return -1;
    }

    // Ensure that the framebuffer matches the intended resolution and color depth
    vinfo.bits_per_pixel = 16;  // 16 bits per pixel for RGB565
    if (ioctl(fb_fd, FBIOPUT_VSCREENINFO, &vinfo)) {
        perror("Failed to set framebuffer info");
        close(fb_fd);
        return -1;
    }

    return fb_fd;
}

// Function to display an image on the framebuffer
void displayFramebuffer(int fb_fd, const cv::Mat& image) {
    struct fb_var_screeninfo vinfo;
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo)) {
        perror("Failed to get framebuffer info");
        return;
    }

    int screensize = vinfo.yres_virtual * vinfo.xres_virtual * vinfo.bits_per_pixel / 8;
    uint8_t* fb_data = (uint8_t*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (fb_data == MAP_FAILED) {
        perror("Failed to map framebuffer device to memory");
        return;
    }

    // Resize the input image to match the framebuffer resolution
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(vinfo.xres, vinfo.yres));

    // Convert the image to RGB565 format
    cv::Mat rgb565_image;
    cv::cvtColor(resized_image, rgb565_image, cv::COLOR_BGR2BGR565);

    // Copy the image data to the framebuffer
    memcpy(fb_data, rgb565_image.data, std::min(screensize, (int)(rgb565_image.total() * rgb565_image.elemSize())));

    munmap(fb_data, screensize);
}

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

    auto fd_resize_start = std::chrono::high_resolution_clock::now();
    Mat resized;
    cv::resize(frame, resized, Size(inWidth, inHeight));
    auto fd_resize_end = std::chrono::high_resolution_clock::now();
    auto fd_resize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fd_resize_end - fd_resize_start);
    std::cout << "fd_resize time: " << fd_resize_duration.count() << " ms" << std::endl;

    auto fd_setImageBGR_start = std::chrono::high_resolution_clock::now();
    setImageBGR({resized}, runner, imageInputs.data(), mean, scale);
    auto fd_setImageBGR_end = std::chrono::high_resolution_clock::now();
    auto fd_setImageBGR_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fd_setImageBGR_end - fd_setImageBGR_start);
    std::cout << "fd_setImageBGR time: " << fd_setImageBGR_duration.count() << " ms" << std::endl;


    auto fd_inf_start = std::chrono::high_resolution_clock::now();
    vector<unique_ptr<vart::TensorBuffer>> inputs, outputs;
    vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    
    inputs.push_back(make_unique<CpuFlatTensorBuffer>(imageInputs.data(), inputTensors[0]));
    outputs.push_back(make_unique<CpuFlatTensorBuffer>(FCResult.data(), outputTensors[0]));
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    auto status = runner->wait(job_id.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";

    auto fd_inf_end = std::chrono::high_resolution_clock::now();
    auto fd_inf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fd_inf_end - fd_inf_start);
    std::cout << "fd_inf time: " << fd_inf_duration.count() << " ms" << std::endl;

    const float det_threshold = 0.9f;
    const float nms_threshold = 0.3f;
    
    auto fd_post_start = std::chrono::high_resolution_clock::now();
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

    auto fd_post_end = std::chrono::high_resolution_clock::now();
    auto fd_post_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fd_post_end - fd_post_start);
    std::cout << "fd_post time: " << fd_post_duration.count() << " ms" << std::endl;

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

void runFacerecog(vart::Runner* runner, const Mat& frame, const FaceDetectResult& faceDetectResult,
                  const vector<vector<float>>& embedding_arr,
                  const vector<float>& embedding_norm_arr,
                  const vector<string>& embedding_class_arr) {
    auto fr_init_start = std::chrono::high_resolution_clock::now();

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

    auto fr_init_end = std::chrono::high_resolution_clock::now();
    auto fr_init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fr_init_end - fr_init_start);
    std::cout << "fr_init time: " << fr_init_duration.count() << " ms" << std::endl;

    for (const auto &r : faceDetectResult.rects) {
        auto fr_pre_start = std::chrono::high_resolution_clock::now();

        Mat cropped_img = frame(Rect(r.x * frame.cols, r.y * frame.rows,
                                     r.width * frame.cols, r.height * frame.rows));
        Mat resized_img;
        cv::resize(cropped_img, resized_img, Size(inWidth, inHeight));

        vector<int8_t> imageInputs(inSize * batchSize, 0);
        vector<int8_t> FCResult(batchSize * outSize, 0);

        NormalizeInputDataRGB(resized_img.data, inHeight, inWidth, 3, resized_img.step,
                              mean, scale, imageInputs.data());

        auto fr_pre_end = std::chrono::high_resolution_clock::now();
        auto fr_pre_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fr_pre_end - fr_pre_start);
        std::cout << "fr_pre time: " << fr_pre_duration.count() << " ms" << std::endl;

        auto fr_inf_start = std::chrono::high_resolution_clock::now();

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
        auto fr_inf_end = std::chrono::high_resolution_clock::now();
        auto fr_inf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fr_inf_end - fr_inf_start);
        std::cout << "fr_inf time: " << fr_inf_duration.count() << " ms" << std::endl;

        auto fr_post_start = std::chrono::high_resolution_clock::now();

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

        auto fr_post_end = std::chrono::high_resolution_clock::now();
        auto fr_post_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fr_post_end - fr_post_start);
        std::cout << "fr_post time: " << fr_post_duration.count() << " ms" << std::endl;
    }
}

auto measureTime(const std::function<void()>& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main(int argc, char* argv[]) {
    auto main_init_start = std::chrono::high_resolution_clock::now();
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <video_file> <fd_model_file> <fr_model_file>" << endl;
        return -1;
    }

    argv[1] = "/home/root/VART/video_fd_fr/all5_720p.mp4";
    argv[2] = "/usr/share/vitis_ai_library/models/densebox_640_360/densebox_640_360.xmodel";
    argv[3] = "/usr/share/vitis_ai_library/models/InceptionResnetV1/InceptionResnetV1.xmodel";

    string video_path = argv[1];
    string fd_model_path = argv[2];
    string fr_model_path = argv[3];

    VideoCapture video(video_path);
    if (!video.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    int fb_fd = initFramebufferOutput("/dev/fb0");  // Adjust as needed
    if (fb_fd < 0) {
        std::cerr << "Failed to initialize framebuffer output" << std::endl;
        return -1;
    }

    vector<vector<float>> embedding_arr;
    vector<float> embedding_norm_arr;
    vector<string> embedding_class_arr;
    auto embeddings_npzpath = "/usr/share/vitis_ai_library/models/InceptionResnetV1/embeddings_xmodel.npz";
    tie(embedding_arr, embedding_norm_arr, embedding_class_arr) = load_embeddings(embeddings_npzpath);

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
    
    auto main_init_end = std::chrono::high_resolution_clock::now();
    auto main_init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(main_init_end - main_init_start);
    std::cout << "main_init time: " << main_init_duration.count() << " ms" << std::endl;

    while (true) {
        cout << "Processing frame " << frame_count++ << endl;

        //read
        auto read_start = std::chrono::high_resolution_clock::now();
        if (!video.read(frame)) {
            std::cout << "End of video stream" << std::endl;
            return -1;
        }
        auto read_end = std::chrono::high_resolution_clock::now();
        auto read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start);
        std::cout << "Read time: " << read_duration.count() << " ms" << std::endl;

        //fd
        auto fd_start = std::chrono::high_resolution_clock::now();
        FaceDetectResult faceDetectResult = runFacedetect(runner_fd.get(), frame);
        auto fd_end = std::chrono::high_resolution_clock::now();
        auto fd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fd_end - fd_start);
        std::cout << "fd time: " << fd_duration.count() << " ms" << std::endl;

        //fr
        auto fr_start = std::chrono::high_resolution_clock::now();
        runFacerecog(runner_fr.get(), frame, faceDetectResult, embedding_arr, embedding_norm_arr, embedding_class_arr);
        auto fr_end = std::chrono::high_resolution_clock::now();
        auto fr_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fr_end - fr_start);
        std::cout << "fr time: " << fr_duration.count() << " ms" << std::endl;

        auto display_hdmi_start = std::chrono::high_resolution_clock::now();
        displayFramebuffer(fb_fd, frame);
        auto display_hdmi_end = std::chrono::high_resolution_clock::now();
        auto display_hdmi_duration = std::chrono::duration_cast<std::chrono::milliseconds>(display_hdmi_end - display_hdmi_start);
        std::cout << "display_hdmi time: " << display_hdmi_duration.count() << " ms" << std::endl;

    }

    video.release();
    close(fb_fd);

    return 0;
}
