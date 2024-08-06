// Include necessary headers
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <chrono>
#include <atomic>
#include <iomanip>

using namespace std;
using namespace cv;

// Global variables
VideoCapture video;  // OpenCV video capture object
mutex mtx_display_queue;  // Mutex for thread-safe access to the display queue
atomic<bool> is_reading(true);  // Flag to control the reading thread
atomic<bool> is_displaying(true);  // Flag to control the displaying thread

// Custom comparator for the priority queue to sort frames by index
struct FrameComparator {
    bool operator()(const pair<int, pair<long long, Mat>>& a, const pair<int, pair<long long, Mat>>& b) {
        return a.first > b.first;  // Sort by frame index
    }
};

// Priority queue to store frames for display
priority_queue<pair<int, pair<long long, Mat>>, vector<pair<int, pair<long long, Mat>>>, FrameComparator> display_queue;

// Function to read frames from the video
void Read() {
    int frame_index = 0;
    auto start_time = chrono::steady_clock::now();
    
    while (is_reading) {
        auto loop_start = chrono::high_resolution_clock::now();

        Mat frame;
        double readTime = 0;
        
        // Measure video read time
        auto read_start = chrono::high_resolution_clock::now();
        bool read_success = video.read(frame);
        auto read_end = chrono::high_resolution_clock::now();
        readTime = chrono::duration_cast<chrono::microseconds>(read_end - read_start).count() / 1000.0;

        if (!read_success) {
            cout << "End of video." << endl;
            is_reading = false;
            break;
        }

        auto current_time = chrono::steady_clock::now();
        auto timestamp = chrono::duration_cast<chrono::milliseconds>(current_time - start_time);
        
        double queueWaitTime = 0, pushTime = 0;

        // Measure queue wait time (if queue is full)
        auto queue_start = chrono::high_resolution_clock::now();
        while (display_queue.size() >= 30 && is_reading) {
            this_thread::sleep_for(chrono::milliseconds(10));
        }
        auto queue_end = chrono::high_resolution_clock::now();
        queueWaitTime = chrono::duration_cast<chrono::microseconds>(queue_end - queue_start).count() / 1000.0;

        // Measure frame push time
        auto push_start = chrono::high_resolution_clock::now();
        mtx_display_queue.lock();
        display_queue.push(make_pair(frame_index++, make_pair(timestamp.count(), frame.clone())));
        mtx_display_queue.unlock();
        auto push_end = chrono::high_resolution_clock::now();
        pushTime = chrono::duration_cast<chrono::microseconds>(push_end - push_start).count() / 1000.0;

        auto loop_end = chrono::high_resolution_clock::now();
        double totalLoopTime = chrono::duration_cast<chrono::microseconds>(loop_end - loop_start).count() / 1000.0;

        // Output timing information
        cout << fixed << setprecision(3)
             << "Read timings (ms) - "
             << "Read: " << readTime
             << ", Queue wait: " << queueWaitTime
             << ", Push: " << pushTime
             << ", Total loop: " << totalLoopTime
             << endl;
    }
}

// Function to display frames
void Display(double original_fps) {
    namedWindow("Input Video", WINDOW_AUTOSIZE);

    int frame_count = 0;
    double fps = 0;
    auto start_time = chrono::steady_clock::now();
    long long last_timestamp = 0;

    while (is_displaying) {
        auto loop_start = chrono::high_resolution_clock::now();

        auto current_time = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(current_time - start_time).count();

        pair<int, pair<long long, Mat>> current_frame;
        double queueWaitTime = 0, popTime = 0;

        // Measure queue wait and pop time
        auto queue_start = chrono::high_resolution_clock::now();
        mtx_display_queue.lock();
        while (display_queue.empty() && is_reading) {
            mtx_display_queue.unlock();
            this_thread::sleep_for(chrono::milliseconds(1));
            mtx_display_queue.lock();
        }
        auto queue_end = chrono::high_resolution_clock::now();
        queueWaitTime = chrono::duration_cast<chrono::microseconds>(queue_end - queue_start).count() / 1000.0;

        if (!display_queue.empty()) {
            auto pop_start = chrono::high_resolution_clock::now();
            current_frame = display_queue.top();
            display_queue.pop();
            auto pop_end = chrono::high_resolution_clock::now();
            popTime = chrono::duration_cast<chrono::microseconds>(pop_end - pop_start).count() / 1000.0;
        }
        mtx_display_queue.unlock();

        if (display_queue.empty() && !is_reading) {
            break;
        }

        // Calculate FPS
        frame_count++;
        auto display_duration = chrono::duration_cast<chrono::seconds>(current_time - start_time);
        
        if (display_duration.count() >= 1) {
            fps = frame_count / static_cast<double>(display_duration.count());
            frame_count = 0;
            start_time = current_time;
        }

        double cloneTime = 0, textTime = 0, imshowTime = 0;

        // Measure clone time
        auto clone_start = chrono::high_resolution_clock::now();
        Mat display_frame = current_frame.second.second.clone();
        auto clone_end = chrono::high_resolution_clock::now();
        cloneTime = chrono::duration_cast<chrono::microseconds>(clone_end - clone_start).count() / 1000.0;

        // Measure text addition time
        auto text_start = chrono::high_resolution_clock::now();
        putText(display_frame, "FPS: " + to_string(fps) + " / " + to_string(original_fps), 
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        auto text_end = chrono::high_resolution_clock::now();
        textTime = chrono::duration_cast<chrono::microseconds>(text_end - text_start).count() / 1000.0;

        // Measure imshow time
        auto imshow_start = chrono::high_resolution_clock::now();
        imshow("Input Video", display_frame);
        auto imshow_end = chrono::high_resolution_clock::now();
        imshowTime = chrono::duration_cast<chrono::microseconds>(imshow_end - imshow_start).count() / 1000.0;

        int delay = max(1, static_cast<int>(1000 / original_fps - (elapsed - last_timestamp)));
        last_timestamp = elapsed;
        
        // Check for user input to exit
        auto wait_start = chrono::high_resolution_clock::now();
        int key = waitKey(1);
        auto wait_end = chrono::high_resolution_clock::now();
        double waitTime = chrono::duration_cast<chrono::microseconds>(wait_end - wait_start).count() / 1000.0;

        if (key == 'q' || key == 27) {  // 'q' or ESC key
            is_displaying = false;
            is_reading = false;
            break;
        }

        auto loop_end = chrono::high_resolution_clock::now();
        double totalLoopTime = chrono::duration_cast<chrono::microseconds>(loop_end - loop_start).count() / 1000.0;

        // Output timing information
        cout << fixed << setprecision(3)
             << "Display timings (ms) - "
             << "Queue wait: " << queueWaitTime
             << ", Pop: " << popTime
             << ", Clone: " << cloneTime
             << ", Text: " << textTime
             << ", Imshow: " << imshowTime
             << ", Wait: " << waitTime
             << ", Total loop: " << totalLoopTime
             << endl;
    }

    destroyAllWindows();
    cout << "Display finished" << endl;
}

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc != 2) {
        cout << "Usage: ./video_player [video_file]" << endl;
        return -1;
    }

    string file_name = argv[1];
    cout << "Opening video: " << file_name << endl;
    video.open(file_name);
    if (!video.isOpened()) {
        cout << "Failed to open video: " << file_name << endl;
        return -1;
    }

    // Get original video FPS
    double original_fps = video.get(cv::CAP_PROP_FPS);
    cout << "Original video FPS: " << original_fps << endl;

    // Start read and display threads
    thread read_thread(Read);
    thread display_thread(Display, original_fps);

    // Wait for threads to finish
    read_thread.join();
    display_thread.join();

    // Clean up
    video.release();
    cout << "Program finished." << endl;

    return 0;
}
