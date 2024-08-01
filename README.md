'Read' is responsible for reading frames from video input
'FD' is responsible for get appropriate bounding box coordinates
'FR' is responsible for recognize the label of detected person, draw bounding box and frames on the frame and store it to  display_queue
'Display' is responsible for displaying result frame
'main' function leverages 4 threads (Read, FD, FR, Display) for now.
