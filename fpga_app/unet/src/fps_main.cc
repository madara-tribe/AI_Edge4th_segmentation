#include <assert.h>
#include <algorithm>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic> //DB
#include <sys/stat.h>
#include <unistd.h> //DB
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip> //DB
#include <queue>
#include <string>
#include <vector>
#include <thread> //DB
#include <opencv2/opencv.hpp>
#include <dnndk/dnndk.h>

#include <mutex>  //DB
std::mutex mtx_;

using namespace std;
using namespace std::chrono;
using namespace cv;


// constants for segmentation network
#define KERNEL_CONV       "unet"
#define CONV_INPUT_NODE   "conv2d_1_convolution"
#define CONV_OUTPUT_NODE  "separable_conv2d_1_separable_conv2d"
#define IMAGEDIR "./workspace/"
#define OUTPUT_DIR "./output/"
#define DPU_MODE_NORMAL 0
#define DPU_MODE_PROF   1
#define DPU_MODE_DUMP   2
#define CLS 5
#define IMG_HEIGHT 304
#define IMG_WIGHT 480
#define THREADS 2
#define BLOCK_SIZE 5
// colors for segmented classes (that are 19, as in COCO)
uint8_t colorB[] = {0, 0, 255, 255, 69};
uint8_t colorG[] = {0, 0, 255, 0, 47};
uint8_t colorR[] = {0, 255, 0, 0, 142};

int image_num;
/*List all images's name in path.*/
std::string output_filenames;
std::vector<string> img_filenames;

#define SLEEP 1
int t_cnt = 0;
void barrier(int tid){
    {
        std::lock_guard<std::mutex> lock(mtx_);
        t_cnt++;
    }
    while(1){
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if(t_cnt % THREADS == 0) break;
        }
        usleep(SLEEP);
    }
}


vector<string> ListImages(const char *path) {
  vector<string> images;
  images.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path, &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path);
    exit(1);
  }

  DIR *dir = opendir(path);
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path);
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
  sort(images.begin(), images.end());
  for (size_t i=0; i<images.size(); i++){
    cout << "names in dir  = "<< images[i] << endl;
  }
  return images;
}


inline double etime_sum(timespec ts02, timespec ts01){
    return (ts02.tv_sec+(double)ts02.tv_nsec/(double)1000000000)
            - (ts01.tv_sec+(double)ts01.tv_nsec/(double)1000000000);
}


void PostProc(const float *data, int height, int width, int depth){
  assert(data);
  Mat segMat(height, width, CV_8UC3);
  for (int r=0; r<height; r++) { //loop over rows
    for (int c=0; c<width; c++) { //loop over columns
      float pix_val[CLS];
      for (int d=0; d<depth; d++) { //loop over depth (classes)
            int i = d*width*height+r*width+c;
            pix_val[d]= data[i];
      }
     auto max_ind = max_element(pix_val, pix_val + CLS);
     int posit = distance(pix_val, max_ind);
     segMat.at<Vec3b>(r, c) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);
    }
  }
  cv::imwrite(OUTPUT_DIR + output_filenames, segMat);
}



/* Normalize Images and leave them as BGR instead of RGB */
int8_t normalize_and_quantize(int pixel, int i, int j, int k, float scale)
{
    int8_t q_pixel;

    if ((i < IMG_HEIGHT) && (j < IMG_WIGHT))
        q_pixel = pixel / 255.0 * scale;
    else
        q_pixel = 0;
    return q_pixel;
}


void setInputImage(DPUTask *task, const char *inNode, const cv::Mat &image)
{
    DPUTensor *in = dpuGetInputTensor(task, inNode);
    float scale = dpuGetTensorScale(in);
    int w = dpuGetTensorWidth(in);
    int h = dpuGetTensorHeight(in);
    int c = 3;
    int8_t *data = dpuGetTensorAddress(in);
    image.forEach<Vec3b>([&](Vec3b &p, const int pos[2]) -> void {
        int start = pos[0] * w * c + pos[1] * c;
        for (int k = 0; k < 3; k++)
            data[start + k] = normalize_and_quantize(p[0], pos[0], pos[1], k, scale);
    });
}


int main_thread(DPUKernel *kernelConv, int s_num, int e_num, int tid){
  assert(kernelConv);
  /*Load all image names */
  DPUTask *task = dpuCreateTask(kernelConv, DPU_MODE_NORMAL); 

  struct timespec ts01, ts02, ts03, tt01, tt02;
  double sum1 = 0, sum2 = 0, sum3 = 0, sumt = 0;

  string image_file_name[BLOCK_SIZE];
  cv::Mat input_image[BLOCK_SIZE];

  DPUTensor *conv_out_tensor = dpuGetOutputTensor(task, CONV_OUTPUT_NODE);
  int outHeight = dpuGetTensorHeight(conv_out_tensor);
  int outWidth  = dpuGetTensorWidth(conv_out_tensor);
  int outChannel= CLS;
  int outSize = dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE);
  float outScale = dpuGetOutputTensorScale(task, CONV_OUTPUT_NODE);
  cout << "GET OUTPUT TENSOR: size  = " << outSize   << endl; //602112
  
  // Main Loop
  int cnt=0;
  for(cnt=s_num; cnt<=e_num; cnt+=BLOCK_SIZE){
      clock_gettime(CLOCK_REALTIME, &ts01);

      for(int i=0; i<BLOCK_SIZE;i++){
        if(cnt+i>e_num) break;
        image_file_name[i] = img_filenames[cnt+i];
        input_image[i] = cv::imread(IMAGEDIR+image_file_name[i]);
        if (input_image[i].empty()) {
            printf("cannot load %s\n", image_file_name[i].c_str());
            abort();
        }
      }
      
      barrier(tid);

      usleep(1000);
      clock_gettime(CLOCK_REALTIME, &ts02);
      sum1 += etime_sum(ts02,ts01);
      barrier(tid);

      for(int i=0; i<BLOCK_SIZE;i++){
        if(cnt+i>e_num) break;
        output_filenames = image_file_name[i];
        cout << "filename : " << image_file_name[i] << endl;
        // resize
        cv::Mat img;
        resize(input_image[i], img, Size(IMG_WIGHT, IMG_HEIGHT), INTER_NEAREST);
  
        int8_t *outAddr = (int8_t *)dpuGetOutputTensorAddress(task, CONV_OUTPUT_NODE);
        float *softmax = new float[outWidth*outHeight*outChannel];
        
        // Set image into Conv Task with mean value
        setInputImage(task, CONV_INPUT_NODE, img);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            dpuRunTask(task);
            // Calculate softmax on DPU 
            dpuRunSoftmax(outAddr, softmax, outChannel,outSize/outChannel, outScale);
        }
        // Post process
        PostProc(softmax, outHeight, outWidth, outChannel);

        delete[] softmax;
        cout << " delete[] softmax; " << endl;
        double tmp_time = etime_sum(tt02, tt01);
        sumt += tmp_time;
      }

      barrier(tid);
      clock_gettime(CLOCK_REALTIME, &ts03);
      sum2 += etime_sum(ts03, ts02);
      sum3 += etime_sum(ts03, ts01);
  }
  dpuDestroyTask(task);
  printf("sum1 : loaded images time : %8.3lf[s]\n", sum1);
  printf("sum2 : proproc dpu postproc time: %8.3lf[s]\n", sum2);
  printf("FPS        : %8.3lf (%8.3lf [ms])\n", (float)image_num/sum2, (float)sum2/image_num*1000);   

  int tmp = image_num%(THREADS*BLOCK_SIZE);
  //printf("%d %d\n", tid, tmp);
  if(tid >= tmp){
      usleep(SLEEP);
      barrier(tid);
      usleep(SLEEP);
      barrier(tid);
      usleep(SLEEP);
      barrier(tid);
   }
  
   return 0;
}


int main(int argc, char **argv){
  // DPU Kernels/Tasks for runing SSD
  DPUKernel *kernelConv;
  // Check args
  cout << "now running " << argv[0] << endl;
  img_filenames = ListImages(IMAGEDIR);
  if (img_filenames.size() == 0) {
    cout << "\nError: Not images exist in " << IMAGEDIR << endl;
  } else {
    image_num = img_filenames.size();
    cout << "total image : " << img_filenames.size() << endl;
  }
  
  int th_srt[THREADS];
  int th_end[THREADS];
  th_srt[0] = 0;
  th_end[0] = image_num / THREADS;
  if((image_num%THREADS)==0) {
      th_end[0]--;
  }
  for(int i=1;i<THREADS;i++){
      th_srt[i] = th_end[i-1]+1;
      th_end[i] = th_srt[i]+(image_num / THREADS);
      if(i>=(image_num%THREADS)){
          th_end[i]--;
      }
  }

  for(int i=0;i<THREADS;i++){
      printf("th_srt[%d] = %d, th_end[%d] = %d\n", i, th_srt[i], i, th_end[i]);
  }
  // Attach to DPU driver and prepare for runing
  dpuOpen();
  kernelConv = dpuLoadKernel(KERNEL_CONV);
  // Parallel processing
  vector<thread> ths;
  for (int i = 1; i < THREADS; i++){
      ths.emplace_back(thread(main_thread, kernelConv, th_srt[i], th_end[i], i));
  }
  main_thread(kernelConv, th_srt[0], th_end[0], 0);

  for (auto& th: ths){
      th.join();
  }
  //
  dpuDestroyKernel(kernelConv);
  dpuClose();
  cout << "\nFinished ..." << endl;

  return 0;
}