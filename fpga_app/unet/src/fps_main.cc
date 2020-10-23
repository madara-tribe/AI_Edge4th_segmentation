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
#include <mutex>  //DB
#include <string>
#include <vector>
#include <thread> //DB
#include <opencv2/opencv.hpp>

// Header files for DNNDK APIs
#include <dnndk/dnndk.h>

#define IMG_DIR "./img_test/"
#define OUTPUT_DIR "./output/"
#define IMAGE_WIDTH 224
#define IMAGE_HEIGHT 224
#define NAME_TXT "./test_name.txt"
#define BUFSIZE 7
#define IMG_NUM 649
#define CLS 5

using namespace std;
using namespace std::chrono;
using namespace cv;

/* ****************************************************************************************** */
int threadnum;
std::vector<string> img_filenames;
vector<string> kinds, images; //DB
const auto for_resize = cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT);
// constants for segmentation network
#define KERNEL_CONV       "unet_0"
#define CONV_INPUT_NODE   "conv2d_1_convolution"
#define CONV_OUTPUT_NODE  "conv2d_24_convolution"

// colors for segmented classes (that are 19, as in COCO)
uint8_t colorB[] = {0, 255, 69, 0, 255};
uint8_t colorG[] = {0, 0, 47, 0, 255};
uint8_t colorR[] = {0, 0, 142, 255, 0};

// comparison algorithm for priority_queue
class Compare {
    public:
    bool operator()(const pair<int, Mat> &n1, const pair<int, Mat> &n2) const {
        return n1.first > n2.first;
    }
};

/* ****************************************************************************************** */
//#define SHOWTIME

#ifdef SHOWTIME
#define _T(func)                                                              \
    {                                                                         \
        auto _start = system_clock::now();                                    \
        func;                                                                 \
        auto _end = system_clock::now();                                      \
        auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
        string tmp = #func;                                                   \
        tmp = tmp.substr(0, tmp.find('('));                                   \
        cout << "[TimeTest]" << left << setw(30) << tmp;                      \
        cout << left << setw(10) << duration << "us" << endl;                 \
    }
#else
#define _T(func) func;
#endif

/* ****************************************************************************************** */

/*List all images's name in path.*/
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
        std::string name = entry->d_name;
        std::string ext = name.substr(name.find_last_of(".") + 1);
        if ((ext == "jpg") || (ext == "png")) {
            images.push_back(name);
    }
   }
  }

  closedir(dir);
  sort(images.begin(), images.end());
  return images;
}


/* Calculate softmax.*/
void CPUCalcSoftmax(const float *data, int height, int width, int depth, float *result)
{
  assert(data && result);
  FILE* fp;
  ifstream ifs(NAME_TXT, ios::in);
  string tmp;
  cout << "sofmax pass" << endl;
  float *local_res = new float [height*width*depth];
  for (int r=0; r<height; r++) { //loop over rows
    for (int c=0; c<width; c++) { //loop over columns
      double sum = 0.0f;
      for (int d=0; d<depth; d++) { //loop over depth (classes)
	local_res[d] = exp(data[d*width*height+r*width +c]);
	sum=sum+local_res[d];
      }
      for (int d=0; d<depth; d++) { //loop over depth (classes)
	local_res[d] = local_res[d]/sum;
	result[d*width*height+r*width+c] = local_res[d];
      }
    }
  }
  
  Mat segMat(width, height, CV_8UC3);
  for (int col = 0; col < height; col++) {
    for (int row = 0; row < width; row++) {
      int i = col * IMAGE_HEIGHT * CLS + row * CLS;
      auto max_ind = max_element(result + i, result + i + CLS);
      int pos = *max_ind;
      //int pos = distance(result + i, max_ind);
      segMat.at<Vec3b>(col, row) = Vec3b(colorB[pos], colorG[pos], colorR[pos]);
    }
  }
  if(getline(ifs, tmp)){
    cv::imwrite(OUTPUT_DIR+tmp, segMat);
  };
  ifs.close();
  delete[] local_res;
}

/* Normalize Images and leave them as BGR instead of RGB */
void normalize_image(const Mat& image, int8_t* data, float scale, float* mean) {
  for(int i = 0; i < 3; ++i) {
    for(int j = 0; j < image.rows; ++j) {
      for(int k = 0; k < image.cols; ++k) {
	//data[j*image.rows*3+k*3+2-i] = (float(image.at<Vec3b>(j,k)[i])/127.5 -1.0 ) * scale; //from BGR to RGB
	data[j*image.cols*3+k*3+i] = (float(image.at<Vec3b>(j,k)[i])/255)* scale;
      }
     }
   }
}


inline void set_input_image(DPUTask *task, const string& input_node, const cv::Mat& image, float* mean)
{
  //Mat cropped_img;
  DPUTensor* dpu_in = dpuGetInputTensor(task, input_node.c_str());
  float scale       = dpuGetTensorScale(dpu_in);
  int width         = dpuGetTensorWidth(dpu_in);
  int height        = dpuGetTensorHeight(dpu_in);
  int size          = dpuGetTensorSize(dpu_in);
  int8_t* data      = dpuGetTensorAddress(dpu_in);

    //cout << "SET INPUT IMAGE: scale = " << scale  << endl; //64
    //cout << "SET INPUT IMAGE: width = " << width  << endl; //224
    //cout << "SET INPUT IMAGE: height= " << height << endl; //224
    //cout << "SET INPUT IMAGE: size  = " << size   << endl; //150528
  cv::Mat img;
  cv::resize(image, img, for_resize, cv::INTER_NEAREST);
  normalize_image(img, data, scale, mean);
}


void run_CNN(DPUTask *taskConv, Mat img)
{
  assert(taskConv);
  
  // Get info from the the output Tensor
  DPUTensor *conv_out_tensor = dpuGetOutputTensor(taskConv, CONV_OUTPUT_NODE);
  int outHeight = dpuGetTensorHeight(conv_out_tensor);
  int outWidth  = dpuGetTensorWidth(conv_out_tensor);
  int outChannel= dpuGetOutputTensorChannel(taskConv, CONV_OUTPUT_NODE);
  int outSize   = dpuGetTensorSize(conv_out_tensor);
    //cout << "GET OUTPUT TENSOR: chan =  " << outChannel<< endl; //12
    //cout << "GET OUTPUT TENSOR: width = " << outWidth  << endl; //224
    //cout << "GET OUTPUT TENSOR: height= " << outHeight << endl; //224
    //cout << "GET OUTPUT TENSOR: size  = " << outSize   << endl; //602112
  cout << "\npass step1" << endl;
  float *softmax   = new float[outWidth*outHeight*outChannel];
  float *outTensor = new float[outWidth*outHeight*outChannel];

  float mean[3] = {0.0f, 0.0f, 0.0f};
  cout << "\npass step2" << endl;
  // Set image into Conv Task with mean value
  set_input_image(taskConv, CONV_INPUT_NODE, img, mean);

  //cout << "\nRun MNIST CONV ..." << endl;
  _T(dpuRunTask(taskConv));

  // Get output tensor result and convert from INT8 to FP32 format
  _T(dpuGetOutputTensorInHWCFP32(taskConv, CONV_OUTPUT_NODE, outTensor, outChannel));

  // Calculate softmax on CPU
  _T(CPUCalcSoftmax(outTensor, outHeight, outWidth, outChannel, softmax));


  delete[] softmax;
  delete[] outTensor;

}



/**
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param task - pointer to Segmentation Task
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runSegmentation(DPUKernel *kernelConv)
{
  /*Load all image names */
  cv::Mat input_image[IMG_NUM];

  img_filenames = ListImages(IMG_DIR);
   
#define DPU_MODE_NORMAL 0
#define DPU_MODE_PROF   1
#define DPU_MODE_DUMP   2

  thread workers[threadnum];
  auto _start = system_clock::now();

  for (auto i = 0; i < threadnum; i++)
  {
  workers[i] = thread([&,i]()
  {

    /* Create DPU Tasks for CONV  */
    DPUTask *taskConv = dpuCreateTask(kernelConv, DPU_MODE_NORMAL); // profiling not enabled
    //DPUTask *taskConv = dpuCreateTask(kernelConv, DPU_MODE_PROF); // profiling enabled
    //enable profiling
    //int res1 = dpuEnableTaskProfile(taskConv);
    //if (res1!=0) printf("ERROR IN ENABLING TASK PROFILING FOR CONV KERNEL\n");

    for(unsigned int ind = i  ;ind < images.size();ind+=threadnum)
      {
        input_image[ind] = cv::imread(IMG_DIR+img_filenames[ind], IMREAD_UNCHANGED);
        if (input_image[i].empty()){
            printf("cannot load %s%s\n", IMG_DIR, img_filenames[ind].c_str());
            abort();
        };
        cout << "DBG imread " << input_image[ind] << endl;
        run_CNN(taskConv, input_image[ind]);
      }
    // Destroy DPU Tasks & free resources
    dpuDestroyTask(taskConv);
  });
  }

  // Release thread resources.
  for (auto &w : workers) {
    if (w.joinable()) w.join();
  }

  auto _end = system_clock::now();
  auto duration = (duration_cast<microseconds>(_end - _start)).count();
  cout << "[Time]" << duration << "us" << endl;
  cout << "[FPS]" << images.size()*1000000.0/duration  << endl;
}

/**
 * @brief Entry for runing Segmentation neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char **argv)
{

  // DPU Kernels/Tasks for runing SSD
  DPUKernel *kernelConv;

  // Check args
  if(argc == 2) {
    threadnum = stoi(argv[1]);
    cout << "now running " << argv[0] << " " << argv[1] << endl;
  }
  else
      cout << "now running " << argv[0] << endl;

  // Attach to DPU driver and prepare for runing
  dpuOpen();

  // Create DPU Kernels and Tasks for CONV Nodes in FCN8
  kernelConv = dpuLoadKernel(KERNEL_CONV);

  /* run FCN8 Semantic Segmentation */
  runSegmentation(kernelConv);

  // Destroy DPU T Kernels and free resources
  dpuDestroyKernel(kernelConv);

  // Detach from DPU driver and release resources
  dpuClose();
  
  return 0;
}
