#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/image_seg_data_layer.hpp"

namespace caffe {

template <typename Dtype>
ImageSegDataLayer<Dtype>::~ImageSegDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const bool is_label_color =
      this->layer_param_.image_data_param().is_label_color();
  const int label_type = this->layer_param_.image_data_param().label_type();
  CHECK(!(label_type == ImageDataParameter_LabelType_IMAGE && is_label_color))
      << "Label type: IMAGE is incompatible with is_label_color: true.\n"
      << "Label type: IMAGE refers to a grayscale label image "
      << "with all pixels equal to a specified scalar intensity.";
  const bool is_label_seg =
      this->layer_param_.image_data_param().is_label_seg();
  if (!is_label_seg) {
    CHECK(!this->layer_param_.image_data_param().has_ignore_label())
        << "ignore_label can only be supplied when is_label_seg is true";
  }
  string root_folder = this->layer_param_.image_data_param().root_folder();
  TransformationParameter transform_param =
      this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false)
      << "ImageSegDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0))
      << "Current implementation requires "
      << "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = "";
    if (label_type != ImageDataParameter_LabelType_NONE) {
      iss >> segfn;
    }
    lines_.push_back(std::make_pair(imgfn, segfn));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  int height, width;
  const int crop_size = transform_param.crop_size();
  if (crop_size > 0) {
    height = width = crop_size;
  } else {
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
    new_height, new_width, is_color);
    CHECK(cv_img.data) << "Fail to load input: "
        << root_folder + lines_[lines_id_].first;
    height = cv_img.rows;
    width = cv_img.cols;
  }
  // Obtain number of data/label channels.
  const int channels = is_color ? 3 : 1;
  const int label_channels = is_label_color ? 3 : 1;
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  // data
  const vector<int> data_shape{batch_size, channels, height, width};
  top[0]->Reshape(data_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(data_shape);
  }
  const vector<int> one_data_shape{1, channels, height, width};
  this->transformed_data_.Reshape(one_data_shape);
  // label
  const vector<int> label_shape{batch_size, label_channels, height, width};
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
  const vector<int> one_label_shape{1, label_channels, height, width};
  this->transformed_label_.Reshape(one_label_shape);
  // image dimensions, for each image, stores (img_height, img_width)
  const vector<int> data_dim_shape{batch_size, 1, 1, 2};
  top[2]->Reshape(data_dim_shape);
  this->prefetch_data_dim_.Reshape(data_dim_shape);
  // data
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  // image_dim
  LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
      << top[2]->channels() << "," << top[2]->height() << ","
      << top[2]->width();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageSegDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->transformed_data_.count());
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  Dtype* top_data_dim = this->prefetch_data_dim_.mutable_cpu_data();
  const int max_height = batch->data_.height();
  const int max_width  = batch->data_.width();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int label_type = this->layer_param_.image_data_param().label_type();
  const bool is_color  = image_data_param.is_color();
  const bool is_label_color = image_data_param.is_label_color();
  const bool is_label_seg = image_data_param.is_label_seg();
  const int ignore_label = is_label_seg ? image_data_param.ignore_label() : 0;
  string root_folder   = image_data_param.root_folder();
  const int lines_size = lines_.size();
  int top_data_dim_offset;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    top_data_dim_offset = this->prefetch_data_dim_.offset(item_id);
    std::vector<cv::Mat> cv_img_seg;
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    int img_row, img_col;
    cv_img_seg.push_back(ReadImageToCVMat(
        root_folder + lines_[lines_id_].first, new_height,
        new_width, is_color, &img_row, &img_col));
    top_data_dim[top_data_dim_offset] =
        static_cast<Dtype>(std::min(max_height, img_row));
    top_data_dim[top_data_dim_offset + 1] =
        static_cast<Dtype>(std::min(max_width, img_col));
    CHECK(cv_img_seg[0].data)
        << "Fail to load img: " << root_folder + lines_[lines_id_].first;
    if (label_type == ImageDataParameter_LabelType_PIXEL) {
      if (is_label_seg) {
        cv_img_seg.push_back(ReadImageToCVMatNearest(
            root_folder + lines_[lines_id_].second,
            new_height, new_width, is_label_color));
      } else {
        cv_img_seg.push_back(ReadImageToCVMat(
            root_folder + lines_[lines_id_].second,
            new_height, new_width, is_label_color));
      }
      CHECK(cv_img_seg[1].data)
        << "Fail to load seg: " << root_folder + lines_[lines_id_].second;
    } else if (label_type == ImageDataParameter_LabelType_IMAGE) {
      const int label = atoi(lines_[lines_id_].second.c_str());
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols,
                  CV_8UC1, cv::Scalar(label));
      cv_img_seg.push_back(seg);
    } else {
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols,
                  CV_8UC1, cv::Scalar(ignore_label));
      cv_img_seg.push_back(seg);
    }
    read_time += timer.MicroSeconds();
    timer.Start();
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    offset = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset);
    // Apply transformations (mirror, crop...) to the image
    this->data_transformer_->TransformImgAndSeg(cv_img_seg,
        &(this->transformed_data_), &(this->transformed_label_), ignore_label);
    trans_time += timer.MicroSeconds();
    // go to the next std::vector<int>::iterator iter;
    ++lines_id_;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegDataLayer);
REGISTER_LAYER_CLASS(ImageSegData);

}  // namespace caffe
#endif  // USE_OPENCV
