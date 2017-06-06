#include <vector>

#include "caffe/layers/custom_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void replace_elements(const int nhwc, Dtype* in, int middle, int value) {
  int j =  blockIdx.x*blockDim.x + middle;
  if (j < nhwc)
    in[j] = value;
}

template <typename Dtype>
__global__ void normalize_kernel(const int nhwc, Dtype* in, int num_threads) { 
  if (threadIdx.x == 0) { 
    int sum=0;
    for (int i=0;i<num_threads;++i)
      sum+= in[blockIdx.x*blockDim.x + i];

    for (int j=0;j < num_threads;++j)
      in[j] = in[j]/sum;
  
  }
}
  



template <typename Dtype>
void CustomConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

//Code added starts here
  //Dtype* weight_dum = this->blobs_[0]->mutable_gpu_data();
  int n = this->blobs_[0]->shape(0),k=this->blobs_[0]->shape(1),
      h = this->blobs_[0]->shape(2),w = this->blobs_[0]->shape(3);
  
  replace_elements<Dtype><<<n*k, w*h>>>(n*k*w*h,this->blobs_[0]->mutable_gpu_data(),(w*h)/2,0);
  normalize_kernel<Dtype><<<n*k, w*h>>>(n*k*w*h,this->blobs_[0]->mutable_gpu_data(),w*h);
  replace_elements<Dtype><<<n*k, w*h>>>(n*k*w*h,this->blobs_[0]->mutable_gpu_data(),(w*h)/2,-1);



//Code added ends here
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void CustomConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CustomConvolutionLayer);

}  // namespace caffe
