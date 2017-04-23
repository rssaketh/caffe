#ifndef CAFFE_RECURSIVE_CONV_LAYER_HPP_
#define CAFFE_RECURSIVE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    
    template <typename Dtype>
    class RecursiveConvLayer : public Layer<Dtype> {
    public:
        explicit RecursiveConvLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
        
        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline bool EqualNumBottomTopBlobs() const { return true; }
        virtual inline const char* type() const { return "RecursiveConv"; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        
        void forward_activation_func_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        void backward_activation_func_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
        void permute_blobs_cpu(const vector<Blob<Dtype>*>& bottom, vector<int> new_orders, bool permute_diffs);
        void forward_ReLU_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        void forward_BN_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top, const int iter);
        void backward_BN_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom, const int iter);
        void backward_ReLU_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
        
        void forward_activation_func_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        void backward_activation_func_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
        void permute_blobs_gpu(const vector<Blob<Dtype>*>& bottom, bool channel_last, bool permute_diffs);
        void forward_ReLU_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        void forward_BN_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top, const int iter);
        void backward_BN_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom, const int iter);
        void backward_ReLU_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
        
        // Temporary (for debugging)
        void test_print(const int M, const int N, Dtype* A);
        void test_QR();
        void test_inverse_QR_case1();
        void test_inverse_QR_case2();
        void test_inverse_QR_case3();
        void test_inverse_QR_case4();
        
        // Buffer blobs
        Blob<Dtype> wt_buffer_; // C_ x C_ buffer used in Cayley transform
        Blob<Dtype> eye_; // Identity matrix C_ x C_
        Blob<Dtype> wt_inv_; // Copy of inverse of weights
        Blob<Dtype> mid_;
        Blob<Dtype> bn_mu_; // local BN mean: Nrec x No 
        Blob<Dtype> bn_sigma_; // local BN sigma: Nrec x No
        Blob<Dtype> temp_bn_sum_; // cache for backward BN: 1 x No 
        Blob<Dtype> batch_sum_multiplier_; //All ones size of 1x (N_*H_*W_)
        Blob<int> permute_order_;
        Blob<int> inv_permute_order_;
        Blob<int> old_steps_;
        Blob<int> new_steps_;
        // For cusolverDN
        Blob<Dtype> tau_; // C_ x 1 buffer used for QR factorization
        Blob<Dtype> workspace_; // buffer on GPU memory for cusolver functions
        int Lwork_; //size of workspace_
        Blob<int> dev_info_; // integer on gpu memory, describes if QR, SVD, etc was successful
        
        
        //Other useful member vars
        int N_;
        int C_;
        int H_;
        int W_;
        int batch_size_; // Effective batch size : B = N*H*W
        int Nrec_; // # of recursive convolutions
        int Nwts_; // # of unique weights (<= Nrec_)
        vector<int> rand_wt_order_; // Ordering of weights
        vector<int> old_mid_shape_;
        vector<int> new_mid_shape_;
        // For Batch-Norm
        bool use_global_stats_;
        Dtype moving_average_fraction_;
        Dtype eps_;
        Dtype bias_correction_factor_;
        Dtype inv_batch_size_;
        // For ReLU
        Dtype negative_slope_;
        Dtype inv_negative_slope_;

    };  

} // namespace caffe

#endif  // CAFFE_RECURSIVE_CONV_LAYER_HPP_
