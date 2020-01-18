/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "extra_tree_learner.h"

#include <LightGBM/utils/array_args.h>

#include <queue>

#include "cost_effective_gradient_boosting.hpp"
#include "feature_histogram.hpp"

namespace LightGBM {

#ifdef TIMETAG
std::chrono::duration<double, std::milli> init_train_time;
std::chrono::duration<double, std::milli> init_split_time;
std::chrono::duration<double, std::milli> find_split_time;
std::chrono::duration<double, std::milli> split_time;
#endif  // TIMETAG

ExtraTreeLearner::ExtraTreeLearner(const Config* config)
  : SerialTreeLearner(config) {
  rand_threshold_ = Random(config_->seed);
}

ExtraTreeLearner::~ExtraTreeLearner() {
  #ifdef TIMETAG
  Log::Info("ExtraTreeLearner::init_train costs %f", init_train_time * 1e-3);
  Log::Info("ExtraTreeLearner::init_split costs %f", init_split_time * 1e-3);
  Log::Info("ExtraTreeLearner::find_split costs %f", find_split_time * 1e-3);
  Log::Info("ExtraTreeLearner::split costs %f", split_time * 1e-3);
  #endif
}

void ExtraTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();
  is_constant_hessian_ = is_constant_hessian;

  // push split information for all leaves
  best_split_per_leaf_.resize(config_->num_leaves);

  // initialize splits for leaf
  smaller_leaf_splits_.reset(new LeafSplits(train_data_->num_data()));
  larger_leaf_splits_.reset(new LeafSplits(train_data_->num_data()));

  // initialize data partition
  data_partition_.reset(new DataPartition(num_data_, config_->num_leaves));
  is_feature_used_.resize(num_features_);
  valid_feature_indices_ = train_data_->ValidFeatureIndices();
  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  Log::Info("Number of data points in the train set: %d, number of used features: %d", num_data_, num_features_);
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    cegb_.reset(new CostEfficientGradientBoosting(this));
    cegb_->Init();
  }
}

void ExtraTreeLearner::ResetTrainingData(const Dataset* train_data) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  CHECK(num_features_ == train_data_->num_features());

  // initialize splits for leaf
  smaller_leaf_splits_->ResetNumData(num_data_);
  larger_leaf_splits_->ResetNumData(num_data_);

  // initialize data partition
  data_partition_->ResetNumData(num_data_);

  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);

  if (cegb_ != nullptr) {
    cegb_->Init();
  }
}

void ExtraTreeLearner::ResetConfig(const Config* config) {
  if (config_->num_leaves != config->num_leaves) {
    config_ = config;
    // push split information for all leaves
    best_split_per_leaf_.resize(config_->num_leaves);
    data_partition_->ResetLeaves(config_->num_leaves);
  } else {
    config_ = config;
  }
  histogram_pool_.ResetConfig(config_);
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    cegb_.reset(new CostEfficientGradientBoosting(this));
    cegb_->Init();
  }
}

void ExtraTreeLearner::BeforeTrain() {
  // reset histogram pool
  histogram_pool_.ResetMap();

  if (config_->feature_fraction < 1.0f) {
    is_feature_used_ = GetUsedFeatures(true);
  } else {
    #pragma omp parallel for schedule(static, 512) if (num_features_ >= 1024)
    for (int i = 0; i < num_features_; ++i) {
      is_feature_used_[i] = 1;
    }
  }

  // initialize data partition
  data_partition_->Init();

  // reset the splits for leaves
  for (int i = 0; i < config_->num_leaves; ++i) {
    best_split_per_leaf_[i].Reset();
  }

  // Sumup for root
  if (data_partition_->leaf_count(0) == num_data_) {
    // use all data
    smaller_leaf_splits_->Init(gradients_, hessians_);

  } else {
    // use bagging, only use part of data
    smaller_leaf_splits_->Init(0, data_partition_.get(), gradients_, hessians_);
  }

  larger_leaf_splits_->Init();
}

bool ExtraTreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  // check depth of current leaf
  if (config_->max_depth > 0) {
    // only need to check left leaf, since right leaf is in same level of left leaf
    if (tree->leaf_depth(left_leaf) >= config_->max_depth) {
      best_split_per_leaf_[left_leaf].gain = kMinScore;
      if (right_leaf >= 0) {
        best_split_per_leaf_[right_leaf].gain = kMinScore;
      }
      return false;
    }
  }
  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
  // no enough data to continue
  if (num_data_in_right_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)
      && num_data_in_left_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)) {
    best_split_per_leaf_[left_leaf].gain = kMinScore;
    if (right_leaf >= 0) {
      best_split_per_leaf_[right_leaf].gain = kMinScore;
    }
    return false;
  }
   return true;
}

void ExtraTreeLearner::GatherInfoForThreshold(int feature_index, const uint32_t threshold, bool default_left,
                                              const double min_gain_shift, const data_size_t num_data, const data_size_t* data_indices, SplitInfo *output)  {
  train_data_->GetSplitInfo(feature_index, &threshold, 1, default_left, data_indices, num_data,
                            ordered_gradients_.data(), ordered_hessians_.data(), output);
  double curr_gain = FeatureHistogram::GetSplitGains(output->left_sum_gradient, output->left_sum_hessian,
                                                     output->right_sum_gradient, output->right_sum_hessian, 
                                                     config_->lambda_l1, config_->lambda_l2, config_->max_delta_step, 
                                                     output->min_constraint, output->max_constraint,
                                                     train_data_->FeatureMonotone(feature_index));
  if (curr_gain > min_gain_shift) {
    output->left_output = FeatureHistogram::CalculateSplittedLeafOutput(output->left_sum_gradient, output->left_sum_hessian, config_->lambda_l1,
                                                                        config_->lambda_l2, config_->max_delta_step);
    output->right_output = FeatureHistogram::CalculateSplittedLeafOutput(output->right_sum_gradient, output->right_sum_hessian, config_->lambda_l1,
                                                                         config_->lambda_l2, config_->max_delta_step);
    output->gain = curr_gain - min_gain_shift;
    output->monotone_type = train_data_->FeatureMonotone(feature_index);
  }
}

void ExtraTreeLearner::FindBestSplits() {
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  std::vector<int8_t> smaller_node_used_features(num_features_, 1);
  std::vector<int8_t> larger_node_used_features(num_features_, 1);
  std::vector<SplitInfo> smaller_best(num_threads_);
  std::vector<SplitInfo> larger_best(num_threads_);
  if (config_->feature_fraction_bynode < 1.0f) {
    smaller_node_used_features = GetUsedFeatures(false);
    larger_node_used_features = GetUsedFeatures(false);
  }
  int dummy;
  const data_size_t* data_indices = data_partition_->GetIndexOnLeaf(smaller_leaf_splits_->LeafIndex(), &dummy);
  data_size_t num_data = smaller_leaf_splits_->num_data_in_leaf();
  train_data_->ConstructOrderedBins(data_indices, num_data, gradients_, hessians_, ordered_gradients_.data(), ordered_hessians_.data(),
                                    is_constant_hessian_);
  double gain_shift = FeatureHistogram::GetLeafSplitGain(smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
                                                         config_->lambda_l1, config_->lambda_l2, config_->max_delta_step);
  double min_gain_shift = gain_shift + config_->min_gain_to_split;
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_used_[feature_index]) { continue; }
    // smaller split
    const int tid = omp_get_thread_num();
    if (!is_feature_used_[feature_index]) { continue; }
    int max_threshold;
    MissingType missing_type = train_data_->FeatureBinMapper(feature_index)->missing_type();
    int num_bin = train_data_->FeatureBinMapper(feature_index)->num_bin();
    if (missing_type == MissingType::NaN) {
      max_threshold = num_bin - 3;
    } else {
      max_threshold = num_bin - 2;
    }
    int threshold = 0;
    if (max_threshold > 0) {
      threshold = rand_threshold_.NextInt(0, max_threshold);
    }
    bool default_left = true;
    BinType bin_type = train_data_->FeatureBinMapper(feature_index)->bin_type();
    if (bin_type == BinType::NumericalBin) {
      default_left = static_cast<bool>(rand_threshold_.NextInt(0, 1));
    } else {
      default_left = true;
    }
    SplitInfo smaller_split;
    GatherInfoForThreshold(feature_index, threshold, default_left, min_gain_shift, num_data, data_indices, &smaller_split);
    int real_fidx = train_data_->RealFeatureIndex(feature_index);
    smaller_split.feature = real_fidx;
    smaller_split.default_left = default_left;
    if (bin_type == CategoricalBin) {
      smaller_split.cat_threshold = std::vector<uint32_t>(1, threshold);
      smaller_split.num_cat_threshold = 1;
    } else {
      smaller_split.threshold = threshold;
    }
    if (cegb_ != nullptr) {
      smaller_split.gain -= cegb_->DetlaGain(feature_index, real_fidx, smaller_leaf_splits_->LeafIndex(), smaller_leaf_splits_->num_data_in_leaf(), smaller_split);
    }
    if (smaller_split > smaller_best[tid] && smaller_node_used_features[feature_index]) {
      smaller_best[tid] = smaller_split;
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  // only has root leaf
  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0) {
    data_indices = data_partition_->GetIndexOnLeaf(larger_leaf_splits_->LeafIndex(), &dummy);
    num_data = larger_leaf_splits_->num_data_in_leaf();
    train_data_->ConstructOrderedBins(data_indices, num_data, gradients_, hessians_, ordered_gradients_.data(),
                                      ordered_hessians_.data(), is_constant_hessian_);
    gain_shift = FeatureHistogram::GetLeafSplitGain(larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
                                                           config_->lambda_l1, config_->lambda_l2, config_->max_delta_step);
    min_gain_shift = gain_shift + config_->min_gain_to_split;
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
      OMP_LOOP_EX_BEGIN();
      if (!is_feature_used_[feature_index]) { continue; }
      const int tid = omp_get_thread_num();
      int max_threshold;
      MissingType missing_type = train_data_->FeatureBinMapper(feature_index)->missing_type();
      int num_bin = train_data_->FeatureBinMapper(feature_index)->num_bin();
      if (missing_type == MissingType::NaN) {
        max_threshold = num_bin - 3;
      } else {
        max_threshold = num_bin - 2;
      }
      int threshold = 0;
      if (max_threshold > 0) {
        threshold = rand_threshold_.NextInt(0, max_threshold);
      }
      bool default_left = true;
      BinType bin_type = train_data_->FeatureBinMapper(feature_index)->bin_type();
      if (bin_type == BinType::NumericalBin) {
        default_left = static_cast<bool>(rand_threshold_.NextInt(0, 1));
      } else {
        default_left = true;
      }
      // larger split
      SplitInfo larger_split;
      GatherInfoForThreshold(feature_index, threshold, default_left, min_gain_shift, num_data, data_indices, &larger_split);
      int real_fidx = train_data_->RealFeatureIndex(feature_index);
      larger_split.feature = real_fidx;
      larger_split.default_left = default_left;
      if (bin_type == CategoricalBin) {
        larger_split.cat_threshold = std::vector<uint32_t>(1, threshold);
        larger_split.num_cat_threshold = 1;
      } else {
        larger_split.threshold = threshold;
      }
      if (cegb_ != nullptr) {
        larger_split.gain -= cegb_->DetlaGain(feature_index, real_fidx, larger_leaf_splits_->LeafIndex(), larger_leaf_splits_->num_data_in_leaf(), larger_split);
      }
      if (larger_split > larger_best[tid] && larger_node_used_features[feature_index]) {
        larger_best[tid] = larger_split;
      }
    OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_best);
  int leaf = smaller_leaf_splits_->LeafIndex();
  best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx];
  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0) {
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_best);
    leaf = larger_leaf_splits_->LeafIndex();
    best_split_per_leaf_[leaf] = larger_best[larger_best_idx];
  }
  #ifdef TIMETAG
  find_split_time += std::chrono::steady_clock::now() - start_time;
  #endif
}

int32_t ExtraTreeLearner::ForceSplits(Tree* tree, const Json& forced_split_json, int* left_leaf,
                                       int* right_leaf, int *cur_depth,
                                       bool *aborted_last_force_split) {
  int32_t result_count = 0;
  // start at root leaf
  *left_leaf = 0;
  std::queue<std::pair<Json, int>> q;
  Json left = forced_split_json;
  Json right;
  bool left_smaller = true;
  std::unordered_map<int, SplitInfo> forceSplitMap;
  q.push(std::make_pair(forced_split_json, *left_leaf));
  while (!q.empty()) {
    // before processing next node from queue, store info for current left/right leaf
    // store "best split" for left and right, even if they might be overwritten by forced split
    if (BeforeFindBestSplit(tree, *left_leaf, *right_leaf)) {
      FindBestSplits();
    }
    // then, compute own splits
    SplitInfo left_split;
    SplitInfo right_split;

    if (!left.is_null()) {
      const int left_feature = left["feature"].int_value();
      const double left_threshold_double = left["threshold"].number_value();
      const int left_inner_feature_index = train_data_->InnerFeatureIndex(left_feature);
      const uint32_t left_threshold = train_data_->BinThreshold(
              left_inner_feature_index, left_threshold_double);
      auto leaf_histogram_array = (left_smaller) ? smaller_leaf_histogram_array_ : larger_leaf_histogram_array_;
      auto left_leaf_splits = (left_smaller) ? smaller_leaf_splits_.get() : larger_leaf_splits_.get();
      leaf_histogram_array[left_inner_feature_index].GatherInfoForThreshold(
              left_leaf_splits->sum_gradients(),
              left_leaf_splits->sum_hessians(),
              left_threshold,
              left_leaf_splits->num_data_in_leaf(),
              &left_split);
      left_split.feature = left_feature;
      forceSplitMap[*left_leaf] = left_split;
      if (left_split.gain < 0) {
        forceSplitMap.erase(*left_leaf);
      }
    }

    if (!right.is_null()) {
      const int right_feature = right["feature"].int_value();
      const double right_threshold_double = right["threshold"].number_value();
      const int right_inner_feature_index = train_data_->InnerFeatureIndex(right_feature);
      const uint32_t right_threshold = train_data_->BinThreshold(
              right_inner_feature_index, right_threshold_double);
      auto leaf_histogram_array = (left_smaller) ? larger_leaf_histogram_array_ : smaller_leaf_histogram_array_;
      auto right_leaf_splits = (left_smaller) ? larger_leaf_splits_.get() : smaller_leaf_splits_.get();
      leaf_histogram_array[right_inner_feature_index].GatherInfoForThreshold(
        right_leaf_splits->sum_gradients(),
        right_leaf_splits->sum_hessians(),
        right_threshold,
        right_leaf_splits->num_data_in_leaf(),
        &right_split);
      right_split.feature = right_feature;
      forceSplitMap[*right_leaf] = right_split;
      if (right_split.gain < 0) {
        forceSplitMap.erase(*right_leaf);
      }
    }

    std::pair<Json, int> pair = q.front();
    q.pop();
    int current_leaf = pair.second;
    // split info should exist because searching in bfs fashion - should have added from parent
    if (forceSplitMap.find(current_leaf) == forceSplitMap.end()) {
        *aborted_last_force_split = true;
        break;
    }
    SplitInfo current_split_info = forceSplitMap[current_leaf];
    const int inner_feature_index = train_data_->InnerFeatureIndex(
            current_split_info.feature);
    auto threshold_double = train_data_->RealThreshold(
            inner_feature_index, current_split_info.threshold);

    // split tree, will return right leaf
    *left_leaf = current_leaf;
    if (train_data_->FeatureBinMapper(inner_feature_index)->bin_type() == BinType::NumericalBin) {
      *right_leaf = tree->Split(current_leaf,
                                inner_feature_index,
                                current_split_info.feature,
                                current_split_info.threshold,
                                threshold_double,
                                static_cast<double>(current_split_info.left_output),
                                static_cast<double>(current_split_info.right_output),
                                static_cast<data_size_t>(current_split_info.left_count),
                                static_cast<data_size_t>(current_split_info.right_count),
                                static_cast<double>(current_split_info.left_sum_hessian),
                                static_cast<double>(current_split_info.right_sum_hessian),
                                static_cast<float>(current_split_info.gain),
                                train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
                                current_split_info.default_left);
      data_partition_->Split(current_leaf, train_data_, inner_feature_index,
                             &current_split_info.threshold, 1,
                             current_split_info.default_left, *right_leaf);
    } else {
      std::vector<uint32_t> cat_bitset_inner = Common::ConstructBitset(
              current_split_info.cat_threshold.data(), current_split_info.num_cat_threshold);
      std::vector<int> threshold_int(current_split_info.num_cat_threshold);
      for (int i = 0; i < current_split_info.num_cat_threshold; ++i) {
        threshold_int[i] = static_cast<int>(train_data_->RealThreshold(
                    inner_feature_index, current_split_info.cat_threshold[i]));
      }
      std::vector<uint32_t> cat_bitset = Common::ConstructBitset(
              threshold_int.data(), current_split_info.num_cat_threshold);
      *right_leaf = tree->SplitCategorical(current_leaf,
                                           inner_feature_index,
                                           current_split_info.feature,
                                           cat_bitset_inner.data(),
                                           static_cast<int>(cat_bitset_inner.size()),
                                           cat_bitset.data(),
                                           static_cast<int>(cat_bitset.size()),
                                           static_cast<double>(current_split_info.left_output),
                                           static_cast<double>(current_split_info.right_output),
                                           static_cast<data_size_t>(current_split_info.left_count),
                                           static_cast<data_size_t>(current_split_info.right_count),
                                           static_cast<double>(current_split_info.left_sum_hessian),
                                           static_cast<double>(current_split_info.right_sum_hessian),
                                           static_cast<float>(current_split_info.gain),
                                           train_data_->FeatureBinMapper(inner_feature_index)->missing_type());
      data_partition_->Split(current_leaf, train_data_, inner_feature_index,
                             cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()),
                             current_split_info.default_left, *right_leaf);
    }

    if (current_split_info.left_count < current_split_info.right_count) {
      left_smaller = true;
      smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                 current_split_info.left_sum_gradient,
                                 current_split_info.left_sum_hessian);
      larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                current_split_info.right_sum_gradient,
                                current_split_info.right_sum_hessian);
    } else {
      left_smaller = false;
      smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                 current_split_info.right_sum_gradient, current_split_info.right_sum_hessian);
      larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                current_split_info.left_sum_gradient, current_split_info.left_sum_hessian);
    }

    left = Json();
    right = Json();
    if ((pair.first).object_items().count("left") > 0) {
      left = (pair.first)["left"];
      if (left.object_items().count("feature") > 0 && left.object_items().count("threshold") > 0) {
        q.push(std::make_pair(left, *left_leaf));
      }
    }
    if ((pair.first).object_items().count("right") > 0) {
      right = (pair.first)["right"];
      if (right.object_items().count("feature") > 0 && right.object_items().count("threshold") > 0) {
        q.push(std::make_pair(right, *right_leaf));
      }
    }
    result_count++;
    *(cur_depth) = std::max(*(cur_depth), tree->leaf_depth(*left_leaf));
  }
  return result_count;
}
}  // namespace LightGBM
