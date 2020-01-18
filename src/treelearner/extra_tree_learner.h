/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_EXTRA_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_EXTRA_TREE_LEARNER_H_

#include "serial_tree_learner.h"

using namespace json11;

namespace LightGBM {

class ExtraTreeLearner: public SerialTreeLearner {
public:
  friend CostEfficientGradientBoosting;

  explicit ExtraTreeLearner(const Config* config);

  ~ExtraTreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data) override;

  void ResetConfig(const Config* config) override;

  void BeforeTrain() override;

  bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) override;

  void FindBestSplits() override;

  int32_t ForceSplits(Tree* tree, const Json& forced_split_json, int* left_leaf, int* right_leaf, int *cur_depth,
                   bool *aborted_last_force_split) override;

  void ExtraTreeLearner::GatherInfoForThreshold(int feature_index, const uint32_t threshold, bool default_left,
                                                const double min_gain_shift, const data_size_t num_data, const data_size_t* data_indices,
                                                SplitInfo *output);

private:
  /*! \brief random generator for split thresholds */
  Random rand_threshold_;
};
}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_EXTRA_TREE_LEARNER_H_
