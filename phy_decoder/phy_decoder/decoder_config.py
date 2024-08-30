#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
config = {
    "mode": "test",
    "max_epochs": 50,
    "num_workers": 8,
    "batch_size": 64,
    "seq_length": 50,
    "use_rnn": True,
    "reset_hidden_each_epoch": True,
    "rnn_mode": "parallel",  # Choose 'full', or 'base' or 'parallel'
    "use_weight": True,
    "SEPARATE_DATASET_MODE": True,
    "input_type": "pro+exte",  # Choose 'hidden', 'pro','pro+exte', or 'all'or 'all-e'
    "overlapp": 25,
    "paradigm": None,  # "ordinal" or None
}
