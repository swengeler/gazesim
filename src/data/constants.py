# NOTE: currently, these stem from splits/split004_info.json and are computed over all
#       frames that have drone measurements and drone_control_frame_mean_gt available
STATISTICS = {
    "mean": {
        "screen": [0.23185605229941816, 0.20987627008239895, 0.21252105159994594]
    },
    "std": {
        "screen": [0.14304103712954377, 0.1309625291794035, 0.14716040743971653]
    }
}

HIGH_LEVEL_COMMAND_LABEL = {
    "flat_left_half": 0,
    "flat_right_half": 1,
    "wave_left_half": 2,
    "wave_right_half": 3,
    "flat_none": 4,
    "wave_none": 4
}
