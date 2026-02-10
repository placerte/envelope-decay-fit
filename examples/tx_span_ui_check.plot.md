# Plot Report: tx_span_ui_check

## Context
- schema_version: 1.0
- run_id: None
- timestamp: None
- git_hash: None
- inputs: None
- parameters: None
- style: name=None rcparams_hash=66132a6156c9be7c48dfc67d2c148a68a2c1e5d4887205007c121326c2ecd51e

## PNG
![](tx_span_ui_check.png)
- png_path: tx_span_ui_check.png

## Warnings
- None

## Invariant Results
- None

## Axes

### Axis 0
- title: Span-based Tx measurement (fn=120.0 Hz)
- xlabel: Time (s)
- ylabel: Envelope (dB re max)
- xscale: linear
- yscale: linear
- xlim: [np.float64(-0.07), np.float64(1.47)]
- ylim: [np.float64(-24.856448), np.float64(1.18364)]
- legend: None
- series:

#### Series ax0_s0_Envelope__dB_
- label: Envelope (dB)
- kind: line2d
- stats:
  - n: 600
  - x_min/x_max: 0.0 / 1.4
  - y_min/y_max: -23.672807 / 0.0
  - first/last: [0.0, 1.4] / [0.0, -22.204893]
- diagnostics:
  - n: 600
  - non_finite_count: 0
  - dx_min: 0.002337
  - dx_median: 0.002337
  - dx_non_positive_count: 0
  - dy_max_abs: 0.163782
  - dy_robust_z_max: 3.441828
  - dy_outlier_count: 0
  - slope_mean: -15.860638
  - slope_median: -16.159553
  - slope_std: 21.392702
  - slope_p10: -41.648887
  - slope_p90: 10.865289
  - wiggle_sign_changes: 9
  - curvature_std: 0.0037
  - curvature_max_abs: 0.012786
  - curvature_sign_changes: 15
- data_sample:
  - decimation: original_n=600 max_points=2000 method=None decimated=False
  - head: [(0.0, 0.0), (0.002337, -0.028442), (0.004674, -0.056851), (0.007012, -0.085287), (0.009349, -0.113812)]
  - tail: [(1.390651, -22.669695), (1.392988, -22.554236), (1.395326, -22.437722), (1.397663, -22.321002), (1.4, -22.204893)]

#### Series ax0_s1_series_1
- label: series_1
- kind: line2d
- stats:
  - n: 2
  - x_min/x_max: -0.07 / -0.07
  - y_min/y_max: 0.0 / 1.0
  - first/last: [-0.07, -0.07] / [0.0, 1.0]
- diagnostics:
  - n: 2
  - non_finite_count: 0
  - dx_min: 0.0
  - dx_median: 0.0
  - dx_non_positive_count: 1
  - dy_max_abs: 1.0
  - dy_robust_z_max: 0.0
  - dy_outlier_count: 0
  - slope_mean: None
  - slope_median: None
  - slope_std: None
  - slope_p10: None
  - slope_p90: None
  - wiggle_sign_changes: 0
  - curvature_std: None
  - curvature_max_abs: None
  - curvature_sign_changes: 0
- data_sample:
  - decimation: original_n=2 max_points=2000 method=None decimated=False
  - head: [(-0.07, 0.0), (-0.07, 1.0)]
  - tail: [(-0.07, 0.0), (-0.07, 1.0)]

#### Series ax0_s2_series_2
- label: series_2
- kind: line2d
- stats:
  - n: 2
  - x_min/x_max: 1.47 / 1.47
  - y_min/y_max: 0.0 / 1.0
  - first/last: [1.47, 1.47] / [0.0, 1.0]
- diagnostics:
  - n: 2
  - non_finite_count: 0
  - dx_min: 0.0
  - dx_median: 0.0
  - dx_non_positive_count: 1
  - dy_max_abs: 1.0
  - dy_robust_z_max: 0.0
  - dy_outlier_count: 0
  - slope_mean: None
  - slope_median: None
  - slope_std: None
  - slope_p10: None
  - slope_p90: None
  - wiggle_sign_changes: 0
  - curvature_std: None
  - curvature_max_abs: None
  - curvature_sign_changes: 0
- data_sample:
  - decimation: original_n=2 max_points=2000 method=None decimated=False
  - head: [(1.47, 0.0), (1.47, 1.0)]
  - tail: [(1.47, 0.0), (1.47, 1.0)]

#### Series ax0_s3_series_3
- label: series_3
- kind: line2d
- stats:
  - n: 2
  - x_min/x_max: 0.0 / 1.4
  - y_min/y_max: -10.0 / 0.0
  - first/last: [0.0, 1.4] / [0.0, -10.0]
- diagnostics:
  - n: 2
  - non_finite_count: 0
  - dx_min: 1.4
  - dx_median: 1.4
  - dx_non_positive_count: 0
  - dy_max_abs: 10.0
  - dy_robust_z_max: 0.0
  - dy_outlier_count: 0
  - slope_mean: -7.142857
  - slope_median: -7.142857
  - slope_std: 0.0
  - slope_p10: -7.142857
  - slope_p90: -7.142857
  - wiggle_sign_changes: 0
  - curvature_std: None
  - curvature_max_abs: None
  - curvature_sign_changes: 0
- data_sample:
  - decimation: original_n=2 max_points=2000 method=None decimated=False
  - head: [(0.0, 0.0), (1.4, -10.0)]
  - tail: [(0.0, 0.0), (1.4, -10.0)]

## Raw JSON (minus points)
```json
{
  "axes": [
    {
      "index": 0,
      "legend": [],
      "series": [
        {
          "data": {
            "decimation": {
              "decimated": false,
              "max_points": 2000,
              "method": null,
              "n_original": 600
            },
            "n": 600,
            "omitted": true,
            "sample": {
              "head": [
                [
                  0.0,
                  0.0
                ],
                [
                  0.002337,
                  -0.028442
                ],
                [
                  0.004674,
                  -0.056851
                ],
                [
                  0.007012,
                  -0.085287
                ],
                [
                  0.009349,
                  -0.113812
                ]
              ],
              "tail": [
                [
                  1.390651,
                  -22.669695
                ],
                [
                  1.392988,
                  -22.554236
                ],
                [
                  1.395326,
                  -22.437722
                ],
                [
                  1.397663,
                  -22.321002
                ],
                [
                  1.4,
                  -22.204893
                ]
              ]
            }
          },
          "diagnostics": {
            "curvature_max_abs": 0.012786,
            "curvature_sign_changes": 15,
            "curvature_std": 0.0037,
            "dx_median": 0.002337,
            "dx_min": 0.002337,
            "dx_non_positive_count": 0,
            "dy_max_abs": 0.163782,
            "dy_outlier_count": 0,
            "dy_robust_z_max": 3.441828,
            "n": 600,
            "non_finite_count": 0,
            "slope_mean": -15.860638,
            "slope_median": -16.159553,
            "slope_p10": -41.648887,
            "slope_p90": 10.865289,
            "slope_std": 21.392702,
            "wiggle_sign_changes": 9
          },
          "id": "ax0_s0_Envelope__dB_",
          "kind": "line2d",
          "label": "Envelope (dB)",
          "stats": {
            "n": 600,
            "x_endpoints": [
              0.0,
              1.4
            ],
            "x_max": 1.4,
            "x_min": 0.0,
            "y_endpoints": [
              0.0,
              -22.204893
            ],
            "y_max": 0.0,
            "y_min": -23.672807
          }
        },
        {
          "data": {
            "decimation": {
              "decimated": false,
              "max_points": 2000,
              "method": null,
              "n_original": 2
            },
            "n": 2,
            "omitted": true,
            "sample": {
              "head": [
                [
                  -0.07,
                  0.0
                ],
                [
                  -0.07,
                  1.0
                ]
              ],
              "tail": [
                [
                  -0.07,
                  0.0
                ],
                [
                  -0.07,
                  1.0
                ]
              ]
            }
          },
          "diagnostics": {
            "curvature_max_abs": null,
            "curvature_sign_changes": 0,
            "curvature_std": null,
            "dx_median": 0.0,
            "dx_min": 0.0,
            "dx_non_positive_count": 1,
            "dy_max_abs": 1.0,
            "dy_outlier_count": 0,
            "dy_robust_z_max": 0.0,
            "n": 2,
            "non_finite_count": 0,
            "slope_mean": null,
            "slope_median": null,
            "slope_p10": null,
            "slope_p90": null,
            "slope_std": null,
            "wiggle_sign_changes": 0
          },
          "id": "ax0_s1_series_1",
          "kind": "line2d",
          "label": "series_1",
          "stats": {
            "n": 2,
            "x_endpoints": [
              -0.07,
              -0.07
            ],
            "x_max": -0.07,
            "x_min": -0.07,
            "y_endpoints": [
              0.0,
              1.0
            ],
            "y_max": 1.0,
            "y_min": 0.0
          }
        },
        {
          "data": {
            "decimation": {
              "decimated": false,
              "max_points": 2000,
              "method": null,
              "n_original": 2
            },
            "n": 2,
            "omitted": true,
            "sample": {
              "head": [
                [
                  1.47,
                  0.0
                ],
                [
                  1.47,
                  1.0
                ]
              ],
              "tail": [
                [
                  1.47,
                  0.0
                ],
                [
                  1.47,
                  1.0
                ]
              ]
            }
          },
          "diagnostics": {
            "curvature_max_abs": null,
            "curvature_sign_changes": 0,
            "curvature_std": null,
            "dx_median": 0.0,
            "dx_min": 0.0,
            "dx_non_positive_count": 1,
            "dy_max_abs": 1.0,
            "dy_outlier_count": 0,
            "dy_robust_z_max": 0.0,
            "n": 2,
            "non_finite_count": 0,
            "slope_mean": null,
            "slope_median": null,
            "slope_p10": null,
            "slope_p90": null,
            "slope_std": null,
            "wiggle_sign_changes": 0
          },
          "id": "ax0_s2_series_2",
          "kind": "line2d",
          "label": "series_2",
          "stats": {
            "n": 2,
            "x_endpoints": [
              1.47,
              1.47
            ],
            "x_max": 1.47,
            "x_min": 1.47,
            "y_endpoints": [
              0.0,
              1.0
            ],
            "y_max": 1.0,
            "y_min": 0.0
          }
        },
        {
          "data": {
            "decimation": {
              "decimated": false,
              "max_points": 2000,
              "method": null,
              "n_original": 2
            },
            "n": 2,
            "omitted": true,
            "sample": {
              "head": [
                [
                  0.0,
                  0.0
                ],
                [
                  1.4,
                  -10.0
                ]
              ],
              "tail": [
                [
                  0.0,
                  0.0
                ],
                [
                  1.4,
                  -10.0
                ]
              ]
            }
          },
          "diagnostics": {
            "curvature_max_abs": null,
            "curvature_sign_changes": 0,
            "curvature_std": null,
            "dx_median": 1.4,
            "dx_min": 1.4,
            "dx_non_positive_count": 0,
            "dy_max_abs": 10.0,
            "dy_outlier_count": 0,
            "dy_robust_z_max": 0.0,
            "n": 2,
            "non_finite_count": 0,
            "slope_mean": -7.142857,
            "slope_median": -7.142857,
            "slope_p10": -7.142857,
            "slope_p90": -7.142857,
            "slope_std": 0.0,
            "wiggle_sign_changes": 0
          },
          "id": "ax0_s3_series_3",
          "kind": "line2d",
          "label": "series_3",
          "stats": {
            "n": 2,
            "x_endpoints": [
              0.0,
              1.4
            ],
            "x_max": 1.4,
            "x_min": 0.0,
            "y_endpoints": [
              0.0,
              -10.0
            ],
            "y_max": 0.0,
            "y_min": -10.0
          }
        }
      ],
      "title": "Span-based Tx measurement (fn=120.0 Hz)",
      "xlabel": "Time (s)",
      "xlim": [
        -0.07,
        1.47
      ],
      "xscale": "linear",
      "ylabel": "Envelope (dB re max)",
      "ylim": [
        -24.856448,
        1.18364
      ],
      "yscale": "linear"
    }
  ],
  "context": {
    "Tx_active": "T10",
    "Tx_value": 10.0,
    "fn_hz": 120.0,
    "plot_kind": "tx_span",
    "style": {
      "name": null,
      "rcparams_hash": "66132a6156c9be7c48dfc67d2c148a68a2c1e5d4887205007c121326c2ecd51e"
    },
    "t0": 0.0,
    "t1": 1.4
  },
  "figure": {
    "backend": "Agg",
    "dpi": 150.0,
    "size_inches": [
      12.0,
      6.0
    ]
  },
  "invariants": [],
  "schema_version": "1.0",
  "warnings": []
}
```
