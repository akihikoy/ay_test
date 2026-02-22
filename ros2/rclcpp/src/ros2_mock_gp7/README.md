# ros2_mock_gp7

ros2_control test with mock_components/GenericSystem and motoman_gp7_support (motoman_ros2_support_packages ).

# Build

```bash
$ colcon build --symlink-install
```

# Run

```bash
$ ros2 launch ros2_mock_gp7 mock_gp7.launch.py

$ $ ./test_gp7_mock.py
[INFO] [1769424784.164791018] [mock_tester]: 初期化完了
[INFO] [1769424784.165202143] [mock_tester]: 現在値待機中...
[INFO] [1769424784.165899476] [mock_tester]: Start: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[INFO] [1769424784.166233099] [mock_tester]: 送信中...
[INFO] [1769424784.167359893] [mock_tester]: 実行中...
[INFO] [1769424790.174840835] [mock_tester]: 完了: 結果コード 0
```
