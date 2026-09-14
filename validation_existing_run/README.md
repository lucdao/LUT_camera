# Polynomial both-stages experiment

Thử nghiệm này thay cả hai tầng: Polynomial bậc 2 cho từng cặp PTZ->Wide và Polynomial robust cho Wide->grid.
Tầng grid->pan/tilt/zoom vẫn dùng pose surface hiện có. Không điều khiển camera.

- Local Polynomial degree: 2.
- Global Polynomial selected degree: 2.
- `local_pair_comparison.csv`: so sánh tâm từ H cũ và Polynomial cho từng PTZ.
- `polynomial_both_stages_mapping.json`: đầy đủ hệ số và chất lượng mô hình.
- `diagnostics/local_h_centres_vs_polynomial_centres.jpg`: cam = tâm H, xanh = tâm Polynomial.
- `diagnostics/both_stages_polynomial_grid_target_vs_prediction.jpg`: grid thật và dự đoán.
