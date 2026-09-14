# RAY World — two-stage Polynomial PTZ mapping

Gói độc lập này nằm cạnh thư mục `RAY_WORLD` và chạy pipeline Wide → PTZ với
`Polynomial` ở cả hai tầng:

1. ALIKED + LightGlue + MAGSAC RANSAC tạo các match và lưu H cục bộ làm mốc
   hình học.
2. Với từng cặp PTZ–Wide, fit Polynomial bậc 2 robust từ các điểm match
   `PTZ pixel → Wide pixel`, sau đó chiếu tâm ảnh PTZ thành 30 tâm trên Wide.
3. Fit Polynomial robust lần hai từ 30 tâm Wide → tọa độ lưới `(column,row)`.
4. Dùng grid trạng thái đã ghi nhận để nội suy `(pan,tilt,zoom)` bằng affine
   hoặc PCHIP 2-D khi dữ liệu không tuyến tính.

Tầng 1 và tầng 2 đều được ghi hệ số, residual, inlier và chẩn đoán. H chỉ được
dùng ở bước match/RANSAC nền để kiểm tra và so sánh; mapping chạy cuối là
`Polynomial local + Polynomial global + grid pose surface`.

## Cài đặt

```powershell
cd C:\thuctap\RAY_WORLD_POLYNOMIAL_TWO_STAGE
python -m pip install -r requirements.txt
```

Không lưu mật khẩu trong mã nguồn hoặc metadata. Đặt bí mật trong session:

```powershell
$env:WIDE_RTSP_URL = "rtsp://..."
$env:PTZ_PASSWORD = "..."
# Nếu RTSP PTZ không theo mặc định, dùng thêm:
# $env:PTZ_RTSP_URL = "rtsp://..."
```

## Chạy toàn bộ: capture động + fit hai tầng

Collector hỏi giới hạn thật của camera bằng ONVIF
`GetConfigurationOptions/Absolute*PositionSpace`, mặc định tạo 30 mức pan ×
6 mức tilt = 180 ảnh. Không có khoảng tilt camera-specific hard-code. Chạy:

```powershell
python .\run_two_stage_pipeline.py `
  --capture `
  --continue-on-error
```

Pipeline sẽ tuần tự:

```text
00  Wide current + 8 Wide frames mới
01  PTZ dynamic grid 30x6; raw image + metadata raw/normalized
02  ALIKED + LightGlue + MAGSAC cho từng PTZ với từng Wide
03  Polynomial local PTZ→Wide + Polynomial global Wide→grid + PCHIP grid→pose
```

Trạng thái ONVIF thô được lưu ở `actual_*_raw`. Nếu thiếu, stale, ngoài range
hoặc lệch target quá tolerance, trường chuẩn `actual_*` dùng đúng target đã gửi
`AbsoluteMove`; lý do fallback cũng được lưu.

## Fit từ một run đã capture

Nếu đã có run hoàn chỉnh theo cấu trúc `00_wide_current_and_new_set` và
`01_ptz_overlap_grid`, không cần capture lại:

```powershell
python .\run_two_stage_pipeline.py --run-root C:\path\to\run_YYYYMMDD_HHMMSS
```

Run phải có đúng 30 cặp PTZ và đủ 6 hàng × 5 cột cho bộ fit hai tầng hiện tại.

## Dự đoán pose cho một pixel Wide

```powershell
python .\predict_wide_pixel_pose.py `
  --mapping .\run_...\03_polynomial_both_stages\polynomial_both_stages_mapping.json `
  --x 1280 --y 720
```

Lệnh này chỉ tính toán, không điều khiển camera.

## Điều khiển PTZ kiểm tra 30 điểm

Chỉ thêm `--execute` sau khi đã kiểm tra mapping:

```powershell
python .\run_two_stage_pipeline.py `
  --run-root .\run_... `
  --execute `
  --continue-on-error
```

Lệnh này điều khiển camera thật, chụp 30 ảnh raw và tạo preview có tâm PTZ.
Có thể gộp capture, fit và execute bằng cách thêm cả `--capture --execute`.

## Cấu trúc output

```text
run_.../
  run_metadata.json
  two_stage_pipeline_metadata.json
  pipeline_logs/
  00_wide_current_and_new_set/
    current_reference/
    new_set/
    wide_capture_metadata.json
  01_ptz_overlap_grid/
    raw/
    preview_center_marker/
    ptz_capture_metadata.json
  02_aliked_lightglue_wide_ptz_mapping/
    features/
    matches/
    diagnostics/
    wide_pixel_to_ptz_mapping.json
  03_polynomial_both_stages/
    polynomial_both_stages_mapping.json
    global_model_comparison.json
    local_pair_comparison.csv
    diagnostics/
  04_execute_polynomial_30_points/   # chỉ có khi dùng --execute
```

## Các file chính

- `collect_full_dynamic_pipeline.py`: capture Wide và lưới PTZ không biết
  trước giới hạn.
- `build_wide_ptz_mapping.py`: ALIKED, LightGlue, MAGSAC và H nền; lưu match
  để tầng Polynomial dùng lại.
- `fit_polynomial_both_stages.py`: fit Polynomial cả local và global.
- `fit_polynomial_wide_to_grid.py`: thư viện Polynomial, RANSAC, PCHIP và
  grid pose surface.
- `run_two_stage_pipeline.py`: entrypoint một lệnh.
- `capture_onvif.py`: bản helper ONVIF/RTSP được đóng gói cục bộ, không cần
  import từ `RAY_WORLD`.

## Ghi chú mô hình

Polynomial hai tầng là mô hình thực nghiệm cho dữ liệu hiện tại. Nó không thay
thế mô hình quang học fisheye/ray nếu Wide có méo rất mạnh hoặc PTZ zoom thay
đổi lớn. Trước khi dùng vận hành, nên kiểm tra các file residual và preview;
`--execute` chỉ là bước validation thực tế.
