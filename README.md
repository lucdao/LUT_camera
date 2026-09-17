# RAY World — PTZ pixel-motion Polynomial pipeline

Đây là repository độc lập cho bài toán camera không biết trước mô hình. Luồng
được khuyến nghị hiện tại là:

```text
30 ảnh PTZ + metadata + 30 H PTZ→Wide
              │
              ▼
  tạo mẫu sai số pixel → delta pan/tilt
              │
              ▼
  fit Polynomial bậc 2 không intercept
              │
              ▼
  pixel bất kỳ trên ảnh PTZ → lệnh AbsoluteMove
```

Luồng online không dùng SIFT, AI, epipolar line, calibration hay hình học 3-D.
Các đặc trưng ALIKED/LightGlue và RANSAC chỉ được dùng offline để tạo 30 ma
trận H ban đầu. Pipeline Polynomial hai tầng Wide→grid cũ vẫn được giữ lại để
tái lập/so sánh dữ liệu, nhưng không phải là công thức đang dùng trong bài
kiểm thử centre-only case 10.

## 1. Cài đặt

```powershell
cd C:\thuctap\RAY_WORLD_POLYNOMIAL_TWO_STAGE
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Không ghi mật khẩu vào source, JSON hoặc Git. Đặt trong PowerShell session:

```powershell
$env:PTZ_PASSWORD = "mat-khau-camera"
$env:WIDE_RTSP_URL = "rtsp://user:password@wide-host:554/path"
# RTSP PTZ mặc định là rtsp://user:password@ptz-host:554/0.
# Nếu khác, truyền --rtsp-url ở lệnh batch.
```

## 2. Dữ liệu đầu vào

`--mapping-dir` phải trỏ tới thư mục chứa `selected_homographies.json`. File này
phải có 30 phần tử `selected`; mỗi phần tử cần có:

- `ptz_index`, `row`, `column`;
- `H_ptz_to_wide` kích thước 3×3;
- `actual_after_capture.pan`, `.tilt`, `.zoom`;
- đường dẫn ảnh PTZ và ảnh Wide (`ptz_image`/`raw_image`, `wide_image`).

Đường dẫn tương đối được hiểu theo run root của bộ mapping. Kích thước ảnh được
đọc trực tiếp từ file, không hard-code ngoài dữ liệu model.

## 3. Fit Polynomial bậc 2 từ 30 H

Chạy offline, không kết nối và không quay camera:

```powershell
python .\build_ptz_pixel_motion_relation.py `
  --mapping-dir C:\path\to\02_aliked_lightglue_wide_ptz_mapping `
  --output-dir C:\path\to\ptz_motion_relation
```

Hoặc dùng wrapper:

```powershell
.\run_ptz_pixel_motion_pipeline.ps1 `
  -MappingDir C:\path\to\02_aliked_lightglue_wide_ptz_mapping `
  -OutputDir C:\path\to\ptz_motion_relation `
  -Plan10
```

Wrapper chỉ fit model và tùy chọn lập manifest 10 điểm; không điều khiển camera.

### Công thức tạo mẫu

Gọi `H_i` là H của ảnh PTZ ở trạng thái `i`, và `c` là tâm ảnh PTZ. Với mọi cặp
trạng thái nguồn `i` và đích `j`:

```text
w_j = H_j c
p_i = H_i⁻¹ w_j
e_i = c - p_i
Δq_ij = [pan_j - pan_i, tilt_j - tilt_i]
```

Mẫu huấn luyện là:

```text
(e_i.x, e_i.y) → (Δpan, Δtilt)
```

Chỉ giữ mẫu nếu điểm Wide và điểm PTZ ngược đều nằm trong biên ảnh, có margin
20 pixel. Với bộ hiện tại tạo được 257 mẫu centre-goal. Ngoài ra script lưu
8046 mẫu chuyển điểm bất kỳ giữa các PTZ pose vào `ptz_pair_transfer_samples.*`
để kiểm tra hình học; các mẫu này không được đưa vào fit centre-goal hiện tại.

### Công thức online

Với ảnh PTZ kích thước `W×H`, pixel chọn `(u,v)` và tâm `c=(W/2,H/2)`:

```text
ex = (W/2 - u) / (W/2)
ey = (H/2 - v) / (H/2)
φ  = [ex, ey, ex², ex·ey, ey²]
```

Mô hình bậc 2 không intercept:

```text
[Δpan, Δtilt] = φ B
```

Trong đó `B` được fit bằng `numpy.linalg.lstsq`. Không có intercept vì `φ=0`
phải tương ứng với không dịch chuyển. Trạng thái gửi tới camera là:

```text
pan_goal  = pan_actual  + Δpan
tilt_goal = tilt_actual + Δtilt
zoom_goal = zoom_actual
```

Model hiện tại chỉ được huấn luyện ở `zoom=0`; chưa được phép suy luận cho zoom
khác. Các hệ số thật và sai số LOO luôn đọc từ
`ptz_motion_relation_models.json`, không chép thủ công vào source.

## 4. Kiểm tra offline model

```powershell
$map = "C:\path\to\02_aliked_lightglue_wide_ptz_mapping"
$out = "C:\path\to\ptz_motion_relation"

python .\build_ptz_pixel_motion_relation.py `
  --mapping-dir $map `
  --output-dir $out

Get-Content "$out\ptz_motion_relation_models.json" | ConvertFrom-Json |
  Select-Object -ExpandProperty models
```

Model hiện tại được chọn là `quadratic_no_intercept`. Trong bộ dữ liệu đã kiểm
tra trước đây, RMSE LOO vector của tuyến tính khoảng `0.01936`, còn bậc 2 khoảng
`0.01439` theo đơn vị pan/tilt chuẩn hóa của camera.

## 5. Lập 10 ca kiểm thử

```powershell
python .\plan_ptz_center_only_batch.py `
  --relation-dir C:\path\to\ptz_motion_relation `
  --mapping-dir C:\path\to\02_aliked_lightglue_wide_ptz_mapping `
  --output C:\path\to\ptz_motion_relation\center_only_batch_manifest.json
```

Manifest gồm 5 điểm được dự đoán ngoài khung Wide và 5 điểm control trong
khung Wide, nếu các điểm đó vẫn nằm trong giới hạn điều khiển camera. Nhãn
`outside_wide` chỉ là phép chiếu kiểm tra qua H; nó không có nghĩa camera PTZ
có thể nhìn thấy toàn bộ vùng ngoài Wide.

## 6. Kiểm thử một pixel thật — không SIFT

Lệnh sau có điều khiển camera thật và luôn cố gắng khôi phục trạng thái ban đầu:

```powershell
python .\run_ptz_pixel_center_only_test.py `
  --relation-dir C:\path\to\ptz_motion_relation `
  --mapping-dir C:\path\to\02_aliked_lightglue_wide_ptz_mapping `
  --source-index 0 `
  --u 800 --v 720 `
  --host 192.168.1.8 --port 80 `
  --user admin --password $env:PTZ_PASSWORD `
  --position-tolerance 0.01 `
  --output-dir C:\path\to\live_case
```

Ảnh nguồn đánh dấu tâm và pixel chọn. Ảnh đích chỉ đánh dấu tâm PTZ theo yêu
cầu; không dùng SIFT để đánh dấu lại vật thể. JSON kết quả ghi cả trạng thái
yêu cầu, trạng thái camera thực tế, delta, thời gian tính toán và trạng thái
khôi phục.

## 7. Kiểm thử batch 10 ca — không SIFT

```powershell
$manifest = "C:\path\to\ptz_motion_relation\center_only_batch_manifest.json"
$map = "C:\path\to\02_aliked_lightglue_wide_ptz_mapping"
$rel = "C:\path\to\ptz_motion_relation"

python .\run_ptz_pixel_center_only_batch.py `
  --manifest $manifest `
  --relation-dir $rel `
  --mapping-dir $map `
  --host 192.168.1.8 --port 80 `
  --user admin --password $env:PTZ_PASSWORD `
  --rtsp-url "rtsp://admin:$env:PTZ_PASSWORD@192.168.1.8:554/0" `
  --position-tolerance 0.01 `
  --output-dir C:\path\to\center_only_batch_10
```

Không chạy lệnh này nếu chưa chắc chắn camera được phép quay. Kết quả nằm ở
`batch_result.json` và từng thư mục `case_01`...`case_10`.

## 8. Xử lý trạng thái camera không đạt yêu cầu

Đây là điểm quan trọng với camera hiện tại. Mọi lệnh live đều:

1. gửi `AbsoluteMove`;
2. đọc `GetStatus` lặp lại;
3. chỉ chụp khi pan/tilt/zoom nằm trong `--position-tolerance` ở nhiều mẫu ổn
   định liên tiếp;
4. lưu `source_actual_position` và `actual_destination_position`.

Mặc định của test centre-only là `0.01`, nhỏ hơn ngưỡng cũ `0.035`. Nếu camera
không thể đạt sai số này, pipeline dừng thay vì lấy trạng thái gần đúng để tính
delta. Với camera báo tilt dương bằng giá trị đặc thù `2.81111121`, helper
chuẩn hóa về tilt đã truyền và ghi rõ trong metadata.

Không được thay `source_actual_position` bằng trạng thái yêu cầu khi tính delta:
delta phải cộng vào trạng thái thực tế camera đã đạt. Nếu trạng thái nguồn lệch
lớn, cần dừng/điều chỉnh việc settle; nếu vẫn cố chạy, sai lệch sẽ truyền sang
trạng thái đích như đã thấy ở case 10.

## 9. Các file chính

- `build_ptz_pixel_motion_relation.py`: tạo mẫu từ 30 H và fit linear/quadratic;
  model vận hành là Polynomial bậc 2.
- `test_ptz_motion_relation_live.py`: công thức dự đoán và helper chụp ONVIF;
  có thêm các hàm SIFT chỉ để tương thích test cũ.
- `run_ptz_pixel_center_only_test.py`: kiểm thử một pixel, không SIFT.
- `run_ptz_pixel_center_only_batch.py`: kiểm thử 10 ca, không SIFT.
- `plan_ptz_center_only_batch.py`: tạo manifest 5 ngoài Wide + 5 control.
- `capture_onvif.py`: ONVIF, RTSP, chuẩn hóa tilt và xác nhận settle.
- `run_ptz_pixel_motion_pipeline.ps1`: wrapper fit offline và lập manifest.
- `run_two_stage_pipeline.py`: pipeline cũ thu thập dữ liệu và fit Polynomial
  local/global Wide→grid; giữ lại để tái tạo bộ 30 H khi cần.

## 10. Giới hạn phương pháp

Đây là mô hình thực nghiệm từ 30 pose, không phải mô hình quang học tổng quát.
Nó không tự chứng minh rằng vật thể sau khi quay đã nằm đúng tâm, vì bản
centre-only cố ý không theo dõi vật thể trong ảnh đích. Nó cũng chưa mô hình
hóa zoom, không bảo đảm ngoại suy xa khỏi vùng pose đã thu thập và không thay
thế calibration khi yêu cầu độ chính xác đo lường.
